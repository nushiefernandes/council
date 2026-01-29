#!/usr/bin/env python3
"""
CrewAI-based multi-model council with Claude, GPT-5.2, and DeepSeek.

Dual-Perspective Design:
- 2 Architects: Claude (clean design) + DeepSeek (performance)
- 1 Builder: GPT-5.2 (implementation)
- 2 Reviewers: Claude (security) + DeepSeek (performance)

Prerequisites:
- ANTHROPIC_API_KEY set
- OPENAI_API_KEY set
- Ollama running with deepseek-coder-v2:16b pulled

Usage:
    python crew.py "Your task description here"

Or activate the venv first:
    source ~/.council/crewai-council/venv/bin/activate
    python crew.py "Build a REST API for user authentication"
"""
import sys
import os
import json
import argparse
import requests
import subprocess
import time
import shutil
import fcntl
import tempfile
import atexit
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# Workspace for generated files
WORKSPACE_DIR = Path.home() / ".council" / "crewai-council" / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# OLLAMA AUTO-START
# =============================================================================

# Global state for Ollama process management
_OLLAMA_PID_FILE = Path(tempfile.gettempdir()) / "crewai_ollama.pid"
_OLLAMA_LOCK_FILE = Path(tempfile.gettempdir()) / "crewai_ollama.lock"
_started_ollama_pid: int | None = None

def _check_ollama_quick(timeout: float = 5.0, verify_model: str | None = None) -> bool:
    """Quick check if Ollama is responding, optionally verify model exists."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return False
        if verify_model:
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(verify_model in m for m in models)
        return True
    except Exception:
        return False


def ensure_ollama_running(max_wait: int = 30, verify_model: str = "deepseek") -> bool:
    """
    Start Ollama if not running, wait until ready.
    Uses file locking to prevent race conditions across processes.

    Args:
        max_wait: Maximum seconds to wait for Ollama to start
        verify_model: Model name substring to verify (default: "deepseek")

    Returns:
        True if Ollama is running with required model, False otherwise
    """
    global _started_ollama_pid

    # Quick check if already running with correct model
    if _check_ollama_quick(verify_model=verify_model):
        return True

    # Acquire lock to prevent race condition
    lock_fd = None
    try:
        lock_fd = open(_OLLAMA_LOCK_FILE, 'w')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        # Another process is starting Ollama - wait for it
        print("Another process is starting Ollama, waiting...")
        if lock_fd:
            lock_fd.close()
        return _wait_for_ollama(max_wait, verify_model)

    try:
        # Double-check after acquiring lock
        if _check_ollama_quick(verify_model=verify_model):
            return True

        print("Ollama not running. Attempting to start...")

        # Find ollama binary securely
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            print("Error: 'ollama' command not found. Install from https://ollama.ai")
            return False

        # Start Ollama
        try:
            proc = subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            _started_ollama_pid = proc.pid
            _OLLAMA_PID_FILE.write_text(str(proc.pid))
        except Exception as e:
            print(f"Error starting Ollama: {e}")
            return False

        # Wait for ready
        return _wait_for_ollama(max_wait, verify_model)

    finally:
        if lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()


def _wait_for_ollama(max_wait: int, verify_model: str | None) -> bool:
    """Wait for Ollama to become ready."""
    start = time.time()
    while time.time() - start < max_wait:
        if _check_ollama_quick(verify_model=verify_model):
            print(f"✓ Ollama ready ({time.time() - start:.1f}s)")
            return True
        time.sleep(1)

    print(f"Ollama failed to start within {max_wait}s")
    _cleanup_ollama()
    return False


def _cleanup_ollama():
    """Clean up Ollama process if we started it."""
    global _started_ollama_pid
    if _started_ollama_pid:
        try:
            import signal
            os.kill(_started_ollama_pid, signal.SIGTERM)
            print(f"Cleaned up Ollama process {_started_ollama_pid}")
        except ProcessLookupError:
            pass  # Already dead
        except Exception as e:
            print(f"Warning: Could not clean up Ollama: {e}")
        _started_ollama_pid = None
    _OLLAMA_PID_FILE.unlink(missing_ok=True)


# Register cleanup handler
atexit.register(_cleanup_ollama)


# =============================================================================
# PHASE 2: Structured Output Schemas
# =============================================================================

class Issue(BaseModel):
    """A single issue found during code review."""
    severity: Literal["critical", "major", "minor"]
    file: str
    description: str
    suggestion: str


class ReviewResult(BaseModel):
    """Structured output from a code review."""
    issues: list[Issue]
    approved: bool
    summary: str


# =============================================================================
# PHASE 2: File Writing Tool
# =============================================================================

@tool("write_file")
def write_file(filename: str, content: str) -> str:
    """
    Write content to a file in the workspace directory.

    Args:
        filename: Name of the file to create (e.g., 'main.py', 'utils/helpers.py')
        content: The content to write to the file

    Returns:
        Confirmation message with file path
    """
    # Security: Reject absolute paths
    if filename.startswith('/') or (len(filename) > 1 and filename[1] == ':'):
        raise ValueError(f"Absolute paths not allowed: {filename}")

    # Security: Reject path traversal attempts
    if '..' in filename:
        raise ValueError(f"Path traversal not allowed: {filename}")

    # Construct the target path
    file_path = WORKSPACE_DIR / filename

    # Security: Resolve the path and verify it's still within workspace
    # This catches symlink attacks and any other path manipulation
    try:
        resolved_path = file_path.resolve()
        workspace_resolved = WORKSPACE_DIR.resolve()

        # Check that the resolved path is within the workspace
        if not str(resolved_path).startswith(str(workspace_resolved)):
            raise ValueError(f"Path escapes workspace: {filename}")
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid path: {filename} ({e})")

    # Create parent directories (within workspace only)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Final safety check after mkdir (in case of race conditions)
    resolved_path = file_path.resolve()
    if not str(resolved_path).startswith(str(workspace_resolved)):
        raise ValueError(f"Path escapes workspace after resolution: {filename}")

    file_path.write_text(content)
    return f"✓ Wrote {len(content)} chars to {file_path}"


# =============================================================================
# SKILL INTEGRATION: Allow agents to query Claude Code skills
# =============================================================================

SKILLS_DIR = Path.home() / ".claude" / "skills"
ALLOWED_SKILLS = {
    "postgres-best-practices",
    "react-best-practices",
    "web-design-guidelines",
}


@tool("query_skill")
def query_skill(skill_name: str, query: str) -> str:
    """
    Query a Claude Code skill for best practices and guidelines.

    Use this to get expert guidance on specific topics. Available skills:
    - postgres-best-practices: Database optimization, indexes, queries
    - react-best-practices: React/Next.js patterns, performance, hooks
    - web-design-guidelines: UI/UX, accessibility, design systems

    Args:
        skill_name: One of the available skill names listed above
        query: What you want to look up (e.g., "index optimization", "React hooks")

    Returns:
        Relevant guidance from the skill documentation
    """
    if skill_name not in ALLOWED_SKILLS:
        return f"Error: Unknown skill '{skill_name}'. Available: {', '.join(ALLOWED_SKILLS)}"

    skill_path = SKILLS_DIR / skill_name

    if not skill_path.exists():
        return f"Error: Skill '{skill_name}' not found at {skill_path}"

    # Try to read AGENTS.md first (compiled for agents), then SKILL.md
    agents_file = skill_path / "AGENTS.md"
    skill_file = skill_path / "SKILL.md"

    if agents_file.exists():
        content = agents_file.read_text()
    elif skill_file.exists():
        content = skill_file.read_text()
    else:
        return f"Error: No readable content in skill '{skill_name}'"

    # Simple keyword search - return matching lines
    query_lower = query.lower()
    lines = content.split('\n')
    relevant = []

    for i, line in enumerate(lines):
        if query_lower in line.lower():
            # Include surrounding context (2 lines before/after)
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            context = '\n'.join(lines[start:end])
            if context not in relevant:
                relevant.append(context)

    if relevant:
        result = f"From {skill_name} (query: '{query}'):\n\n"
        result += "\n---\n".join(relevant[:5])  # Limit to 5 matches
        return result
    else:
        # Return a summary of the skill if no matches
        return f"No matches for '{query}' in {skill_name}. Skill contains {len(lines)} lines of guidance on {skill_name.replace('-', ' ')}."


# =============================================================================
# PHASE 2: Provider Status Check
# =============================================================================

def check_providers() -> dict[str, bool]:
    """Check which providers are available at startup."""
    status = {}

    # Check Anthropic - validate key exists and has reasonable length
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    status["anthropic"] = bool(anthropic_key) and len(anthropic_key) > 20

    # Check OpenAI - validate key exists and has reasonable length
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    status["openai"] = bool(openai_key) and len(openai_key) > 20

    # Check Ollama with retry logic
    status["ollama"] = _check_ollama_with_retry()

    return status


def _check_ollama_with_retry(retries: int = 2, initial_timeout: float = 2.0) -> bool:
    """Check Ollama availability with exponential backoff retry."""
    for attempt in range(retries + 1):
        timeout = initial_timeout * (2 ** attempt)  # 2s, 4s, 8s
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=timeout)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                return any("deepseek" in m for m in models)
        except requests.Timeout:
            pass  # Expected when Ollama is slow/starting
        except requests.ConnectionError:
            pass  # Expected when Ollama not running
        except Exception as e:
            # Log unexpected errors but don't crash
            print(f"Warning: Unexpected error checking Ollama: {e}", file=sys.stderr)
    return False


def print_provider_status(status: dict[str, bool]) -> bool:
    """Print provider status and return True if all available."""
    print("\nProvider Status:")
    all_ok = True
    for provider, available in status.items():
        icon = "✓" if available else "✗"
        print(f"  {icon} {provider}")
        if not available:
            all_ok = False

    if not all_ok:
        print("\n⚠️  Some providers unavailable. Council may not function correctly.")

    return all_ok


# =============================================================================
# PHASE 3: Checkpoint System
# =============================================================================

def checkpoint(stage: str, summary: str) -> bool:
    """
    Pause execution and ask user for approval.

    Args:
        stage: Name of the completed stage (e.g., "PLANNING", "BUILDING")
        summary: Brief summary of what was accomplished

    Returns:
        True if user approves, False otherwise (exits on abort)
    """
    print(f"\n{'='*60}")
    print(f"CHECKPOINT: {stage} COMPLETE")
    print(f"{'='*60}")
    print(f"\n{summary[:2000]}")  # Truncate very long summaries

    while True:
        try:
            response = input("\n[A]pprove and continue, [R]eject and abort? [A/r]: ").strip().lower()
            if response in ('', 'a', 'approve', 'yes', 'y'):
                print("✓ Approved. Continuing to next stage...")
                return True
            elif response in ('r', 'reject', 'no', 'n', 'abort'):
                print("✗ Rejected. Aborting session.")
                sys.exit(0)
            else:
                print("Please enter 'A' to approve or 'R' to reject.")
        except (EOFError, KeyboardInterrupt):
            print("\n✗ Session interrupted. Aborting.")
            sys.exit(0)


def create_agents():
    """Create the council agents with different LLMs (dual-perspective design)."""

    # === ARCHITECTS (Dual Perspective) ===

    # Claude Opus 4.5 - focuses on clean architecture
    architect_claude = Agent(
        role="Software Architect (Claude)",
        goal="Design clean, maintainable system architecture and make key technical decisions",
        backstory="""You are a senior software architect with 15+ years of experience.
        You focus on clean architecture, separation of concerns, and pragmatic design decisions.
        You prefer simple solutions that are easy to understand and maintain.""",
        llm="anthropic/claude-opus-4-5-20251101",
        verbose=True,
        allow_delegation=False
    )

    # DeepSeek - focuses on performance and scalability
    architect_deepseek = Agent(
        role="Software Architect (DeepSeek)",
        goal="Review architecture for performance, scalability, and practical implementation concerns",
        backstory="""You are an architect focused on performance and scalability.
        You review designs for efficiency, identify potential bottlenecks, and suggest optimizations.
        You ensure the architecture will perform well under load.""",
        llm=LLM(model="ollama/deepseek-coder-v2:16b", base_url="http://localhost:11434"),
        verbose=True,
        allow_delegation=False
    )

    # === BUILDER ===

    # GPT 5.2 as Builder (with file writing tool)
    builder = Agent(
        role="Backend Developer",
        goal="Implement features following the architecture with clean, tested code",
        backstory="""You are a pragmatic backend developer who writes clean, efficient code.
        You follow best practices, write comprehensive tests, and document your work.
        You implement exactly what's specified, no more, no less.
        IMPORTANT: Use the write_file tool to save your code to actual files.""",
        llm="openai/gpt-5.2",
        tools=[write_file],
        verbose=True,
        allow_delegation=False
    )

    # === REVIEWERS (Dual Perspective) ===

    # Claude Opus 4.5 - focuses on security and edge cases
    reviewer_claude = Agent(
        role="Code Reviewer (Claude)",
        goal="Review code for security vulnerabilities, edge cases, and maintainability",
        backstory="""You are a security-focused code reviewer who catches subtle bugs.
        You focus on security vulnerabilities, error handling, and edge cases.
        You provide specific, actionable feedback with code examples.
        Use query_skill to check best practices when reviewing.""",
        llm="anthropic/claude-opus-4-5-20251101",
        tools=[query_skill],
        verbose=True,
        allow_delegation=False
    )

    # DeepSeek - focuses on performance and efficiency
    reviewer_deepseek = Agent(
        role="Code Reviewer (DeepSeek)",
        goal="Review code for performance issues, inefficiencies, and optimization opportunities",
        backstory="""You are a performance-focused code reviewer.
        You identify slow algorithms, memory issues, and optimization opportunities.
        You suggest concrete performance improvements with benchmarks when possible.
        Use query_skill to check postgres/react best practices when relevant.""",
        llm=LLM(model="ollama/deepseek-coder-v2:16b", base_url="http://localhost:11434"),
        tools=[query_skill],
        verbose=True,
        allow_delegation=False
    )

    return architect_claude, architect_deepseek, builder, reviewer_claude, reviewer_deepseek


def create_tasks(architect_claude, architect_deepseek, builder, reviewer_claude, reviewer_deepseek, task_description: str):
    """Create tasks for the dual-perspective workflow."""

    # === PLANNING PHASE (Dual Perspective) ===

    # Task 1: Claude designs the architecture
    plan_task_claude = Task(
        description=f"""Design the architecture for the following task:

{task_description}

Provide:
1. Overview of the solution
2. Key components and their responsibilities
3. Data flow between components
4. API contracts (if applicable)
5. File structure recommendation

Keep it simple and pragmatic. No over-engineering.""",
        expected_output="""A clear architecture document with:
- Solution overview
- Component breakdown
- Data flow
- File structure
- Key technical decisions and rationale""",
        agent=architect_claude
    )

    # Task 2: DeepSeek reviews and enhances with performance focus
    plan_task_deepseek = Task(
        description="""Review the architecture proposal from the previous task.

Your focus areas:
1. Performance implications of the design choices
2. Scalability concerns
3. Potential bottlenecks
4. Memory and resource usage
5. Suggestions for optimization

Enhance the architecture with performance considerations. Keep the good parts, suggest improvements where needed.""",
        expected_output="""Enhanced architecture review with:
- Agreement/disagreement with key decisions
- Performance analysis
- Scalability notes
- Specific optimization suggestions
- Final recommended architecture (incorporating both perspectives)""",
        agent=architect_deepseek,
        context=[plan_task_claude]
    )

    # === BUILD PHASE ===

    # Task 3: GPT-5.2 implements (sees both architecture perspectives)
    build_task = Task(
        description="""Implement the architecture from the previous tasks.

You have received TWO architecture perspectives:
1. Claude's design (clean architecture focus)
2. DeepSeek's enhancements (performance focus)

Incorporate the best of both in your implementation.

Requirements:
1. Write clean, well-documented code
2. Include error handling
3. Follow the agreed file structure
4. Include type hints (if Python)
5. Apply the performance optimizations suggested

IMPORTANT: Use the write_file tool to save each file. Do NOT just output code blocks.
For each file, call write_file(filename="path/to/file.py", content="...code...")""",
        expected_output="""Working code implementation with:
- All files written using the write_file tool
- Proper error handling
- Type hints
- Performance optimizations applied
- Summary of files created""",
        agent=builder,
        context=[plan_task_claude, plan_task_deepseek]
    )

    # === REVIEW PHASE (Dual Perspective) ===

    # Task 4: Claude reviews for security and edge cases
    review_task_claude = Task(
        description="""Review the implementation for SECURITY and CORRECTNESS.

Your focus areas:
1. Security vulnerabilities (injection, auth issues, data exposure)
2. Edge cases and error handling
3. Input validation
4. Race conditions or concurrency issues
5. Maintainability concerns

Provide specific, actionable feedback with code examples for fixes.""",
        expected_output="""Security-focused code review with:
- Security issues found (severity: critical/major/minor)
- Edge cases identified
- Specific fixes with code examples
- Overall security assessment""",
        agent=reviewer_claude,
        context=[build_task]
    )

    # Task 5: DeepSeek reviews for performance
    review_task_deepseek = Task(
        description="""Review the implementation for PERFORMANCE and EFFICIENCY.

Your focus areas:
1. Algorithm efficiency (time complexity)
2. Memory usage and potential leaks
3. Database query optimization (if applicable)
4. Unnecessary computations or redundant operations
5. Caching opportunities

Provide specific, actionable feedback with benchmarks or complexity analysis where possible.""",
        expected_output="""Performance-focused code review with:
- Performance issues found (severity: critical/major/minor)
- Complexity analysis
- Optimization suggestions with code examples
- Overall performance assessment""",
        agent=reviewer_deepseek,
        context=[build_task]
    )

    return [plan_task_claude, plan_task_deepseek, build_task, review_task_claude, review_task_deepseek]


def create_crew(task_description: str):
    """Create and return the crew with dual-perspective agents."""
    # Validate task description
    if not task_description or not task_description.strip():
        raise ValueError("Task description cannot be empty")

    architect_claude, architect_deepseek, builder, reviewer_claude, reviewer_deepseek = create_agents()
    tasks = create_tasks(architect_claude, architect_deepseek, builder, reviewer_claude, reviewer_deepseek, task_description)

    crew = Crew(
        agents=[architect_claude, architect_deepseek, builder, reviewer_claude, reviewer_deepseek],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    return crew


def run_with_checkpoints(task_description: str):
    """Run the council with checkpoints between stages."""
    architect_claude, architect_deepseek, builder, reviewer_claude, reviewer_deepseek = create_agents()

    # === STAGE 1: PLANNING ===
    print(f"\n{'='*60}")
    print("STAGE 1: PLANNING")
    print(f"{'='*60}\n")

    plan_task_claude = Task(
        description=f"""Design the architecture for the following task:

{task_description}

Provide:
1. Overview of the solution
2. Key components and their responsibilities
3. Data flow between components
4. API contracts (if applicable)
5. File structure recommendation

Keep it simple and pragmatic. No over-engineering.""",
        expected_output="""A clear architecture document with:
- Solution overview
- Component breakdown
- Data flow
- File structure
- Key technical decisions and rationale""",
        agent=architect_claude
    )

    plan_task_deepseek = Task(
        description="""Review the architecture proposal from the previous task.

Your focus areas:
1. Performance implications of the design choices
2. Scalability concerns
3. Potential bottlenecks
4. Memory and resource usage
5. Suggestions for optimization

Enhance the architecture with performance considerations.""",
        expected_output="""Enhanced architecture review with:
- Agreement/disagreement with key decisions
- Performance analysis
- Scalability notes
- Specific optimization suggestions
- Final recommended architecture""",
        agent=architect_deepseek,
        context=[plan_task_claude]
    )

    planning_crew = Crew(
        agents=[architect_claude, architect_deepseek],
        tasks=[plan_task_claude, plan_task_deepseek],
        process=Process.sequential,
        verbose=True
    )

    planning_result = planning_crew.kickoff()

    # Checkpoint after planning
    checkpoint(
        "PLANNING",
        f"Architecture designed by Claude and reviewed by DeepSeek.\n\nSummary:\n{str(planning_result)[:1500]}"
    )

    # === STAGE 2: BUILDING ===
    print(f"\n{'='*60}")
    print("STAGE 2: BUILDING")
    print(f"{'='*60}\n")

    build_task = Task(
        description=f"""Implement the architecture that was designed.

Original task: {task_description}

Architecture context (from planning phase):
{str(planning_result)[:3000]}

Requirements:
1. Write clean, well-documented code
2. Include error handling
3. Follow the agreed file structure
4. Include type hints (if Python)
5. Apply the performance optimizations suggested

IMPORTANT: Use the write_file tool to save each file. Do NOT just output code blocks.
For each file, call write_file(filename="path/to/file.py", content="...code...")""",
        expected_output="""Working code implementation with:
- All files written using the write_file tool
- Proper error handling
- Type hints
- Performance optimizations applied
- Summary of files created""",
        agent=builder
    )

    building_crew = Crew(
        agents=[builder],
        tasks=[build_task],
        process=Process.sequential,
        verbose=True
    )

    building_result = building_crew.kickoff()

    # List files created
    files_created = list(WORKSPACE_DIR.rglob("*"))
    files_list = "\n".join(f"  - {f.relative_to(WORKSPACE_DIR)}" for f in files_created if f.is_file())

    checkpoint(
        "BUILDING",
        f"Code implemented by GPT-5.2.\n\nFiles created:\n{files_list}\n\nSummary:\n{str(building_result)[:1000]}"
    )

    # === STAGE 3: REVIEW ===
    print(f"\n{'='*60}")
    print("STAGE 3: REVIEW")
    print(f"{'='*60}\n")

    review_task_claude = Task(
        description=f"""Review the implementation for SECURITY and CORRECTNESS.

Implementation context:
{str(building_result)[:2000]}

Your focus areas:
1. Security vulnerabilities (injection, auth issues, data exposure)
2. Edge cases and error handling
3. Input validation
4. Race conditions or concurrency issues
5. Maintainability concerns

Provide specific, actionable feedback with code examples for fixes.""",
        expected_output="""Security-focused code review with:
- Security issues found (severity: critical/major/minor)
- Edge cases identified
- Specific fixes with code examples
- Overall security assessment""",
        agent=reviewer_claude
    )

    review_task_deepseek = Task(
        description=f"""Review the implementation for PERFORMANCE and EFFICIENCY.

Implementation context:
{str(building_result)[:2000]}

Your focus areas:
1. Algorithm efficiency (time complexity)
2. Memory usage and potential leaks
3. Database query optimization (if applicable)
4. Unnecessary computations or redundant operations
5. Caching opportunities

Provide specific, actionable feedback with benchmarks or complexity analysis where possible.""",
        expected_output="""Performance-focused code review with:
- Performance issues found (severity: critical/major/minor)
- Complexity analysis
- Optimization suggestions with code examples
- Overall performance assessment""",
        agent=reviewer_deepseek
    )

    review_crew = Crew(
        agents=[reviewer_claude, reviewer_deepseek],
        tasks=[review_task_claude, review_task_deepseek],
        process=Process.sequential,
        verbose=True
    )

    review_result = review_crew.kickoff()

    checkpoint(
        "REVIEW",
        f"Code reviewed by Claude (security) and DeepSeek (performance).\n\nReview Summary:\n{str(review_result)[:1500]}"
    )

    return review_result


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="CrewAI Council - Multi-model AI deliberation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crew.py "Build a REST API for user authentication"
  python crew.py --checkpoint "Create a CLI tool"
        """
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Task description for the council"
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Pause for approval between stages (planning, building, review)"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip provider confirmation prompt (continue even if some providers unavailable)"
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Skip Ollama auto-start (use if you don't have DeepSeek)"
    )

    args = parser.parse_args()

    # Handle missing task
    if not args.task:
        parser.print_help()
        sys.exit(1)

    task_description = args.task.strip()
    if not task_description:
        print("Error: Task description cannot be empty")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("CREWAI COUNCIL (Dual-Perspective)")
    print(f"{'='*60}")

    if args.checkpoint:
        print("Mode: CHECKPOINT (will pause between stages)")
    else:
        print("Mode: CONTINUOUS (no pauses)")

    # Auto-start Ollama if needed (unless --no-ollama flag)
    if not args.no_ollama:
        ensure_ollama_running()

    # Check provider status
    status = check_providers()
    print_provider_status(status)

    if not all(status.values()):
        if args.yes:
            print("\n--yes flag: continuing despite missing providers")
        else:
            response = input("\nContinue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)

    print(f"\nTask: {task_description}")
    print(f"Workspace: {WORKSPACE_DIR}")
    print(f"\nAgents (5 total, 3 models):")
    print("  PLANNING:")
    print("    - Architect A (Claude Opus 4.5): Clean architecture design")
    print("    - Architect B (DeepSeek): Performance review")
    print("  BUILDING:")
    print("    - Builder (GPT-5.2): Implementation")
    print("  REVIEW:")
    print("    - Reviewer A (Claude Opus 4.5): Security & edge cases")
    print("    - Reviewer B (DeepSeek): Performance & efficiency")
    print(f"\n{'='*60}\n")

    # Run with or without checkpoints
    if args.checkpoint:
        result = run_with_checkpoints(task_description)
    else:
        crew = create_crew(task_description)
        result = crew.kickoff()

    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    print(result)

    return result


if __name__ == "__main__":
    main()
