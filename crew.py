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
import requests
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# Workspace for generated files
WORKSPACE_DIR = Path.home() / ".council" / "crewai-council" / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


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
# PHASE 2: Provider Status Check
# =============================================================================

def check_providers() -> dict[str, bool]:
    """Check which providers are available at startup."""
    status = {}

    # Check Anthropic
    status["anthropic"] = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # Check OpenAI
    status["openai"] = bool(os.environ.get("OPENAI_API_KEY"))

    # Check Ollama
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = [m["name"] for m in resp.json().get("models", [])]
        status["ollama"] = any("deepseek" in m for m in models)
    except Exception:
        status["ollama"] = False

    return status


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
        You provide specific, actionable feedback with code examples.""",
        llm="anthropic/claude-opus-4-5-20251101",
        verbose=True,
        allow_delegation=False
    )

    # DeepSeek - focuses on performance and efficiency
    reviewer_deepseek = Agent(
        role="Code Reviewer (DeepSeek)",
        goal="Review code for performance issues, inefficiencies, and optimization opportunities",
        backstory="""You are a performance-focused code reviewer.
        You identify slow algorithms, memory issues, and optimization opportunities.
        You suggest concrete performance improvements with benchmarks when possible.""",
        llm=LLM(model="ollama/deepseek-coder-v2:16b", base_url="http://localhost:11434"),
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
    architect_claude, architect_deepseek, builder, reviewer_claude, reviewer_deepseek = create_agents()
    tasks = create_tasks(architect_claude, architect_deepseek, builder, reviewer_claude, reviewer_deepseek, task_description)

    crew = Crew(
        agents=[architect_claude, architect_deepseek, builder, reviewer_claude, reviewer_deepseek],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    return crew


def main():
    if len(sys.argv) < 2:
        print("Usage: python crew.py <task description>")
        print('Example: python crew.py "Build a REST API for user authentication"')
        sys.exit(1)

    task_description = " ".join(sys.argv[1:]).strip()

    if not task_description:
        print("Error: Task description cannot be empty")
        print("Usage: python crew.py <task description>")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("CREWAI COUNCIL (Dual-Perspective)")
    print(f"{'='*60}")

    # Phase 2: Check provider status
    status = check_providers()
    print_provider_status(status)

    if not all(status.values()):
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

    crew = create_crew(task_description)
    result = crew.kickoff()

    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    print(result)

    return result


if __name__ == "__main__":
    main()
