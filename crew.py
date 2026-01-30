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
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# Workspace for generated files
WORKSPACE_DIR = Path.home() / ".council" / "crewai-council" / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Add workspace to Python path for skill_system imports
sys.path.insert(0, str(WORKSPACE_DIR))


# =============================================================================
# DELIBERATION STREAMING
# =============================================================================

DELIBERATION_LOG = WORKSPACE_DIR / "deliberation.jsonl"


def log_deliberation(agent_role: str, event: str, content: str, **metadata):
    """
    Log a deliberation event to JSON lines file for real-time streaming.

    Events can be monitored with: tail -f workspace/deliberation.jsonl | jq -c

    Args:
        agent_role: Name of the agent (e.g., "Software Architect (Claude)")
        event: Event type (e.g., "task_start", "step", "task_complete", "checkpoint")
        content: Event content/output (truncated to 1000 chars)
        **metadata: Additional fields to include in the log entry
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent_role,
        "event": event,
        "content": content[:1000] if content else "",
        **metadata
    }
    try:
        with open(DELIBERATION_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # Don't let logging failures break the main flow
        print(f"Warning: Failed to log deliberation event: {e}", file=sys.stderr)


def agent_step_callback(step_output):
    """
    Callback fired on each agent step for streaming.

    CrewAI calls this after each agent action, allowing real-time
    visibility into the deliberation process.
    """
    try:
        # Extract agent role from step output
        agent_role = getattr(step_output, 'agent', None)
        if hasattr(agent_role, 'role'):
            agent_role = agent_role.role
        else:
            agent_role = str(agent_role) if agent_role else "Unknown"

        # Extract output content
        output = getattr(step_output, 'output', None)
        content = str(output)[:500] if output else ""

        log_deliberation(
            agent_role=agent_role,
            event="step",
            content=content,
            step=getattr(step_output, 'step', None),
            thought=getattr(step_output, 'thought', None)[:200] if hasattr(step_output, 'thought') and step_output.thought else None
        )
    except Exception as e:
        # Don't let callback errors break execution
        print(f"Warning: Step callback error: {e}", file=sys.stderr)


def clear_deliberation_log():
    """Clear the deliberation log at the start of a new session."""
    try:
        DELIBERATION_LOG.unlink(missing_ok=True)
        log_deliberation(
            agent_role="system",
            event="session_start",
            content="New council session started"
        )
    except Exception as e:
        print(f"Warning: Could not clear deliberation log: {e}", file=sys.stderr)


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

def find_skill(skill_name: str) -> Path | None:
    """
    Find a skill by name across all skill locations.
    Returns path to SKILL.md or AGENTS.md, or None if not found.

    Priority: ~/.claude/skills/ > claude-code-plugins > claude-plugins-official
    """
    SKILLS_LOCATIONS = [
        (Path.home() / ".claude" / "skills", "direct"),
        (Path.home() / ".claude" / "plugins" / "marketplaces" / "claude-code-plugins" / "plugins", "nested"),
        (Path.home() / ".claude" / "plugins" / "marketplaces" / "claude-plugins-official" / "plugins", "nested"),
    ]

    for base_path, structure in SKILLS_LOCATIONS:
        if not base_path.exists():
            continue

        if structure == "direct":
            skill_dir = base_path / skill_name
        else:
            skill_dir = base_path / skill_name / "skills" / skill_name

        for filename in ["AGENTS.md", "SKILL.md"]:
            skill_file = skill_dir / filename
            if skill_file.exists():
                return skill_file

    return None


def extract_description(skill_file: Path, max_length: int = 100) -> str:
    """Extract description from skill file frontmatter or first paragraph."""
    try:
        content = skill_file.read_text()

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                for line in parts[1].split("\n"):
                    if line.startswith("description:"):
                        desc = line.replace("description:", "").strip().strip('"\'')
                        return desc[:max_length] + ("..." if len(desc) > max_length else "")

        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("---"):
                return line[:max_length] + ("..." if len(line) > max_length else "")

        return "No description available"
    except Exception:
        return "No description available"


@tool("list_skills")
def list_skills_tool() -> str:
    """
    List all available skills that can be queried.

    Returns skill names with brief descriptions. Use this before
    query_skill to discover what expertise is available for your task.
    """
    SKILLS_LOCATIONS = [
        (Path.home() / ".claude" / "skills", "direct"),
        (Path.home() / ".claude" / "plugins" / "marketplaces" / "claude-code-plugins" / "plugins", "nested"),
        (Path.home() / ".claude" / "plugins" / "marketplaces" / "claude-plugins-official" / "plugins", "nested"),
    ]

    skills = []
    seen_names = set()

    for base_path, structure in SKILLS_LOCATIONS:
        if not base_path.exists():
            continue

        try:
            for item in sorted(base_path.iterdir()):
                if not item.is_dir() or item.name.startswith("."):
                    continue

                skill_name = item.name
                if skill_name in seen_names:
                    continue

                skill_file = find_skill(skill_name)
                if not skill_file:
                    continue

                description = extract_description(skill_file)
                skills.append(f"  - {skill_name}: {description}")
                seen_names.add(skill_name)

        except PermissionError:
            continue

    if not skills:
        return "No skills found. Check ~/.claude/skills/ directory."

    header = f"Available skills ({len(skills)}):\n"

    if len(skills) > 25:
        return header + "\n".join(skills[:25]) + f"\n  ... and {len(skills) - 25} more"

    return header + "\n".join(skills)


@tool("query_skill")
def query_skill(skill_name: str, query: str) -> str:
    """
    Query a skill for relevant guidance.

    Use list_skills first to see available skills, then query specific ones.

    Args:
        skill_name: Name of the skill (from list_skills output)
        query: What you want to look up (e.g., "typography", "accessibility")

    Returns:
        Relevant guidance from the skill documentation
    """
    skill_file = find_skill(skill_name)

    if not skill_file:
        return f"Error: Skill '{skill_name}' not found. Run list_skills to see available skills."

    try:
        content = skill_file.read_text()
    except Exception as e:
        return f"Error reading skill '{skill_name}': {e}"

    query_lower = query.lower()
    lines = content.split('\n')
    relevant = []

    for i, line in enumerate(lines):
        if query_lower in line.lower():
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            context = '\n'.join(lines[start:end])
            if context not in relevant:
                relevant.append(context)

    if relevant:
        result = f"From {skill_name} (query: '{query}'):\n\n"
        result += "\n---\n".join(relevant[:5])
        if len(result) > 4000:
            result = result[:4000] + "\n... (truncated)"
        return result
    else:
        return f"No matches for '{query}' in {skill_name}. Try a different query or run list_skills to find other skills."


# =============================================================================
# SUBAGENT SPAWNING: Parallel research for architects and reviewers
# =============================================================================

# Global coordinator instance (lazy-initialized)
_subagent_coordinator = None


def _get_coordinator():
    """Get or create the global subagent coordinator."""
    global _subagent_coordinator
    if _subagent_coordinator is None:
        # Import subagent module from workspace
        subagent_path = WORKSPACE_DIR / "workspace" / "core" / "subagent"
        if subagent_path.exists():
            # Add the core directory to path so 'subagent' package can be imported
            core_path = subagent_path.parent
            if str(core_path) not in sys.path:
                sys.path.insert(0, str(core_path))
            from subagent import SubagentCoordinator, ResourceManager, TaskGraph, Memory

            _subagent_coordinator = SubagentCoordinator(
                task_graph=TaskGraph(max_workers=3),
                resource_manager=ResourceManager(),
                memory=Memory(),
                research_executor=_web_research_executor,
            )
        else:
            raise ImportError(f"Subagent module not found at {subagent_path}")
    return _subagent_coordinator


def _web_research_executor(query: str) -> tuple[str, list[str]]:
    """Execute a web search and return (content, sources).

    Uses DuckDuckGo search via subprocess for simplicity and no API key requirement.
    """
    import warnings

    try:
        # Try using ddgs (new package name) or duckduckgo-search (old name)
        DDGS = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    from ddgs import DDGS
                except ImportError:
                    from duckduckgo_search import DDGS
        except ImportError:
            pass

        if DDGS is not None:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                if results:
                    content_parts = []
                    sources = []
                    for r in results:
                        content_parts.append(f"**{r.get('title', 'Untitled')}**\n{r.get('body', '')}")
                        if r.get('href'):
                            sources.append(r['href'])
                    return "\n\n".join(content_parts), sources

        # Fallback: simple message indicating no search available
        return f"[Research query: {query}] - No search backend available. Install ddgs package for web research.", []

    except Exception as e:
        return f"Research failed for query '{query}': {str(e)}", []


@tool("spawn_research")
def spawn_research_tool(queries: str) -> str:
    """
    Spawn parallel research agents for multiple queries.

    Use this to research multiple topics simultaneously. The research runs
    in parallel (up to 3 concurrent queries) and results can be collected later.

    Args:
        queries: Comma-separated list of research queries
                 Example: "python async patterns, rust error handling, go channels"

    Returns:
        Comma-separated task IDs for collecting results later with collect_research
    """
    try:
        coordinator = _get_coordinator()
        query_list = [q.strip() for q in queries.split(",") if q.strip()]

        if not query_list:
            return "Error: No valid queries provided. Pass comma-separated queries."

        if len(query_list) > 5:
            return f"Error: Maximum 5 queries at once. You provided {len(query_list)}."

        task_ids = coordinator.spawn_research(query_list, timeout=120)
        return ",".join(task_ids)

    except ImportError as e:
        return f"Error: Subagent module not available - {e}"
    except Exception as e:
        return f"Error spawning research: {e}"


@tool("collect_research")
def collect_research_tool(task_ids: str) -> str:
    """
    Collect results from spawned research agents.

    Use this after spawn_research to retrieve the research findings.
    Will wait for all queries to complete (up to 120s timeout each).

    Args:
        task_ids: Comma-separated task IDs from spawn_research

    Returns:
        Aggregated research findings with sources
    """
    try:
        coordinator = _get_coordinator()
        id_list = [tid.strip() for tid in task_ids.split(",") if tid.strip()]

        if not id_list:
            return "Error: No valid task IDs provided."

        results = coordinator.collect_results(id_list, timeout=120)

        # Format results for agent consumption
        output_parts = []
        for task_id, result in results.items():
            if isinstance(result, dict) and "error" in result:
                output_parts.append(f"[{task_id}] Error: {result['error']}")
            elif hasattr(result, 'content'):
                # ResearchResult object
                sources_str = ", ".join(result.sources[:3]) if result.sources else "no sources"
                output_parts.append(f"[{task_id}] Query: {result.query}\n{result.content}\nSources: {sources_str}")
            else:
                output_parts.append(f"[{task_id}] {str(result)}")

        return "\n\n---\n\n".join(output_parts) if output_parts else "No results collected."

    except ImportError as e:
        return f"Error: Subagent module not available - {e}"
    except Exception as e:
        return f"Error collecting research: {e}"


# =============================================================================
# SKILL DISCOVERY: Search and install skills from GitHub
# =============================================================================

# Trusted sources that auto-approve skill installation
TRUSTED_SKILL_SOURCES = {"anthropics", "claude-skills-official"}


@tool("discover_skill")
def discover_skill_tool(query: str) -> str:
    """
    Search GitHub for Claude Code skills matching a query.

    Searches repositories with the 'claude-skill' topic. Returns skill info
    including name, description, author, stars, and trust status.

    Trusted sources (auto-approve): anthropics, claude-skills-official
    Community sources require explicit approval to install.

    Args:
        query: Search query (e.g., "postgres", "react", "testing")

    Returns:
        List of matching skills with metadata, or error message
    """
    try:
        from skill_system import discover_skill
        from skill_system.core import DiscoveryError

        skills = discover_skill(query, limit=10)

        if not skills:
            return f"No skills found for query '{query}'. Try a different search term."

        result = f"Found {len(skills)} skills for '{query}':\n\n"
        for i, skill in enumerate(skills, 1):
            trust_badge = "[TRUSTED]" if skill.is_trusted else "[community]"
            result += f"{i}. {skill.name} {trust_badge}\n"
            result += f"   Author: {skill.author} | Stars: {skill.stars}\n"
            result += f"   {skill.description[:100]}{'...' if len(skill.description) > 100 else ''}\n"
            result += f"   URL: {skill.repo_url}\n\n"

        result += "\nTo install a skill, use install_skill with the skill name and download URL."
        return result

    except ImportError as e:
        return f"Error: skill_system module not available ({e})"
    except Exception as e:
        return f"Error searching for skills: {e}"


@tool("install_skill")
def install_skill_tool(skill_name: str, download_url: str, source_owner: str, approve: bool = False) -> str:
    """
    Install a Claude Code skill from GitHub.

    Downloads the skill package, verifies security (size limits, blocked extensions,
    path traversal), and installs to ~/.claude/skills/.

    SECURITY:
    - Trusted sources (anthropics, claude-skills-official): auto-approved
    - Community sources: require approve=True parameter
    - Max package size: 5MB
    - Blocked extensions: .exe, .sh, .bat, .dll, .so, .dylib

    Args:
        skill_name: Name to register the skill under
        download_url: URL to the skill zip file (use repo_url + /archive/refs/heads/main.zip)
        source_owner: GitHub username/org that owns the repo
        approve: Set to True to approve community skill installation

    Returns:
        Installation result with success/failure and details
    """
    try:
        from skill_system import install_skill
        from skill_system.core import SecurityError, InstallError

        # Check if source is trusted
        is_trusted = source_owner.lower() in {s.lower() for s in TRUSTED_SKILL_SOURCES}

        if not is_trusted and not approve:
            return (
                f"Security: '{source_owner}' is not a trusted source.\n"
                f"Community skills require explicit approval.\n"
                f"Call install_skill with approve=True to proceed, or use a skill from a trusted source."
            )

        result = install_skill(
            name=skill_name,
            download_url=download_url,
            source_owner=source_owner,
            human_approved=approve,
        )

        if result.success:
            return f"✓ Successfully installed skill '{result.skill_name}' to {result.install_path}"
        else:
            return f"✗ Failed to install skill: {result.message}"

    except ImportError as e:
        return f"Error: skill_system module not available ({e})"
    except Exception as e:
        return f"Error installing skill: {e}"


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
    # Log checkpoint event
    log_deliberation(
        agent_role="system",
        event="checkpoint",
        content=summary[:500],
        stage=stage
    )

    print(f"\n{'='*60}")
    print(f"CHECKPOINT: {stage} COMPLETE")
    print(f"{'='*60}")
    print(f"\n{summary[:2000]}")  # Truncate very long summaries

    while True:
        try:
            response = input("\n[A]pprove and continue, [R]eject and abort? [A/r]: ").strip().lower()
            if response in ('', 'a', 'approve', 'yes', 'y'):
                log_deliberation(
                    agent_role="system",
                    event="checkpoint_approved",
                    content=f"Stage {stage} approved by user",
                    stage=stage
                )
                print("✓ Approved. Continuing to next stage...")
                return True
            elif response in ('r', 'reject', 'no', 'n', 'abort'):
                log_deliberation(
                    agent_role="system",
                    event="checkpoint_rejected",
                    content=f"Stage {stage} rejected by user",
                    stage=stage
                )
                print("✗ Rejected. Aborting session.")
                sys.exit(0)
            else:
                print("Please enter 'A' to approve or 'R' to reject.")
        except (EOFError, KeyboardInterrupt):
            log_deliberation(
                agent_role="system",
                event="session_interrupted",
                content=f"Session interrupted during {stage} checkpoint",
                stage=stage
            )
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
        You prefer simple solutions that are easy to understand and maintain.
        Use discover_skill to find relevant skills/best practices for the task domain.
        Use spawn_research to research multiple topics in parallel, then collect_research to get results.""",
        llm="anthropic/claude-opus-4-5-20251101",
        tools=[discover_skill_tool, spawn_research_tool, collect_research_tool],
        verbose=True,
        allow_delegation=False,
        step_callback=agent_step_callback
    )

    # DeepSeek - focuses on performance and scalability
    architect_deepseek = Agent(
        role="Software Architect (DeepSeek)",
        goal="Review architecture for performance, scalability, and practical implementation concerns",
        backstory="""You are an architect focused on performance and scalability.
        You review designs for efficiency, identify potential bottlenecks, and suggest optimizations.
        You ensure the architecture will perform well under load.
        Use spawn_research to research performance patterns in parallel, then collect_research to get results.""",
        llm=LLM(model="ollama/deepseek-coder-v2:16b", base_url="http://localhost:11434"),
        tools=[spawn_research_tool, collect_research_tool],
        verbose=True,
        allow_delegation=False,
        step_callback=agent_step_callback
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
        allow_delegation=False,
        step_callback=agent_step_callback
    )

    # === REVIEWERS (Dual Perspective) ===

    # Claude Opus 4.5 - focuses on security and edge cases
    reviewer_claude = Agent(
        role="Code Reviewer (Claude)",
        goal="Review code for security vulnerabilities, edge cases, and maintainability",
        backstory="""You are a security-focused code reviewer who catches subtle bugs.
        You focus on security vulnerabilities, error handling, and edge cases.
        You provide specific, actionable feedback with code examples.
        Use list_skills to see available expertise, then query_skill to get guidance.
        Use discover_skill to find new skills and spawn_research for parallel research.""",
        llm="anthropic/claude-opus-4-5-20251101",
        tools=[list_skills_tool, query_skill, discover_skill_tool, spawn_research_tool, collect_research_tool],
        verbose=True,
        allow_delegation=False,
        step_callback=agent_step_callback
    )

    # DeepSeek - focuses on performance and efficiency
    reviewer_deepseek = Agent(
        role="Code Reviewer (DeepSeek)",
        goal="Review code for performance issues, inefficiencies, and optimization opportunities",
        backstory="""You are a performance-focused code reviewer.
        You identify slow algorithms, memory issues, and optimization opportunities.
        You suggest concrete performance improvements with benchmarks when possible.
        Use list_skills to see available expertise, then query_skill to get guidance.
        Use discover_skill to find new skills and spawn_research for parallel research.""",
        llm=LLM(model="ollama/deepseek-coder-v2:16b", base_url="http://localhost:11434"),
        tools=[list_skills_tool, query_skill, discover_skill_tool, spawn_research_tool, collect_research_tool],
        verbose=True,
        allow_delegation=False,
        step_callback=agent_step_callback
    )

    return architect_claude, architect_deepseek, builder, reviewer_claude, reviewer_deepseek


# =============================================================================
# DESIGN MODE: UI/UX focused agents for mockup generation
# =============================================================================

def create_design_agents():
    """Create agents specialized for design exploration."""

    ux_designer = Agent(
        role="UX Designer (Claude)",
        goal="Design intuitive user flows, information architecture, and interaction patterns",
        backstory="""You are a senior UX designer focused on how interfaces FEEL to use.
        You think about user journeys, cognitive load, thumb reachability on mobile,
        and emotional resonance. You avoid complexity unless it serves the user.

        Start by running list_skills to see available expertise, then query relevant
        skills like web-design-guidelines for accessibility guidance.""",
        llm="anthropic/claude-opus-4-5-20251101",
        tools=[list_skills_tool, query_skill, discover_skill_tool, spawn_research_tool, collect_research_tool],
        verbose=True,
        allow_delegation=False,
        step_callback=agent_step_callback
    )

    visual_designer = Agent(
        role="Visual Designer (GPT)",
        goal="Create distinctive, memorable visual designs that avoid generic AI aesthetics",
        backstory="""You are a bold visual designer who hates cookie-cutter interfaces.
        You think about typography (never use Inter/Arial), color palettes with personality,
        spatial composition, and motion/micro-interactions.

        Run list_skills first, then query_skill("frontend-design", "aesthetics") for guidance.
        Use write_file to save mockups as viewable HTML files.""",
        llm="openai/gpt-5.2",
        tools=[write_file, list_skills_tool, query_skill],
        verbose=True,
        allow_delegation=False,
        step_callback=agent_step_callback
    )

    tech_designer = Agent(
        role="Technical Designer (DeepSeek)",
        goal="Review designs for technical feasibility, mobile performance, and implementation complexity",
        backstory="""You review design proposals with an engineer's eye.
        You flag animations that cause jank, layouts that break on small screens,
        and complexity that slows development. Suggest simpler alternatives when needed.

        Use list_skills and query_skill for react-best-practices guidance.""",
        llm=LLM(model="ollama/deepseek-coder-v2:16b", base_url="http://localhost:11434"),
        tools=[list_skills_tool, query_skill],
        verbose=True,
        allow_delegation=False,
        step_callback=agent_step_callback
    )

    synthesizer = Agent(
        role="Design Synthesizer (Claude)",
        goal="Combine design perspectives into clear options for human decision-making",
        backstory="""You synthesize input from UX, visual, and technical designers
        into 2-3 distinct design directions. Present tradeoffs clearly without
        making the decision for the human. Save output as DESIGN-BRIEF.md.""",
        llm="anthropic/claude-opus-4-5-20251101",
        tools=[write_file],
        verbose=True,
        allow_delegation=False,
        step_callback=agent_step_callback
    )

    return ux_designer, visual_designer, tech_designer, synthesizer


def create_design_tasks(ux_designer, visual_designer, tech_designer, synthesizer, task_description: str):
    """Create the 4-stage design pipeline tasks."""

    ux_task = Task(
        description=f"""Analyze the design requirements and create UX specifications:

{task_description}

First, run list_skills to see available expertise. Query relevant skills for guidance.

Provide:
1. User goals for this screen/flow
2. Information hierarchy (what's most important?)
3. Interaction patterns (tap, swipe, scroll behaviors)
4. Mobile-first layout recommendations (thumb zones, reachability)
5. Accessibility considerations

Focus on HOW IT FEELS TO USE, not how it looks.""",
        expected_output="""UX specification with:
- User goals and success metrics
- Information architecture
- Interaction flow description
- Mobile layout guidance
- Accessibility notes""",
        agent=ux_designer
    )

    visual_task = Task(
        description="""Based on the UX specification, create 2-3 DISTINCT visual design directions.

First, run list_skills and query_skill("frontend-design", "aesthetics") for guidelines.

For EACH direction:
1. Choose a unique aesthetic (e.g., warm minimalism, bold typography, soft organic)
2. Define typography (distinctive fonts, not Inter/Arial/system fonts)
3. Define color palette (with CSS variables)
4. Create a viewable HTML mockup with embedded CSS

REQUIREMENTS:
- Each direction must be visually DIFFERENT (not variations of same theme)
- Use write_file to save: design/option-a.html, design/option-b.html, etc.
- Mobile-first (375px width, use max-width for desktop)
- Include hover states and any micro-interactions

If you discover a useful skill via discover_skill, note it for installation at checkpoint.""",
        expected_output="""2-3 HTML mockup files saved to workspace/design/:
- design/option-a.html
- design/option-b.html
- design/option-c.html (optional)

Each viewable in browser with complete styling.""",
        agent=visual_designer,
        context=[ux_task]
    )

    tech_task = Task(
        description="""Review the visual designs for technical feasibility.

Run list_skills and query react-best-practices for guidance.

For each design direction, assess:
1. Implementation complexity (1-10 scale)
2. Mobile performance concerns (animations, repaints, large images)
3. Accessibility issues (contrast, focus states, screen reader)
4. React component structure recommendations
5. CSS concerns (browser support, layout stability)

Flag issues but don't kill creativity - suggest alternatives that preserve the design intent.""",
        expected_output="""Technical assessment for each design:
- Complexity rating with justification
- Performance flags and fixes
- Accessibility audit results
- Recommended component breakdown
- CSS/implementation notes""",
        agent=tech_designer,
        context=[visual_task]
    )

    synthesis_task = Task(
        description="""Synthesize all input into a design brief for human decision-making.

Create a summary document that:
1. Names each direction clearly (e.g., "Warm Minimalism", "Bold Editorial")
2. Shows a side-by-side comparison table
3. Lists pros/cons from UX, visual, and technical perspectives
4. Notes which direction best fits the stated product vision
5. Does NOT make the decision - presents options for human choice

Use write_file to save as design/DESIGN-BRIEF.md""",
        expected_output="""design/DESIGN-BRIEF.md containing:
- Direction summaries with names
- Comparison matrix
- Tradeoffs from each perspective
- Recommendation (without deciding)
- Links to mockup files""",
        agent=synthesizer,
        context=[ux_task, visual_task, tech_task]
    )

    return [ux_task, visual_task, tech_task, synthesis_task]


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
    log_deliberation(
        agent_role="system",
        event="stage_start",
        content="Starting planning phase with Claude and DeepSeek architects",
        stage="PLANNING"
    )
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
    log_deliberation(
        agent_role="system",
        event="stage_start",
        content="Starting building phase with GPT-5.2 builder",
        stage="BUILDING"
    )
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
    log_deliberation(
        agent_role="system",
        event="stage_start",
        content="Starting review phase with Claude (security) and DeepSeek (performance) reviewers",
        stage="REVIEW"
    )
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


# =============================================================================
# DESIGN MODE: Run the design pipeline
# =============================================================================

def run_design_mode(task_description: str, use_checkpoints: bool = False):
    """Run the council in design mode."""

    design_dir = WORKSPACE_DIR / "design"
    design_dir.mkdir(parents=True, exist_ok=True)

    log_deliberation(
        agent_role="system",
        event="design_mode_start",
        content=f"Starting design mode: {task_description[:200]}"
    )

    print(f"\n{'='*60}")
    print("DESIGN MODE")
    print(f"{'='*60}")
    print(f"\nAgents (4 total, 3 models):")
    print("  UX ANALYSIS:")
    print("    - UX Designer (Claude Opus 4.5): User flows, interaction patterns")
    print("  VISUAL DESIGN:")
    print("    - Visual Designer (GPT-5.2): Aesthetics, mockup generation")
    print("  TECHNICAL REVIEW:")
    print("    - Technical Designer (DeepSeek): Feasibility, performance")
    print("  SYNTHESIS:")
    print("    - Synthesizer (Claude Opus 4.5): Combine into options")
    print(f"\nOutput: {design_dir}")
    print(f"{'='*60}\n")

    ux_designer, visual_designer, tech_designer, synthesizer = create_design_agents()
    tasks = create_design_tasks(ux_designer, visual_designer, tech_designer, synthesizer, task_description)

    if use_checkpoints:
        # Stage 1: UX
        log_deliberation(agent_role="system", event="stage_start", content="UX Analysis", stage="UX")
        print(f"\n{'='*60}")
        print("STAGE 1: UX ANALYSIS")
        print(f"{'='*60}\n")

        ux_crew = Crew(agents=[ux_designer], tasks=[tasks[0]], process=Process.sequential, verbose=True)
        ux_result = ux_crew.kickoff()
        checkpoint("UX ANALYSIS", f"UX specification complete.\n\n{str(ux_result)[:1500]}")

        # Stage 2: Visual (inject UX context)
        log_deliberation(agent_role="system", event="stage_start", content="Visual Design", stage="VISUAL")
        print(f"\n{'='*60}")
        print("STAGE 2: VISUAL DESIGN")
        print(f"{'='*60}\n")

        # Re-create visual task with context from UX result
        visual_task_with_context = Task(
            description=f"""Based on the UX specification below, create 2-3 DISTINCT visual design directions.

UX SPECIFICATION:
{str(ux_result)[:2500]}

First, run list_skills and query_skill("frontend-design", "aesthetics") for guidelines.

For EACH direction:
1. Choose a unique aesthetic (e.g., warm minimalism, bold typography, soft organic)
2. Define typography (distinctive fonts, not Inter/Arial/system fonts)
3. Define color palette (with CSS variables)
4. Create a viewable HTML mockup with embedded CSS

REQUIREMENTS:
- Each direction must be visually DIFFERENT (not variations of same theme)
- Use write_file to save: design/option-a.html, design/option-b.html, etc.
- Mobile-first (375px width, use max-width for desktop)
- Include hover states and any micro-interactions""",
            expected_output="""2-3 HTML mockup files saved to workspace/design/:
- design/option-a.html
- design/option-b.html
- design/option-c.html (optional)

Each viewable in browser with complete styling.""",
            agent=visual_designer
        )

        visual_crew = Crew(agents=[visual_designer], tasks=[visual_task_with_context], process=Process.sequential, verbose=True)
        visual_result = visual_crew.kickoff()

        design_files = list(design_dir.glob("*.html"))
        files_list = "\n".join(f"  - {f.name}" for f in design_files)
        checkpoint("VISUAL DESIGN", f"Mockups created:\n{files_list}\n\n{str(visual_result)[:1000]}")

        # Stage 3: Technical
        log_deliberation(agent_role="system", event="stage_start", content="Technical Review", stage="TECHNICAL")
        print(f"\n{'='*60}")
        print("STAGE 3: TECHNICAL REVIEW")
        print(f"{'='*60}\n")

        tech_task_with_context = Task(
            description=f"""Review the visual designs for technical feasibility.

VISUAL DESIGN OUTPUT:
{str(visual_result)[:2000]}

Run list_skills and query react-best-practices for guidance.

For each design direction, assess:
1. Implementation complexity (1-10 scale)
2. Mobile performance concerns (animations, repaints, large images)
3. Accessibility issues (contrast, focus states, screen reader)
4. React component structure recommendations
5. CSS concerns (browser support, layout stability)

Flag issues but don't kill creativity - suggest alternatives that preserve the design intent.""",
            expected_output="""Technical assessment for each design:
- Complexity rating with justification
- Performance flags and fixes
- Accessibility audit results
- Recommended component breakdown
- CSS/implementation notes""",
            agent=tech_designer
        )

        tech_crew = Crew(agents=[tech_designer], tasks=[tech_task_with_context], process=Process.sequential, verbose=True)
        tech_result = tech_crew.kickoff()
        checkpoint("TECHNICAL REVIEW", f"Technical assessment complete.\n\n{str(tech_result)[:1500]}")

        # Stage 4: Synthesis
        log_deliberation(agent_role="system", event="stage_start", content="Synthesis", stage="SYNTHESIS")
        print(f"\n{'='*60}")
        print("STAGE 4: SYNTHESIS")
        print(f"{'='*60}\n")

        synth_task_with_context = Task(
            description=f"""Synthesize all input into a design brief for human decision-making.

UX SPECIFICATION:
{str(ux_result)[:1500]}

VISUAL DESIGNS:
{str(visual_result)[:1500]}

TECHNICAL ASSESSMENT:
{str(tech_result)[:1500]}

Create a summary document that:
1. Names each direction clearly (e.g., "Warm Minimalism", "Bold Editorial")
2. Shows a side-by-side comparison table
3. Lists pros/cons from UX, visual, and technical perspectives
4. Notes which direction best fits the stated product vision
5. Does NOT make the decision - presents options for human choice

Use write_file to save as design/DESIGN-BRIEF.md""",
            expected_output="""design/DESIGN-BRIEF.md containing:
- Direction summaries with names
- Comparison matrix
- Tradeoffs from each perspective
- Recommendation (without deciding)
- Links to mockup files""",
            agent=synthesizer
        )

        synth_crew = Crew(agents=[synthesizer], tasks=[synth_task_with_context], process=Process.sequential, verbose=True)
        result = synth_crew.kickoff()
    else:
        crew = Crew(
            agents=[ux_designer, visual_designer, tech_designer, synthesizer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        result = crew.kickoff()

    # Summary
    design_files = list(design_dir.glob("*"))
    print(f"\n{'='*60}")
    print("DESIGN OUTPUT")
    print(f"{'='*60}")
    for f in sorted(design_files):
        print(f"  - {f.name}")
    print(f"\nOpen HTML files in browser to view mockups.")
    print(f"Read DESIGN-BRIEF.md for synthesis and recommendations.")

    log_deliberation(agent_role="system", event="design_mode_complete", content=str(result)[:500])

    return result


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="CrewAI Council - Multi-model AI deliberation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crew.py "Build a REST API for user authentication"
  python crew.py --checkpoint "Create a CLI tool"
  python crew.py --design "Design the home screen for a food diary app"
  python crew.py --design --checkpoint "Design a mobile-first checkout flow"
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
    parser.add_argument(
        "--design",
        action="store_true",
        help="Run in design mode - generates UI mockups and design options instead of code"
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

    # Initialize deliberation log for this session
    clear_deliberation_log()
    # Determine mode
    if args.design:
        mode = "design_checkpoint" if args.checkpoint else "design"
    elif args.checkpoint:
        mode = "checkpoint"
    else:
        mode = "continuous"

    log_deliberation(
        agent_role="system",
        event="task_received",
        content=task_description,
        mode=mode
    )

    # Auto-start Ollama if needed (unless --no-ollama flag)
    if not args.no_ollama:
        ensure_ollama_running()

    # Check provider status
    status = check_providers()

    # Design mode has its own banner, so only show code council banner for regular mode
    if not args.design:
        print(f"\n{'='*60}")
        print("CREWAI COUNCIL (Dual-Perspective)")
        print(f"{'='*60}")

        if args.checkpoint:
            print("Mode: CHECKPOINT (will pause between stages)")
        else:
            print("Mode: CONTINUOUS (no pauses)")

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
        print(f"\n💬 Live streaming: tail -f {DELIBERATION_LOG} | jq -c")
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
    else:
        # Design mode - minimal provider check output
        print_provider_status(status)

        if not all(status.values()):
            if args.yes:
                print("\n--yes flag: continuing despite missing providers")
            else:
                response = input("\nContinue anyway? [y/N]: ")
                if response.lower() != 'y':
                    print("Aborted.")
                    sys.exit(1)

    # Route to appropriate execution mode
    if args.design:
        result = run_design_mode(task_description, use_checkpoints=args.checkpoint)
    elif args.checkpoint:
        result = run_with_checkpoints(task_description)
    else:
        crew = create_crew(task_description)
        result = crew.kickoff()

    # Log session complete
    log_deliberation(
        agent_role="system",
        event="session_complete",
        content=str(result)[:500]
    )

    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    print(result)
    print(f"\n📜 Deliberation log: {DELIBERATION_LOG}")

    return result


if __name__ == "__main__":
    main()
