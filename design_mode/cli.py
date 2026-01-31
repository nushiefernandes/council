from __future__ import annotations

"""CLI integration for design mode.

This module is designed to be imported by the project's main CLI entrypoint.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .design_runner import DesignPipelineRunner, RunResult


@dataclass(frozen=True, slots=True)
class DesignArgs:
    task: str
    workspace: Optional[str] = None
    checkpoint: bool = True
    resume: bool = False
    verbose: bool = True


def run_design_mode(args: DesignArgs) -> RunResult:
    """Entry point for a `--design` flag."""
    workspace = Path(args.workspace or ".").resolve()

    runner = DesignPipelineRunner(
        workspace_dir=workspace,
        checkpoint_mode=args.checkpoint,
        verbose=args.verbose,
    )

    return runner.run(task_description=args.task, resume=args.resume)
