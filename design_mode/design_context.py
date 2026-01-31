from __future__ import annotations

"""Context persistence for the design pipeline.

Core principle: explicit context injection.

A DesignContext is treated as an immutable snapshot. Updates return a new
instance via `with_stage_result`.

Persistence uses an atomic write strategy to reduce the chance of corrupted
state on interruption.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Optional

from .errors import ContextLoadError, ContextSaveError


@dataclass(frozen=True, slots=True)
class DesignContext:
    """Immutable snapshot of design pipeline state."""

    task_description: str
    stage_results: dict[str, str] = field(default_factory=dict)
    current_stage: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def with_stage_result(self, stage_name: str, result: str) -> "DesignContext":
        """Return new context with added stage result (immutable update)."""
        new_results = {**self.stage_results, stage_name: result}
        return DesignContext(
            task_description=self.task_description,
            stage_results=new_results,
            current_stage=self.current_stage + 1,
            started_at=self.started_at,
        )


class DesignContextStore:
    """Handles persistence of design context to filesystem."""

    CONTEXT_FILE = "design_context.json"

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.design_dir = workspace_dir / "design"
        self.context_path = self.design_dir / self.CONTEXT_FILE

    def ensure_directory(self) -> None:
        """Create design output directory if it doesn't exist."""
        self.design_dir.mkdir(parents=True, exist_ok=True)

    def save(self, context: DesignContext) -> None:
        """Persist context to disk using atomic write."""
        try:
            self.ensure_directory()
            tmp_path = self.context_path.with_suffix(self.context_path.suffix + ".tmp")
            payload = json.dumps(asdict(context), indent=2, ensure_ascii=False)
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(self.context_path)
        except Exception as exc:  # pragma: no cover (hard to simulate all FS failures)
            raise ContextSaveError(f"Failed to save context to {self.context_path}: {exc}") from exc

    def load(self) -> Optional[DesignContext]:
        """Load context from disk; return None if not found."""
        if not self.context_path.exists():
            return None
        try:
            raw = self.context_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            # Backward/forward tolerance: ignore unknown keys.
            allowed = {"task_description", "stage_results", "current_stage", "started_at"}
            filtered = {k: v for k, v in data.items() if k in allowed}
            return DesignContext(**filtered)
        except Exception as exc:
            raise ContextLoadError(f"Failed to load context from {self.context_path}: {exc}") from exc

    def clear(self) -> None:
        """Remove persisted context (for fresh starts)."""
        try:
            if self.context_path.exists():
                self.context_path.unlink()
        except Exception as exc:  # pragma: no cover
            raise ContextSaveError(f"Failed to clear context at {self.context_path}: {exc}") from exc
