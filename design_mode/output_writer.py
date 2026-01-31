from __future__ import annotations

"""Output writing utilities.

Keeps filesystem concerns separate from pipeline orchestration.
Uses atomic writes for final artifacts.
"""

from dataclasses import dataclass
from pathlib import Path

from .errors import OutputWriteError


@dataclass(frozen=True, slots=True)
class OutputPaths:
    design_dir: Path
    context_json: Path
    design_brief_md: Path
    mockup_html: Path


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def ensure_output_paths(workspace_dir: Path) -> OutputPaths:
    design_dir = workspace_dir / "design"
    design_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        design_dir=design_dir,
        context_json=design_dir / "design_context.json",
        design_brief_md=design_dir / "DESIGN-BRIEF.md",
        mockup_html=design_dir / "mockup.html",
    )


def write_synthesis_outputs(workspace_dir: Path, synthesis_text: str) -> OutputPaths:
    """Write final artifacts.

    Note: We keep parsing intentionally simple as per spec.
    """
    try:
        paths = ensure_output_paths(workspace_dir)

        # Minimal extraction strategy:
        # - Always write DESIGN-BRIEF.md as a wrapper around synthesis.
        # - Also write mockup.html if we can find an <html>...</html> block.
        _atomic_write_text(paths.design_brief_md, f"# Design Brief\n\n{synthesis_text}\n")

        lower = synthesis_text.lower()
        start = lower.find("<html")
        end = lower.rfind("</html>")
        if start != -1 and end != -1 and end > start:
            html = synthesis_text[start : end + len("</html>")]
        else:
            html = (
                "<!doctype html>\n"
                "<html><head><meta charset='utf-8'><title>Mockup</title></head>"
                "<body><pre style='white-space:pre-wrap'>\n"
                + synthesis_text
                + "\n</pre></body></html>\n"
            )
        _atomic_write_text(paths.mockup_html, html)

        return paths
    except Exception as exc:  # pragma: no cover
        raise OutputWriteError(f"Failed to write design outputs: {exc}") from exc
