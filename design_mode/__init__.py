"""Design mode package.

Implements a 4-stage design pipeline with explicit context persistence and
checkpoint/resume support.
"""

from .design_runner import DesignPipelineRunner

__all__ = ["DesignPipelineRunner"]
