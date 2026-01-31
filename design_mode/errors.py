"""Domain-specific exceptions for design mode."""


class DesignModeError(Exception):
    """Base exception for design-mode errors."""


class ContextLoadError(DesignModeError):
    """Raised when persisted context cannot be loaded."""


class ContextSaveError(DesignModeError):
    """Raised when persisted context cannot be saved."""


class StageDependencyError(DesignModeError):
    """Raised when a stage is missing required dependency outputs."""


class StageExecutionError(DesignModeError):
    """Raised when a stage fails to execute."""


class OutputWriteError(DesignModeError):
    """Raised when writing output artifacts fails."""
