class WorkerAgentError(Exception):
    """Base exception for all worker-agent failures."""


class ResearchAgentError(WorkerAgentError):
    """Raised when the research agent cannot complete a search."""


class AnalyticalAgentError(WorkerAgentError):
    """Raised when the analytical agent cannot process the data."""


class SynthesisAgentError(WorkerAgentError):
    """Raised when the synthesis agent cannot build a response."""


class InvalidTaskError(WorkerAgentError):
    """Raised when a worker receives an unsupported task."""