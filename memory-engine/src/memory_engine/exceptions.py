"""
Memory Engine exceptions (Developer 3 - Rajarshi).

Every error raised by this package inherits from MemoryEngineError, so callers
can catch one type. The LangGraph nodes catch these and degrade gracefully:
a memory failure never stops the main pipeline.
"""


class MemoryEngineError(Exception):
    """Base class for all memory engine errors."""


class MemoryValidationError(MemoryEngineError):
    """Input failed validation (bad user id, empty text, wrong type...)."""


class EmbeddingError(MemoryEngineError):
    """The embedding model could not be loaded or failed to encode text."""


class MemoryStoreError(MemoryEngineError):
    """The vector store (ChromaDB) failed to read or write."""


class ExtractionError(MemoryEngineError):
    """The memory extractor (LLM or rules) failed."""
