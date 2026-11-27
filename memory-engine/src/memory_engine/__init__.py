"""
Dual-Layer Memory & Context Engine (Developer 3 - Rajarshi)
AIMLCZG521: Conversational AI - Group 13

Short-term: per-session conversation buffer + scratchpad.
Long-term : ChromaDB + all-MiniLM-L6-v2 embeddings, per-user partitioned.
"""

from .config import MemoryConfig
from .engine import MemoryEngine, get_default_engine, set_default_engine
from .exceptions import (
    EmbeddingError,
    ExtractionError,
    MemoryEngineError,
    MemoryStoreError,
    MemoryValidationError,
)
from .nodes import (
    build_supervisor_memory_block,
    load_scratchpad,
    memory_context_for_synthesis,
    memory_recall_node,
    memory_save_node,
    no_memory_node,
    save_scratchpad,
)
from .schemas import MemoryType, RecallResult, SaveResult

__all__ = [
    "MemoryConfig",
    "MemoryEngine",
    "MemoryType",
    "RecallResult",
    "SaveResult",
    "get_default_engine",
    "set_default_engine",
    "memory_recall_node",
    "memory_save_node",
    "memory_context_for_synthesis",
    "build_supervisor_memory_block",
    "no_memory_node",
    "save_scratchpad",
    "load_scratchpad",
    "MemoryEngineError",
    "MemoryValidationError",
    "MemoryStoreError",
    "EmbeddingError",
    "ExtractionError",
]

__version__ = "1.1.0"
