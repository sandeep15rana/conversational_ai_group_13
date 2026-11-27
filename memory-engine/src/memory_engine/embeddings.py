"""
Text embeddings (Developer 3 - Rajarshi).

Model: sentence-transformers all-MiniLM-L6-v2 (384 dimensions), run through
ONNX Runtime. ChromaDB downloads this exact model from its own server
(chroma-onnx-models.s3.amazonaws.com), not from HuggingFace, and it needs no
PyTorch. Vectors are L2-normalized, so cosine similarity = dot product.

    cosine_similarity(A, B) = (A . B) / (|A| * |B|)
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from typing import Optional, Protocol, Sequence, runtime_checkable

from .exceptions import EmbeddingError

Vector = list[float]


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into vectors and can count tokens."""

    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...

    def count_tokens(self, text: str) -> int: ...


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """(A . B) / (|A| * |B|). Returns 0.0 if either vector is all zeros."""
    if len(a) != len(b):
        raise ValueError(f"Vector sizes differ: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _l2_normalize(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [float(x) / norm for x in vector]


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 via ChromaDB's ONNX build. Loads lazily, once, thread-safe."""

    name = "all-MiniLM-L6-v2"
    dimension = 384
    max_model_tokens = 256

    retry_after_seconds = 60.0  # after a failed load, fail fast instead of re-downloading on every request

    def __init__(self, model_dir: Optional[str] = None, clock=time.monotonic) -> None:
        self._model_dir = model_dir
        self._model = None
        self._counting_tokenizer = None
        self._lock = threading.Lock()
        self._clock = clock
        self._last_failure: Optional[tuple[float, EmbeddingError]] = None

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                if self._last_failure and self._clock() - self._last_failure[0] < self.retry_after_seconds:
                    raise self._last_failure[1]  # cached failure: no network wait on each request
                try:
                    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

                    model = ONNXMiniLM_L6_V2()
                    if self._model_dir:
                        model.DOWNLOAD_PATH = self._model_dir
                    model(["warm-up"])  # downloads on first use, then loads the session
                    self._model = model
                    self._last_failure = None
                except Exception as exc:  # network blocked, corrupt cache, missing runtime...
                    error = EmbeddingError(
                        f"Could not load the {self.name} embedding model: {exc}. "
                        "See INTEGRATION.md > Troubleshooting for the offline option."
                    )
                    self._last_failure = (self._clock(), error)
                    raise error from exc
        return self._model

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        model = self._load()
        try:
            vectors = model(list(texts))
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return [_l2_normalize(v) for v in vectors]

    def count_tokens(self, text: str) -> int:
        """Exact word-piece count with the model's own tokenizer (no truncation)."""
        if not text:
            return 0
        if self._counting_tokenizer is None:
            model = self._load()
            with self._lock:
                if self._counting_tokenizer is None:
                    path = os.path.join(model.DOWNLOAD_PATH, model.EXTRACTED_FOLDER_NAME, "tokenizer.json")
                    tokenizer = model.Tokenizer.from_file(path)
                    tokenizer.no_truncation()
                    tokenizer.no_padding()
                    self._counting_tokenizer = tokenizer
        return len(self._counting_tokenizer.encode(text, add_special_tokens=False).ids)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """
    Deterministic bag-of-words embedder (feature hashing).

    NOT semantic: it only matches shared words. Used by the unit tests so they
    run fast and offline, and usable as an emergency fallback if the real model
    cannot be loaded.
    """

    name = "hashing-bow"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _vector(self, text: str) -> Vector:
        vec = [0.0] * self.dimension
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        return _l2_normalize(vec)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(t) for t in texts]

    def count_tokens(self, text: str) -> int:
        # Rough word-piece estimate: about 1.3 pieces per word.
        return math.ceil(len(text.split()) * 1.3)
