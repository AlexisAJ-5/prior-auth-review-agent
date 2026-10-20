"""Text embedders. FastEmbed (ONNX, local) in production; a hashing embedder for tests and offline use."""

import hashlib
import math
import re
from itertools import pairwise
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    """Feature-hashed unigrams and bigrams, L2-normalized. Deterministic, no model download.

    Retrieval quality is lexical only; it exists so that tests and CI never need the network.
    """

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        vector = [0.0] * self.dimension
        for feature in tokens + [f"{a} {b}" for a, b in pairwise(tokens)]:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbedder:
    """BAAI/bge-small-en-v1.5 by default: 384 dimensions, ~70 MB, CPU-only ONNX runtime."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed import TextEmbedding  # imported lazily: heavy, and unused in tests

        self._model = TextEmbedding(model_name, cache_dir=cache_dir)
        self.dimension = len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(iter(self._model.query_embed(text)))]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
