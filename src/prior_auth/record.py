"""The redacted clinical record, split into addressable sentences the agent can search and cite."""

import re
from dataclasses import dataclass

from .embeddings import Embedder, cosine, tokenize
from .schemas import Sentence, SentenceSource

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
_ABBREVIATION = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Pt|vs|approx|e\.g|i\.e)\.$", re.I)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "with",
})  # fmt: skip

# Hybrid score: dense similarity plus a lexical overlap bonus. Clinical wording is precise
# ("saddle anesthesia", "Kellgren-Lawrence"), and small embedding models blur such terms.
LEXICAL_WEIGHT = 0.35


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.splitlines():
        pieces: list[str] = []
        for piece in (p.strip() for p in _SPLIT.split(line)):
            if pieces and _ABBREVIATION.search(pieces[-1]):  # "Dr." does not end a sentence
                pieces[-1] = f"{pieces[-1]} {piece}"
            elif piece:
                pieces.append(piece)
        sentences += [p for p in pieces if len(p) > 2]
    return sentences


@dataclass
class SearchHit:
    sentence: Sentence
    score: float


class ClinicalRecord:
    def __init__(self, notes: str, document: str | None, embedder: Embedder):
        self.sentences: list[Sentence] = []
        for prefix, source, text in (("N", "notes", notes), ("D", "document", document or "")):
            src: SentenceSource = source  # type: ignore[assignment]
            for i, sentence in enumerate(split_sentences(text), start=1):
                self.sentences.append(Sentence(id=f"{prefix}{i}", source=src, text=sentence))
        self._by_id = {s.id: s for s in self.sentences}
        self._embedder = embedder
        self._vectors = embedder.embed_documents([s.text for s in self.sentences]) if self.sentences else []

    def get(self, sentence_id: str) -> Sentence | None:
        return self._by_id.get(sentence_id)

    def count(self, source: SentenceSource) -> int:
        return sum(1 for s in self.sentences if s.source == source)

    def search(self, query: str, k: int = 4) -> list[SearchHit]:
        if not self.sentences:
            return []
        query_vector = self._embedder.embed_query(query)
        query_terms = {t for t in tokenize(query) if t not in _STOPWORDS}
        hits = []
        for sentence, vector in zip(self.sentences, self._vectors, strict=True):
            terms = set(tokenize(sentence.text))
            lexical = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
            hits.append(
                SearchHit(sentence, round(cosine(query_vector, vector) + LEXICAL_WEIGHT * lexical, 4))
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
