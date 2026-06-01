"""Small TF-IDF vectorizer and vector similarity functions."""

from __future__ import annotations

import math
from collections import Counter


Vector = dict[str, float]


def tfidf_vectors(documents: list[list[str]]) -> list[Vector]:
    """Build TF-IDF vectors for a small in-memory document collection."""

    if not documents:
        return []

    doc_count = len(documents)
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))

    vectors: list[Vector] = []
    for document in documents:
        term_counts = Counter(document)
        total_terms = sum(term_counts.values()) or 1
        vector: Vector = {}
        for term, count in term_counts.items():
            tf = count / total_terms
            idf = math.log((doc_count + 1) / (document_frequency[term] + 1)) + 1
            vector[term] = tf * idf
        vectors.append(vector)
    return vectors


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Return cosine similarity for sparse vectors."""

    if not left or not right:
        return 0.0
    shared_terms = set(left) & set(right)
    numerator = sum(left[term] * right[term] for term in shared_terms)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def top_terms(vector: Vector, limit: int = 8) -> list[tuple[str, float]]:
    """Return top weighted terms for explaining a similarity result."""

    return sorted(vector.items(), key=lambda item: item[1], reverse=True)[:limit]
