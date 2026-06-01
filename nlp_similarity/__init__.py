"""Sentence semantic similarity toolkit."""

from .similarity import SimilarityResult, SentenceSimilarityEngine
from .ml_model import LogisticSimilarityModel

__all__ = ["SentenceSimilarityEngine", "SimilarityResult", "LogisticSimilarityModel"]
