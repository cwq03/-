"""Lightweight supervised feature fusion model."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .similarity import SentenceSimilarityEngine, SimilarityResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "model_weights.json"

FEATURE_NAMES = [
    "word_tfidf",
    "char_ngram_tfidf",
    "normalized_jaccard",
    "edit_similarity",
    "fusion_before_penalty",
    "rule_score",
    "negation_mismatch",
    "opposite_mismatch",
    "negation_penalty",
    "opposite_penalty",
]


@dataclass(frozen=True)
class Prediction:
    probability: float
    label: int
    features: dict[str, float]
    rule_score: float


class LogisticSimilarityModel:
    """A small logistic regression model trained on similarity features."""

    def __init__(
        self,
        weights: list[float] | None = None,
        bias: float = 0.0,
        threshold: float = 0.50,
        feature_names: list[str] | None = None,
    ) -> None:
        self.feature_names = feature_names or FEATURE_NAMES
        self.weights = weights or [0.0 for _ in self.feature_names]
        self.bias = bias
        self.threshold = threshold

    def predict(self, result: SimilarityResult) -> Prediction:
        features = extract_features(result)
        values = [features[name] for name in self.feature_names]
        probability = sigmoid(self.bias + sum(w * x for w, x in zip(self.weights, values)))
        return Prediction(
            probability=round(probability, 4),
            label=int(probability >= self.threshold),
            features=features,
            rule_score=result.score,
        )

    def save(self, path: Path | str = DEFAULT_MODEL_PATH) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "logistic_regression",
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL_PATH) -> "LogisticSimilarityModel":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return cls(
            weights=[float(value) for value in payload["weights"]],
            bias=float(payload["bias"]),
            threshold=float(payload.get("threshold", 0.50)),
            feature_names=list(payload.get("feature_names", FEATURE_NAMES)),
        )


def extract_features(result: SimilarityResult) -> dict[str, float]:
    return {
        "word_tfidf": result.word_tfidf,
        "char_ngram_tfidf": result.char_ngram_tfidf,
        "normalized_jaccard": result.normalized_jaccard,
        "edit_similarity": result.edit_similarity,
        "fusion_before_penalty": result.fusion_before_penalty,
        "rule_score": result.score,
        "negation_mismatch": float(result.negation_mismatch),
        "opposite_mismatch": float(result.opposite_mismatch),
        "negation_penalty": result.negation_penalty,
        "opposite_penalty": result.opposite_penalty,
    }


def train_logistic_model(
    rows: list[tuple[str, str, int]],
    engine: SentenceSimilarityEngine | None = None,
    epochs: int = 120,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    threshold: float = 0.50,
) -> LogisticSimilarityModel:
    engine = engine or SentenceSimilarityEngine()
    training_data: list[tuple[list[float], int]] = []
    for sentence_a, sentence_b, label in rows:
        result = engine.compare(sentence_a, sentence_b)
        features = extract_features(result)
        training_data.append(([features[name] for name in FEATURE_NAMES], label))

    weights = [0.0 for _ in FEATURE_NAMES]
    bias = 0.0
    positive_count = sum(label for _, label in training_data)
    negative_count = len(training_data) - positive_count
    positive_weight = len(training_data) / (2 * positive_count) if positive_count else 1.0
    negative_weight = len(training_data) / (2 * negative_count) if negative_count else 1.0

    for _ in range(epochs):
        for values, label in training_data:
            probability = sigmoid(bias + sum(w * x for w, x in zip(weights, values)))
            sample_weight = positive_weight if label == 1 else negative_weight
            error = (probability - label) * sample_weight
            for index, value in enumerate(values):
                weights[index] -= learning_rate * (error * value + l2 * weights[index])
            bias -= learning_rate * error

    return LogisticSimilarityModel(weights=weights, bias=bias, threshold=threshold)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)
