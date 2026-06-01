"""Hybrid semantic similarity engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .tokenizer import MixedTokenizer
from .vectorizer import cosine_similarity, tfidf_vectors, top_terms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNONYM_PATH = PROJECT_ROOT / "data" / "synonyms.json"
NEGATION_WORDS = (
    "不",
    "没",
    "没有",
    "不是",
    "无法",
    "不能",
    "并非",
    "讨厌",
    "no",
    "not",
    "never",
    "cannot",
    "can't",
    "dislike",
)
NEGATION_EXCEPTIONS = ("不错", "不但", "不仅")
OPPOSITE_GROUPS = (
    (("高", "贵", "昂贵", "高价"), ("低", "便宜", "低价")),
    (("快", "快速", "迅速", "很快", "飞快"), ("慢", "缓慢")),
    (("好", "不错", "优秀", "精彩", "好看", "好吃", "清晰", "有趣"), ("差", "糟糕", "无聊", "不好")),
    (("喜欢", "热爱", "喜爱", "感兴趣"), ("讨厌", "不喜欢", "厌恶")),
    (("成功", "解决", "完成", "正常", "可以"), ("失败", "无法", "不能", "没有", "不能完成")),
)


@dataclass(frozen=True)
class SimilarityResult:
    sentence_a: str
    sentence_b: str
    score: float
    label: str
    word_tfidf: float
    char_ngram_tfidf: float
    normalized_jaccard: float
    edit_similarity: float
    lexical_baseline: float
    semantic_rule_score: float
    fusion_before_penalty: float
    negation_mismatch: bool
    negation_penalty: float
    opposite_mismatch: bool
    opposite_penalty: float
    tokens_a: list[str]
    tokens_b: list[str]
    normalized_tokens_a: list[str]
    normalized_tokens_b: list[str]
    top_terms_a: list[tuple[str, float]]
    top_terms_b: list[tuple[str, float]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SentenceSimilarityEngine:
    """Calculate interpretable semantic similarity for sentence pairs."""

    def __init__(self, synonym_path: Path | str = DEFAULT_SYNONYM_PATH) -> None:
        self.synonym_map = self._load_synonyms(Path(synonym_path))
        self.tokenizer = MixedTokenizer(self.synonym_map)

    def compare(self, sentence_a: str, sentence_b: str) -> SimilarityResult:
        left = self.tokenizer.tokenize(sentence_a)
        right = self.tokenizer.tokenize(sentence_b)

        word_vectors = tfidf_vectors([left.normalized_tokens, right.normalized_tokens])
        char_vectors = tfidf_vectors([left.char_ngrams, right.char_ngrams])

        word_score = cosine_similarity(word_vectors[0], word_vectors[1])
        char_score = cosine_similarity(char_vectors[0], char_vectors[1])
        jaccard_score = self._jaccard(left.normalized_tokens, right.normalized_tokens)
        edit_score = self._edit_similarity(left.text, right.text)
        lexical_baseline = word_score
        semantic_rule_score = jaccard_score

        fusion_score = (
            0.50 * word_score
            + 0.15 * char_score
            + 0.25 * jaccard_score
            + 0.10 * edit_score
        )
        negation_mismatch = self._has_negation(left.text) != self._has_negation(right.text)
        negation_penalty = 1.0
        if negation_mismatch and (fusion_score >= 0.20 or jaccard_score >= 0.20):
            negation_penalty = 0.40

        opposite_mismatch = self._has_opposite_polarity(left.text, right.text)
        opposite_penalty = 0.40 if opposite_mismatch and fusion_score >= 0.20 else 1.0

        score = fusion_score * negation_penalty * opposite_penalty
        score = max(0.0, min(1.0, score))

        return SimilarityResult(
            sentence_a=sentence_a,
            sentence_b=sentence_b,
            score=round(score, 4),
            label=self._label(score),
            word_tfidf=round(word_score, 4),
            char_ngram_tfidf=round(char_score, 4),
            normalized_jaccard=round(jaccard_score, 4),
            edit_similarity=round(edit_score, 4),
            lexical_baseline=round(lexical_baseline, 4),
            semantic_rule_score=round(semantic_rule_score, 4),
            fusion_before_penalty=round(fusion_score, 4),
            negation_mismatch=negation_mismatch,
            negation_penalty=round(negation_penalty, 4),
            opposite_mismatch=opposite_mismatch,
            opposite_penalty=round(opposite_penalty, 4),
            tokens_a=left.tokens,
            tokens_b=right.tokens,
            normalized_tokens_a=left.normalized_tokens,
            normalized_tokens_b=right.normalized_tokens,
            top_terms_a=[(term, round(weight, 4)) for term, weight in top_terms(word_vectors[0])],
            top_terms_b=[(term, round(weight, 4)) for term, weight in top_terms(word_vectors[1])],
        )

    def compare_many(self, pairs: list[tuple[str, str]]) -> list[SimilarityResult]:
        return [self.compare(left, right) for left, right in pairs]

    def rank_candidates(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[SimilarityResult]:
        results = [self.compare(query, candidate) for candidate in candidates]
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def _load_synonyms(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as file:
            groups: dict[str, list[str]] = json.load(file)

        synonym_map: dict[str, str] = {}
        for canonical, words in groups.items():
            synonym_map[canonical.lower()] = canonical.lower()
            for word in words:
                synonym_map[word.lower()] = canonical.lower()
        return synonym_map

    @staticmethod
    def _jaccard(left: list[str], right: list[str]) -> float:
        left_set = set(left)
        right_set = set(right)
        if not left_set and not right_set:
            return 1.0
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / len(left_set | right_set)

    @staticmethod
    def _edit_similarity(left: str, right: str) -> float:
        if left == right:
            return 1.0
        if not left or not right:
            return 0.0
        distance = _levenshtein_distance(left, right)
        return 1 - distance / max(len(left), len(right))

    @staticmethod
    def _has_negation(text: str) -> bool:
        lowered = text.lower()
        for exception in NEGATION_EXCEPTIONS:
            lowered = lowered.replace(exception, "")
        return any(word in lowered for word in NEGATION_WORDS)

    @staticmethod
    def _has_opposite_polarity(left: str, right: str) -> bool:
        left = left.lower()
        right = right.lower()
        for positive_words, negative_words in OPPOSITE_GROUPS:
            left_positive = any(word in left for word in positive_words)
            left_negative = any(word in left for word in negative_words)
            right_positive = any(word in right for word in positive_words)
            right_negative = any(word in right for word in negative_words)
            if (left_positive and right_negative) or (left_negative and right_positive):
                return True
        return False

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.70:
            return "高相似"
        if score >= 0.42:
            return "中等相似"
        return "低相似"


def _levenshtein_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]
