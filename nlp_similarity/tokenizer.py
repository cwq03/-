"""Tokenization utilities for mixed Chinese and English sentences."""

from __future__ import annotations

import re
from dataclasses import dataclass


_EN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


DEFAULT_STOPWORDS = {
    "的",
    "了",
    "和",
    "是",
    "我",
    "你",
    "他",
    "她",
    "它",
    "们",
    "在",
    "对",
    "这",
    "个",
    "部",
    "款",
    "台",
    "门",
    "家",
    "一",
    "正在",
    "已经",
    "可以",
    "能够",
    "需要",
    "应该",
    "准备",
    "想",
    "要",
    "属于",
    "包含",
    "很",
    "非常",
    "the",
    "a",
    "an",
    "is",
    "are",
    "am",
    "to",
    "of",
    "and",
    "in",
    "for",
}


@dataclass(frozen=True)
class TokenizedSentence:
    """Structured tokenization result used by the similarity engine."""

    text: str
    tokens: list[str]
    normalized_tokens: list[str]
    char_ngrams: list[str]


class MixedTokenizer:
    """A small tokenizer designed for Chinese-English course prototypes.

    Chinese text is segmented by dictionary matching first and then by single
    character fallback. English text is lower-cased and split with a word regex.
    """

    def __init__(
        self,
        synonym_map: dict[str, str] | None = None,
        stopwords: set[str] | None = None,
    ) -> None:
        self.synonym_map = synonym_map or {}
        self.stopwords = stopwords or DEFAULT_STOPWORDS
        dictionary = set(self.synonym_map) | set(self.synonym_map.values())
        dictionary.update(
            {
                "自然语言处理",
                "语义相似度",
                "机器学习",
                "深度学习",
                "人工智能",
                "文本分类",
                "情感分析",
                "信息检索",
                "神经网络",
                "预训练模型",
                "大语言模型",
            }
        )
        self._dictionary = sorted(dictionary, key=len, reverse=True)

    def tokenize(self, text: str) -> TokenizedSentence:
        cleaned = self._normalize_text(text)
        tokens = self._split_mixed_text(cleaned)
        tokens = [token for token in tokens if token and token not in self.stopwords]
        normalized_tokens = [self.synonym_map.get(token, token) for token in tokens]
        char_ngrams = self.char_ngrams(cleaned, min_n=2, max_n=3)
        return TokenizedSentence(
            text=cleaned,
            tokens=tokens,
            normalized_tokens=normalized_tokens,
            char_ngrams=char_ngrams,
        )

    def _normalize_text(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _split_mixed_text(self, text: str) -> list[str]:
        tokens: list[str] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char.isspace() or re.match(r"[^\w\u4e00-\u9fff]", char):
                index += 1
                continue

            match = _EN_WORD_RE.match(text, index)
            if match:
                tokens.append(match.group(0))
                index = match.end()
                continue

            if _CJK_RE.match(char):
                phrase = self._match_dictionary(text, index)
                if phrase:
                    tokens.append(phrase)
                    index += len(phrase)
                else:
                    tokens.append(char)
                    index += 1
                continue

            tokens.append(char)
            index += 1
        return tokens

    def _match_dictionary(self, text: str, index: int) -> str | None:
        for phrase in self._dictionary:
            if text.startswith(phrase, index):
                return phrase
        return None

    @staticmethod
    def char_ngrams(text: str, min_n: int = 2, max_n: int = 3) -> list[str]:
        compact = re.sub(r"\s+", "", text)
        compact = re.sub(r"[^\w\u4e00-\u9fff]", "", compact)
        if not compact:
            return []
        grams: list[str] = []
        for n in range(min_n, max_n + 1):
            if len(compact) < n:
                continue
            grams.extend(compact[index : index + n] for index in range(len(compact) - n + 1))
        return grams
