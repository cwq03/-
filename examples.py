"""Batch examples for the sentence similarity prototype."""

from __future__ import annotations

from nlp_similarity import SentenceSimilarityEngine


EXAMPLES = [
    ("我喜欢自然语言处理", "我对NLP很感兴趣"),
    ("这家餐厅味道不错", "这家店很好吃"),
    ("今天北京天气很好", "明天我要学习机器学习"),
    ("老师正在讲解语义相似度算法", "教师在介绍文本相似度方法"),
    ("我喜欢这门课程", "我不喜欢这门课程"),
    ("深度学习模型可以处理文本", "医生正在给病人看病"),
]


def main() -> None:
    engine = SentenceSimilarityEngine()
    for index, (left, right) in enumerate(EXAMPLES, start=1):
        result = engine.compare(left, right)
        print(f"{index}. {result.score:.4f} {result.label}")
        print(f"   A: {left}")
        print(f"   B: {right}")
        print(
            "   features:",
            f"word={result.word_tfidf:.4f}",
            f"char={result.char_ngram_tfidf:.4f}",
            f"jaccard={result.normalized_jaccard:.4f}",
            f"edit={result.edit_similarity:.4f}",
        )


if __name__ == "__main__":
    main()
