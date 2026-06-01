"""Command line interface for sentence similarity."""

from __future__ import annotations

import argparse
import json

from nlp_similarity import SentenceSimilarityEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="计算两个句子的语义相似度")
    parser.add_argument("sentence_a", help="第一个句子")
    parser.add_argument("sentence_b", help="第二个句子")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出完整结果")
    args = parser.parse_args()

    engine = SentenceSimilarityEngine()
    result = engine.compare(args.sentence_a, args.sentence_b)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    print(f"句子A：{result.sentence_a}")
    print(f"句子B：{result.sentence_b}")
    print(f"相似度：{result.score:.4f}（{result.label}）")
    print("方法对比：")
    print(f"  方法一 TF-IDF 词面基线：{result.lexical_baseline:.4f}")
    print(f"  方法二 同义词归一化规则：{result.semantic_rule_score:.4f}")
    print(f"  方法三 多特征融合结果：{result.score:.4f}")
    print("子特征：")
    print(f"词级 TF-IDF：{result.word_tfidf:.4f}")
    print(f"字 n-gram TF-IDF：{result.char_ngram_tfidf:.4f}")
    print(f"归一化 Jaccard：{result.normalized_jaccard:.4f}")
    print(f"编辑相似度：{result.edit_similarity:.4f}")
    print(f"否定极性不一致：{'是' if result.negation_mismatch else '否'}")
    print(f"否定惩罚系数：{result.negation_penalty:.2f}")
    print(f"反义极性冲突：{'是' if result.opposite_mismatch else '否'}")
    print(f"反义惩罚系数：{result.opposite_penalty:.2f}")
    print(f"分词A：{' / '.join(result.tokens_a)}")
    print(f"分词B：{' / '.join(result.tokens_b)}")


if __name__ == "__main__":
    main()
