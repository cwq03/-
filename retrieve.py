"""Top-K similar sentence retrieval demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from nlp_similarity import SentenceSimilarityEngine


DEFAULT_CANDIDATES = Path(__file__).resolve().parent / "data" / "candidates.txt"


def load_candidates(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="从候选句库中检索 Top-K 相似句")
    parser.add_argument("query", help="查询句子")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES, help="候选句库路径")
    parser.add_argument("-k", "--top-k", type=int, default=3, help="返回数量")
    args = parser.parse_args()

    engine = SentenceSimilarityEngine()
    results = engine.rank_candidates(args.query, load_candidates(args.candidates), args.top_k)
    print(f"查询：{args.query}")
    for index, result in enumerate(results, start=1):
        print(f"{index}. {result.sentence_b}  相似度：{result.score:.4f}（{result.label}）")


if __name__ == "__main__":
    main()
