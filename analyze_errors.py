"""Export error cases for qualitative analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluate import load_dataset
from nlp_similarity import SentenceSimilarityEngine
from nlp_similarity.ml_model import DEFAULT_MODEL_PATH, LogisticSimilarityModel


DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "lcqmc_test_sample.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "docs" / "error_cases.csv"


def classify_error(true_label: int, predicted_label: int) -> str:
    if true_label == 0 and predicted_label == 1:
        return "FP_误判为相似"
    if true_label == 1 and predicted_label == 0:
        return "FN_漏判相似"
    return "正确"


def analyze_errors(
    data_path: Path,
    model_path: Path,
    output_path: Path,
    threshold: float,
    limit: int,
) -> None:
    rows = load_dataset(data_path)
    engine = SentenceSimilarityEngine()
    model = LogisticSimilarityModel.load(model_path)
    model.threshold = threshold

    error_rows: list[dict[str, object]] = []
    false_positive = 0
    false_negative = 0

    for sentence_a, sentence_b, true_label in rows:
        result = engine.compare(sentence_a, sentence_b)
        prediction = model.predict(result)
        if prediction.label == true_label:
            continue

        error_type = classify_error(true_label, prediction.label)
        false_positive += int(error_type.startswith("FP"))
        false_negative += int(error_type.startswith("FN"))
        error_rows.append(
            {
                "error_type": error_type,
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
                "true_label": true_label,
                "predicted_label": prediction.label,
                "ml_probability": prediction.probability,
                "rule_score": result.score,
                "word_tfidf": result.word_tfidf,
                "char_ngram_tfidf": result.char_ngram_tfidf,
                "normalized_jaccard": result.normalized_jaccard,
                "edit_similarity": result.edit_similarity,
                "negation_mismatch": result.negation_mismatch,
                "opposite_mismatch": result.opposite_mismatch,
                "tokens_a": " / ".join(result.normalized_tokens_a),
                "tokens_b": " / ".join(result.normalized_tokens_b),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "error_type",
        "sentence_a",
        "sentence_b",
        "true_label",
        "predicted_label",
        "ml_probability",
        "rule_score",
        "word_tfidf",
        "char_ngram_tfidf",
        "normalized_jaccard",
        "edit_similarity",
        "negation_mismatch",
        "opposite_mismatch",
        "tokens_a",
        "tokens_b",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(error_rows[:limit])

    print(f"数据集：{data_path}")
    print(f"模型：{model_path}")
    print(f"阈值：{threshold:.2f}")
    print(f"错误样例总数：{len(error_rows)}")
    print(f"FP 误判为相似：{false_positive}")
    print(f"FN 漏判相似：{false_negative}")
    print(f"已导出：{output_path}")
    print(f"导出数量：{min(limit, len(error_rows))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出监督学习模型误判样例")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="测试数据 CSV")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="模型权重路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="错误样例输出 CSV")
    parser.add_argument("--threshold", type=float, default=0.16, help="监督模型预测阈值")
    parser.add_argument("--limit", type=int, default=80, help="最多导出的错误样例数量")
    args = parser.parse_args()
    analyze_errors(args.data, args.model, args.output, args.threshold, args.limit)


if __name__ == "__main__":
    main()
