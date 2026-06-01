"""Evaluate the supervised similarity fusion model."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import Metrics, load_dataset
from nlp_similarity import SentenceSimilarityEngine
from nlp_similarity.ml_model import DEFAULT_MODEL_PATH, LogisticSimilarityModel


DEFAULT_TEST_DATA = Path(__file__).resolve().parent / "data" / "lcqmc_test_sample.csv"


def evaluate_ml(path: Path, model_path: Path, threshold: float | None = None) -> Metrics:
    rows = load_dataset(path)
    engine = SentenceSimilarityEngine()
    model = LogisticSimilarityModel.load(model_path)
    if threshold is not None:
        model.threshold = threshold
    return evaluate_rows_ml(rows, engine, model)


def evaluate_rows_ml(
    rows: list[tuple[str, str, int]],
    engine: SentenceSimilarityEngine,
    model: LogisticSimilarityModel,
) -> Metrics:

    tp = fp = tn = fn = 0
    for sentence_a, sentence_b, label in rows:
        result = engine.compare(sentence_a, sentence_b)
        predicted = model.predict(result).label
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    total = len(rows)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(total, accuracy, precision, recall, f1, tp, fp, tn, fn)


def find_best_threshold(path: Path, model_path: Path, start: float = 0.10, stop: float = 0.90, step: float = 0.01) -> tuple[float, Metrics]:
    rows = load_dataset(path)
    engine = SentenceSimilarityEngine()
    model = LogisticSimilarityModel.load(model_path)
    best_threshold = start
    model.threshold = start
    best_metrics = evaluate_rows_ml(rows, engine, model)
    current = start
    while current <= stop + 1e-9:
        model.threshold = current
        metrics = evaluate_rows_ml(rows, engine, model)
        if (metrics.f1, metrics.accuracy) > (best_metrics.f1, best_metrics.accuracy):
            best_threshold = current
            best_metrics = metrics
        current += step
    return round(best_threshold, 2), best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="评测监督学习句子相似度融合模型")
    parser.add_argument("--data", type=Path, default=DEFAULT_TEST_DATA, help="测试数据 CSV")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="模型权重路径")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖模型文件中的预测阈值")
    parser.add_argument("--find-threshold", action="store_true", help="在 0.10 到 0.90 间搜索 F1 最优阈值")
    args = parser.parse_args()

    model = LogisticSimilarityModel.load(args.model)
    if args.find_threshold:
        threshold, metrics = find_best_threshold(args.data, args.model)
    else:
        threshold = args.threshold if args.threshold is not None else model.threshold
        metrics = evaluate_ml(args.data, args.model, args.threshold)
    print(f"数据集：{args.data}")
    print(f"模型：{args.model}")
    print(f"样本数：{metrics.total}")
    print(f"阈值：{threshold:.2f}")
    print(f"Accuracy：{metrics.accuracy:.4f}")
    print(f"Precision：{metrics.precision:.4f}")
    print(f"Recall：{metrics.recall:.4f}")
    print(f"F1：{metrics.f1:.4f}")
    print(f"TP={metrics.tp} FP={metrics.fp} TN={metrics.tn} FN={metrics.fn}")


if __name__ == "__main__":
    main()
