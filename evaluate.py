"""Evaluate the sentence similarity system on a small labeled dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from nlp_similarity import SentenceSimilarityEngine


DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "test_pairs.csv"


@dataclass(frozen=True)
class Metrics:
    total: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


def load_dataset(path: Path) -> list[tuple[str, str, int]]:
    if not path.exists():
        print(f"找不到数据集文件：{path}", file=sys.stderr)
        print("如果要评测 LCQMC，请先运行：", file=sys.stderr)
        print("python scripts/download_lcqmc_sample.py --sample-size 1000 --pool-size 5000", file=sys.stderr)
        print("如果只想先测试系统，请运行：python evaluate.py --data data/test_pairs.csv", file=sys.stderr)
        raise SystemExit(1)

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [
            (row["sentence_a"], row["sentence_b"], int(row["label"]))
            for row in reader
        ]


def evaluate(path: Path, threshold: float) -> Metrics:
    engine = SentenceSimilarityEngine()
    rows = load_dataset(path)
    return evaluate_rows(rows, engine, threshold)


def evaluate_rows(
    rows: list[tuple[str, str, int]],
    engine: SentenceSimilarityEngine,
    threshold: float,
) -> Metrics:
    tp = fp = tn = fn = 0

    for sentence_a, sentence_b, label in rows:
        score = engine.compare(sentence_a, sentence_b).score
        predicted = int(score >= threshold)
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


def find_best_threshold(path: Path, start: float = 0.10, stop: float = 0.90, step: float = 0.01) -> tuple[float, Metrics]:
    engine = SentenceSimilarityEngine()
    rows = load_dataset(path)
    best_threshold = start
    best_metrics = evaluate_rows(rows, engine, start)
    current = start
    while current <= stop + 1e-9:
        metrics = evaluate_rows(rows, engine, current)
        if (metrics.f1, metrics.accuracy) > (best_metrics.f1, best_metrics.accuracy):
            best_threshold = current
            best_metrics = metrics
        current += step
    return round(best_threshold, 2), best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="在小型标注数据集上评测句子相似度系统")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET, help="CSV 数据集路径")
    parser.add_argument("--threshold", type=float, default=0.30, help="判定为相似的分数阈值")
    parser.add_argument("--find-threshold", action="store_true", help="在 0.10 到 0.90 间搜索 F1 最优阈值")
    args = parser.parse_args()

    if args.find_threshold:
        threshold, metrics = find_best_threshold(args.data)
    else:
        threshold = args.threshold
        metrics = evaluate(args.data, args.threshold)
    print(f"数据集：{args.data}")
    print(f"样本数：{metrics.total}")
    print(f"阈值：{threshold:.2f}")
    print(f"Accuracy：{metrics.accuracy:.4f}")
    print(f"Precision：{metrics.precision:.4f}")
    print(f"Recall：{metrics.recall:.4f}")
    print(f"F1：{metrics.f1:.4f}")
    print(f"TP={metrics.tp} FP={metrics.fp} TN={metrics.tn} FN={metrics.fn}")


if __name__ == "__main__":
    main()
