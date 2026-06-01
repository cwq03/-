"""Compare rule-based fusion and supervised fusion on the same dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import evaluate
from evaluate_ml import evaluate_ml
from nlp_similarity.ml_model import DEFAULT_MODEL_PATH


DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "lcqmc_test_sample.csv"


def print_row(name: str, accuracy: float, precision: float, recall: float, f1: float) -> None:
    print(f"{name:<22} {accuracy:>8.4f} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="对比规则融合模型和监督学习融合模型")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="测试数据 CSV")
    parser.add_argument("--rule-threshold", type=float, default=0.40, help="规则模型阈值")
    parser.add_argument("--ml-threshold", type=float, default=0.16, help="监督模型阈值")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="监督模型权重文件")
    args = parser.parse_args()

    rule_metrics = evaluate(args.data, args.rule_threshold)
    ml_metrics = evaluate_ml(args.data, args.model, args.ml_threshold)

    print(f"数据集：{args.data}")
    print(f"样本数：{rule_metrics.total}")
    print("方法                     Accuracy  Precision   Recall       F1")
    print("-" * 64)
    print_row("规则多特征融合", rule_metrics.accuracy, rule_metrics.precision, rule_metrics.recall, rule_metrics.f1)
    print_row("监督学习融合", ml_metrics.accuracy, ml_metrics.precision, ml_metrics.recall, ml_metrics.f1)


if __name__ == "__main__":
    main()
