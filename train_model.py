"""Train a supervised sentence similarity fusion model."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import load_dataset
from nlp_similarity.ml_model import DEFAULT_MODEL_PATH, train_logistic_model


DEFAULT_TRAIN_DATA = Path(__file__).resolve().parent / "data" / "lcqmc_sample.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 LCQMC 抽样数据训练监督学习融合模型")
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN_DATA, help="训练数据 CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="模型权重输出路径")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="学习率")
    parser.add_argument("--threshold", type=float, default=0.50, help="预测阈值")
    args = parser.parse_args()

    rows = load_dataset(args.data)
    model = train_logistic_model(
        rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
    )
    model.save(args.output)

    print(f"训练数据：{args.data}")
    print(f"样本数：{len(rows)}")
    print(f"模型已保存：{args.output}")
    print("特征权重：")
    for name, weight in zip(model.feature_names, model.weights):
        print(f"  {name}: {weight:.4f}")
    print(f"bias: {model.bias:.4f}")
    print(f"threshold: {model.threshold:.2f}")


if __name__ == "__main__":
    main()
