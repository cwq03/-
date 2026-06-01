"""Download a small LCQMC sample and convert it to evaluate.py CSV format."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.parse
import urllib.request
from urllib.error import URLError
from pathlib import Path


API_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "lcqmc_sample.csv"


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict[str, object]]:
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{API_URL}?{params}", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["row"] for item in payload["rows"]]


def normalize_row(row: dict[str, object]) -> tuple[str, str, int]:
    sentence_a = str(row.get("sentence1", "")).strip()
    sentence_b = str(row.get("sentence2", "")).strip()
    label = int(row.get("score", row.get("label", 0)))
    return sentence_a, sentence_b, label


def download_sample(
    output: Path,
    sample_size: int,
    pool_size: int,
    page_size: int,
    seed: int,
) -> None:
    rows: list[tuple[str, str, int]] = []
    for offset in range(0, pool_size, page_size):
        current_page_size = min(page_size, pool_size - offset)
        for row in fetch_rows("C-MTEB/LCQMC", "default", "train", offset, current_page_size):
            rows.append(normalize_row(row))

    random.Random(seed).shuffle(rows)
    sampled_rows = rows[:sample_size]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sentence_a", "sentence_b", "label"])
        writer.writerows(sampled_rows)

    positives = sum(label for _, _, label in sampled_rows)
    negatives = len(sampled_rows) - positives
    print(f"saved: {output}")
    print(f"rows: {len(sampled_rows)}")
    print(f"positive: {positives}")
    print(f"negative: {negatives}")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 LCQMC 抽样数据并转换为项目评测 CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 CSV 路径")
    parser.add_argument("--sample-size", type=int, default=1000, help="抽样数量")
    parser.add_argument("--pool-size", type=int, default=5000, help="先下载的候选池数量")
    parser.add_argument("--page-size", type=int, default=100, help="每次 API 请求数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    try:
        download_sample(args.output, args.sample_size, args.pool_size, args.page_size, args.seed)
    except URLError as exc:
        print("下载 LCQMC 失败：无法访问 Hugging Face 数据接口。", file=sys.stderr)
        print(f"原因：{exc}", file=sys.stderr)
        print("解决方法：确认当前终端可以联网，或配置代理后重试。", file=sys.stderr)
        print("生成 data/lcqmc_sample.csv 后，再运行：python evaluate.py --data data/lcqmc_sample.csv", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
