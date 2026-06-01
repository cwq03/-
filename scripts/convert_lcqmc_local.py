"""Convert a locally downloaded LCQMC file to evaluate.py CSV format."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "lcqmc_sample.csv"


def read_jsonl(path: Path) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append(
                (
                    str(item.get("sentence1", item.get("sentence_a", item.get("text_a", "")))).strip(),
                    str(item.get("sentence2", item.get("sentence_b", item.get("text_b", "")))).strip(),
                    int(item.get("label", item.get("score", 0))),
                )
            )
    return rows


def read_delimited(path: Path) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt", ".data"} else ","
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        has_header = any(name in sample.lower() for name in ("sentence1", "sentence_a", "text_a"))
        if has_header:
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                sentence_a = row.get("sentence1") or row.get("sentence_a") or row.get("text_a") or ""
                sentence_b = row.get("sentence2") or row.get("sentence_b") or row.get("text_b") or ""
                label = row.get("label") or row.get("score") or "0"
                rows.append((sentence_a.strip(), sentence_b.strip(), int(label)))
        else:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                if len(row) >= 3:
                    label = int(row[-1].strip())
                    if len(row) == 3:
                        sentence_a, sentence_b = row[0], row[1]
                    else:
                        midpoint = (len(row) - 1) // 2
                        sentence_a = " ".join(part.strip() for part in row[:midpoint] if part.strip())
                        sentence_b = " ".join(part.strip() for part in row[midpoint:-1] if part.strip())
                    rows.append((sentence_a.strip(), sentence_b.strip(), label))
    return rows


def convert(input_path: Path, output_path: Path, sample_size: int, seed: int) -> None:
    if not input_path.exists():
        raise SystemExit(f"找不到输入文件：{input_path}")

    if input_path.suffix.lower() in {".jsonl", ".json"}:
        rows = read_jsonl(input_path)
    else:
        rows = read_delimited(input_path)

    rows = [(a, b, label) for a, b, label in rows if a and b and label in {0, 1}]
    if not rows:
        raise SystemExit("没有读取到有效样本，请确认输入文件是 LCQMC 的 tsv/csv/jsonl 格式。")

    random.Random(seed).shuffle(rows)
    sampled_rows = rows[:sample_size]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sentence_a", "sentence_b", "label"])
        writer.writerows(sampled_rows)

    positives = sum(label for _, _, label in sampled_rows)
    print(f"saved: {output_path}")
    print(f"rows: {len(sampled_rows)}")
    print(f"positive: {positives}")
    print(f"negative: {len(sampled_rows) - positives}")


def main() -> None:
    parser = argparse.ArgumentParser(description="转换本地 LCQMC 文件为项目评测 CSV")
    parser.add_argument("input", type=Path, help="本地 LCQMC 文件路径，支持 tsv/csv/jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 CSV 路径")
    parser.add_argument("--sample-size", type=int, default=1000, help="抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    convert(args.input, args.output, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
