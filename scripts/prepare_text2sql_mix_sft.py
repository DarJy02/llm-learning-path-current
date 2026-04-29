from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "LLaMA-Factory" / "data"
PROJECT_DIR = ROOT / "项目二"
EVAL_DIR = PROJECT_DIR / "评测集"
REPORT_DIR = PROJECT_DIR / "数据处理"


SYSTEM = "你是一个 Text-to-SQL 助手。请根据数据库 schema 和用户问题生成可执行 SQL。"
INSTRUCTION = "根据数据库 schema 和用户问题生成 SQL。只输出 SQL，不要解释。"


def is_select_like(sql: str) -> bool:
    normalized = sql.strip().lower()
    return normalized.startswith("select") or normalized.startswith("with")


def build_input(sample: dict[str, Any]) -> str:
    schema = (sample.get("schema_context") or "").strip()
    question = (sample.get("instruction") or "").strip()
    dialect = (sample.get("dialect") or "").strip()
    difficulty = (sample.get("difficulty") or "").strip()

    meta_parts = []
    if dialect:
        meta_parts.append(f"SQL Dialect: {dialect}")
    if difficulty:
        meta_parts.append(f"Difficulty: {difficulty}")

    meta = "\n".join(meta_parts)
    if meta:
        meta = f"{meta}\n\n"

    return f"{meta}Schema:\n{schema}\n\nQuestion:\n{question}"


def convert_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": INSTRUCTION,
        "input": build_input(sample),
        "output": (sample.get("sql") or "").strip(),
        "system": SYSTEM,
        "history": [],
        "metadata": {
            "id": sample.get("id"),
            "source": sample.get("source"),
            "dialect": sample.get("dialect"),
            "difficulty": sample.get("difficulty"),
        },
    }


def collect_samples(
    split: str,
    limit: int,
    seed: int,
    select_only: bool,
    allowed_dialects: set[str] | None,
    allowed_difficulties: set[str] | None,
) -> list[dict[str, Any]]:
    dataset = load_dataset("DanielRegaladoCardoso/text-to-sql-mix-v2", split=split)
    dataset = dataset.shuffle(seed=seed)
    samples: list[dict[str, Any]] = []

    for raw in dataset:
        sql = (raw.get("sql") or "").strip()
        question = (raw.get("instruction") or "").strip()
        schema = (raw.get("schema_context") or "").strip()
        dialect = (raw.get("dialect") or "").strip()
        difficulty = (raw.get("difficulty") or "").strip()

        if not sql or not question or not schema:
            continue
        if select_only and not is_select_like(sql):
            continue
        if allowed_dialects and dialect not in allowed_dialects:
            continue
        if allowed_difficulties and difficulty not in allowed_difficulties:
            continue

        converted = convert_sample(raw)
        if len(samples) < limit:
            samples.append(converted)
        else:
            break

    return samples


def collect_samples_from_parquet(
    paths: list[Path],
    limit: int,
    seed: int,
    select_only: bool,
    allowed_dialects: set[str] | None,
    allowed_difficulties: set[str] | None,
) -> list[dict[str, Any]]:
    if not paths:
        raise FileNotFoundError("No local parquet files matched.")

    frames = [pd.read_parquet(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    samples: list[dict[str, Any]] = []
    for raw in data.to_dict(orient="records"):
        sql = (raw.get("sql") or "").strip()
        question = (raw.get("instruction") or "").strip()
        schema = (raw.get("schema_context") or "").strip()
        dialect = (raw.get("dialect") or "").strip()
        difficulty = (raw.get("difficulty") or "").strip()

        if not sql or not question or not schema:
            continue
        if select_only and not is_select_like(sql):
            continue
        if allowed_dialects and dialect not in allowed_dialects:
            continue
        if allowed_difficulties and difficulty not in allowed_difficulties:
            continue

        samples.append(convert_sample(raw))
        if len(samples) >= limit:
            break

    return samples


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, train: list[dict[str, Any]], eval_records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = train + eval_records
    dialects = Counter(record["metadata"].get("dialect") or "unknown" for record in combined)
    difficulties = Counter(record["metadata"].get("difficulty") or "unknown" for record in combined)
    sources = Counter(record["metadata"].get("source") or "unknown" for record in combined)

    source_lines = "\n".join(f"- {name}: {count}" for name, count in sources.most_common(20))
    dialect_lines = "\n".join(f"- {name}: {count}" for name, count in dialects.most_common())
    difficulty_lines = "\n".join(f"- {name}: {count}" for name, count in difficulties.most_common())

    text = f"""# text2sql_mix_sft 数据处理报告

## 数据来源

- Hugging Face dataset: DanielRegaladoCardoso/text-to-sql-mix-v2
- split: {args.split}
- eval_split: {args.eval_split}
- train output: LLaMA-Factory/data/{args.output_name}
- eval output: 项目二/评测集/{args.eval_name}

## 处理参数

- max_train: {args.max_train}
- max_eval: {args.max_eval}
- seed: {args.seed}
- select_only: {args.select_only}
- dialects: {args.dialects or "all"}
- difficulties: {args.difficulties or "all"}

## 输出规模

- train records: {len(train)}
- eval records: {len(eval_records)}
- total records: {len(combined)}

## dialect 分布

{dialect_lines}

## difficulty 分布

{difficulty_lines}

## source Top 20

{source_lines}

## LLaMA-Factory 注册名

```text
text2sql_mix_sft
```

## 样本格式

```json
{json.dumps(train[0] if train else {}, ensure_ascii=False, indent=2)}
```
"""
    path.write_text(text, encoding="utf-8")


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def expand_glob(pattern: str | None) -> list[Path]:
    if not pattern:
        return []
    return [Path(path) for path in glob.glob(pattern)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare text-to-sql-mix-v2 for LLaMA-Factory SFT.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--max-eval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select-only", action="store_true", default=True)
    parser.add_argument("--include-dml", dest="select_only", action="store_false")
    parser.add_argument("--dialects", default="generic")
    parser.add_argument("--difficulties", default="easy,medium")
    parser.add_argument("--output-name", default="text2sql_mix_sft.json")
    parser.add_argument("--eval-name", default="text2sql_mix_eval.jsonl")
    parser.add_argument("--report-name", default="text2sql_mix_sft_report.md")
    parser.add_argument("--local-train-glob", default=None)
    parser.add_argument("--local-eval-glob", default=None)
    args = parser.parse_args()

    allowed_dialects = parse_csv_set(args.dialects)
    allowed_difficulties = parse_csv_set(args.difficulties)

    local_train_paths = expand_glob(args.local_train_glob)
    local_eval_paths = expand_glob(args.local_eval_glob)

    if local_train_paths:
        train = collect_samples_from_parquet(
            paths=local_train_paths,
            limit=args.max_train,
            seed=args.seed,
            select_only=args.select_only,
            allowed_dialects=allowed_dialects,
            allowed_difficulties=allowed_difficulties,
        )
    else:
        train = collect_samples(
            split=args.split,
            limit=args.max_train,
            seed=args.seed,
            select_only=args.select_only,
            allowed_dialects=allowed_dialects,
            allowed_difficulties=allowed_difficulties,
        )

    if local_eval_paths:
        eval_records = collect_samples_from_parquet(
            paths=local_eval_paths,
            limit=args.max_eval,
            seed=args.seed + 1,
            select_only=args.select_only,
            allowed_dialects=allowed_dialects,
            allowed_difficulties=allowed_difficulties,
        )
    else:
        eval_records = collect_samples(
            split=args.eval_split,
            limit=args.max_eval,
            seed=args.seed + 1,
            select_only=args.select_only,
            allowed_dialects=allowed_dialects,
            allowed_difficulties=allowed_difficulties,
        )

    if len(train) < args.max_train:
        raise RuntimeError(f"Only collected {len(train)} train records, expected {args.max_train}.")
    if len(eval_records) < args.max_eval:
        raise RuntimeError(f"Only collected {len(eval_records)} eval records, expected {args.max_eval}.")

    write_json(DATA_DIR / args.output_name, train)
    write_jsonl(EVAL_DIR / args.eval_name, eval_records)
    write_report(REPORT_DIR / args.report_name, train, eval_records, args)

    print(f"wrote train: {DATA_DIR / args.output_name} ({len(train)} records)")
    print(f"wrote eval: {EVAL_DIR / args.eval_name} ({len(eval_records)} records)")
    print(f"wrote report: {REPORT_DIR / args.report_name}")


if __name__ == "__main__":
    main()
