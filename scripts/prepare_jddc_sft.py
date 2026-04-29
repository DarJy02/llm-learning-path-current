#!/usr/bin/env python
"""Convert JDDC chat logs into LLaMA-Factory SFT data.

The source file is a tab-separated turn-level chat log. This script builds
Alpaca-style samples with dialogue history, which LLaMA-Factory can load from
data/dataset_info.json without custom converters.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "jddc2019-3th-retrieve-model-master"
    / "jddc2019-3th-retrieve-model-master"
    / "data"
    / "chat_0.1per.txt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "LLaMA-Factory" / "data" / "ecommerce_customer_sft.json"
DEFAULT_DATASET_INFO = PROJECT_ROOT / "LLaMA-Factory" / "data" / "dataset_info.json"

SYSTEM_PROMPT = (
    "你是京东电商场景的中文客服助手。请根据顾客问题和已有对话上下文，"
    "给出礼貌、准确、可执行的客服回复。"
)

NULL_VALUES = {"", "nan", "null", "none", "NULL", "NaN", "NAN"}

NOISE_PATTERNS = (
    re.compile(r"^系统提示[:：]"),
    re.compile(r"^#E-[A-Za-z0-9_-]+"),
    re.compile(r"客服MM还没有等到您的回复"),
    re.compile(r"本次会话将在两分钟后自动切断"),
    re.compile(r"本次会话已经中断"),
)

GREETING_PATTERNS = (
    re.compile(r"^人工客服.*你好$"),
    re.compile(r"^用户发起转人工$"),
)


def normalize_text(text: str) -> str:
    """Normalize one chat turn while preserving anonymized placeholders."""
    text = text.strip()
    if text in NULL_VALUES:
        return ""

    text = text.replace("\ufeff", "")
    text = text.replace("!@@@!", "\n")
    text = text.replace("<sep>", "\n")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise(text: str) -> bool:
    if not text:
        return True
    return any(pattern.search(text) for pattern in NOISE_PATTERNS)


def read_sessions(source: Path) -> tuple[dict[str, list[dict[str, str]]], Counter[str]]:
    sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
    stats: Counter[str] = Counter()

    with source.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) != 7:
                stats["bad_tab_lines"] += 1
                continue

            session_id = parts[0]
            sender = parts[2]
            raw_text = parts[6]
            text = normalize_text(raw_text)

            if sender not in {"0", "1"}:
                stats["bad_sender"] += 1
                continue
            if is_noise(text):
                stats["empty_or_noise_turns"] += 1
                continue

            role = "assistant" if sender == "1" else "user"
            if role == "user" and any(pattern.search(text) for pattern in GREETING_PATTERNS):
                stats["dropped_user_greetings"] += 1
                continue

            sessions[session_id].append({"role": role, "content": text, "line_no": str(line_no)})
            stats[f"{role}_turns"] += 1

    return sessions, stats


def merge_consecutive_same_role(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["content"] = f"{merged[-1]['content']}\n{turn['content']}"
            merged[-1]["line_no"] = f"{merged[-1]['line_no']},{turn['line_no']}"
        else:
            merged.append(dict(turn))
    return merged


def trim_history(history: list[list[str]], max_history: int) -> list[list[str]]:
    if max_history <= 0:
        return []
    return history[-max_history:]


def build_samples(
    sessions: dict[str, list[dict[str, str]]],
    max_history: int,
    min_user_chars: int,
    min_assistant_chars: int,
    max_output_chars: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    samples: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for session_id, raw_turns in sessions.items():
        turns = merge_consecutive_same_role(raw_turns)
        history: list[list[str]] = []
        pending_user = ""

        for turn in turns:
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                pending_user = f"{pending_user}\n{content}".strip() if pending_user else content
                continue

            if not pending_user:
                stats["assistant_without_user"] += 1
                continue

            if len(pending_user) < min_user_chars:
                stats["too_short_user"] += 1
            elif len(content) < min_assistant_chars:
                stats["too_short_assistant"] += 1
            elif len(content) > max_output_chars:
                stats["too_long_assistant"] += 1
            else:
                samples.append(
                    {
                        "instruction": pending_user,
                        "input": "",
                        "output": content,
                        "system": SYSTEM_PROMPT,
                        "history": trim_history(history, max_history),
                        "metadata": {
                            "source": "jddc_chat_0.1per",
                            "session_id": session_id,
                            "line_no": turn["line_no"],
                        },
                    }
                )
                stats["samples"] += 1

            history.append([pending_user, content])
            pending_user = ""

    return samples, stats


def write_dataset_info(dataset_info_path: Path, dataset_name: str, output_path: Path) -> None:
    with dataset_info_path.open("r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    dataset_info[dataset_name] = {
        "file_name": output_path.name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
            "history": "history",
        },
    }

    with dataset_info_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--dataset-name", default="ecommerce_customer_sft")
    parser.add_argument("--max-history", type=int, default=4)
    parser.add_argument("--min-user-chars", type=int, default=2)
    parser.add_argument("--min-assistant-chars", type=int, default=2)
    parser.add_argument("--max-output-chars", type=int, default=800)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means keep all samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()

    sessions, read_stats = read_sessions(args.source)
    samples, build_stats = build_samples(
        sessions=sessions,
        max_history=args.max_history,
        min_user_chars=args.min_user_chars,
        min_assistant_chars=args.min_assistant_chars,
        max_output_chars=args.max_output_chars,
    )

    if args.max_samples and len(samples) > args.max_samples:
        random.Random(args.seed).shuffle(samples)
        samples = samples[: args.max_samples]
        samples.sort(key=lambda item: (item["metadata"]["session_id"], item["metadata"]["line_no"]))
        build_stats["sampled_limit"] = args.max_samples

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if not args.skip_register:
        write_dataset_info(args.dataset_info, args.dataset_name, args.output)

    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "dataset_name": args.dataset_name,
        "sessions": len(sessions),
        "samples": len(samples),
        "read_stats": dict(read_stats),
        "build_stats": dict(build_stats),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
