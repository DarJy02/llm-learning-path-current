#!/usr/bin/env python
"""Clean privacy/noise artifacts in ecommerce_customer_sft_v6_merged.

This pass is intentionally conservative: keep business numbers such as 7 days,
24 hours, 618 and discount amounts, but remove service-number artifacts,
asterisk masks, raw long identifiers and digits wrapped around placeholders.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "LLaMA-Factory" / "data" / "ecommerce_customer_sft_v6_merged.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "LLaMA-Factory" / "data" / "ecommerce_customer_sft_v6_cleaned.json"
DEFAULT_REPORT = PROJECT_ROOT / "项目一" / "数据处理" / "ecommerce_customer_sft_v6_cleaned_report.md"
DEFAULT_DATASET_INFO = PROJECT_ROOT / "LLaMA-Factory" / "data" / "dataset_info.json"


Rule = tuple[str, re.Pattern[str], str]


RULES: list[Rule] = [
    (
        "remove_jd_service_number",
        re.compile(r"(京东客服)\s*[0-9０-９]*\*+\s*号"),
        r"\1",
    ),
    (
        "remove_generic_service_number",
        re.compile(r"(客服)\s*[0-9０-９]*\*+\s*号"),
        r"\1",
    ),
    (
        "remove_plain_service_number",
        re.compile(r"(京东客服|客服)\s*[0-9０-９]{1,6}\s*号"),
        r"\1",
    ),
    (
        "remove_mixed_placeholder_service_number",
        re.compile(r"(京东(?:家电)?客服|客服)\s*[0-9０-９]*\s*\[[^\]]+x\]\s*[0-9０-９]*\s*号"),
        r"\1",
    ),
    (
        "remove_placeholder_service_number",
        re.compile(r"(京东(?:家电)?客服|客服)\s*(?:\[[^\]]+x\]|[0-9０-９]+)\s*号"),
        r"\1",
    ),
    (
        "collapse_mixed_around_placeholder",
        re.compile(r"(?<!\d)[0-9０-９\*]+\s*(\[[^\]]+x\])\s*[0-9０-９\*]+(?!\d)"),
        r"\1",
    ),
    (
        "collapse_digits_around_placeholder",
        re.compile(r"(?<!\d)[0-9０-９]+\s*(\[[^\]]+x\])\s*[0-9０-９]+(?!\d)"),
        r"\1",
    ),
    (
        "collapse_digits_before_placeholder",
        re.compile(r"(?<!\d)[0-9０-９]+\s*(\[[^\]]+x\])"),
        r"\1",
    ),
    (
        "collapse_digits_after_placeholder",
        re.compile(r"(\[[^\]]+x\])\s*[0-9０-９]+(?!\d)"),
        r"\1",
    ),
    (
        "normalize_order_id",
        re.compile(r"(订单(?:号|编号)?\s*[:：]?)\s*[A-Za-z0-9][A-Za-z0-9\-]{5,}"),
        r"\1[订单x]",
    ),
    (
        "normalize_phone_number",
        re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        r"[电话x]",
    ),
    (
        "normalize_long_number",
        re.compile(r"(?<![\d年月日点时分秒\-])\d{8,}(?![\d年月日点时分秒\-])"),
        r"[数字x]",
    ),
    (
        "remove_star_masks",
        re.compile(r"\*{2,}"),
        "",
    ),
    (
        "remove_leftover_separators",
        re.compile(r"!@@@!|<sep>"),
        " ",
    ),
    (
        "remove_null_token",
        re.compile(r"\bNULL\b", re.IGNORECASE),
        "",
    ),
]


def normalize_spacing(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"（\s*(\[[^\]]+x\])\s*）", r"（\1）", text)
    text = re.sub(r"\(\s*(\[[^\]]+x\])\s*\)", r"(\1)", text)
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)
    text = re.sub(r"([（(])\s+", r"\1", text)
    text = re.sub(r"\s+([）)])", r"\1", text)
    text = re.sub(r"([。！？])\1+", r"\1", text)
    text = re.sub(r"[?？]{2,}", "?", text)
    text = re.sub(r"[!！]{2,}", "!", text)
    text = re.sub(r"^[?？]\s*", "", text)
    return text.strip()


def clean_text(text: str, stats: Counter[str]) -> str:
    original = text
    for name, pattern, repl in RULES:
        text, count = pattern.subn(repl, text)
        if count:
            stats[name] += count

    text = normalize_spacing(text)
    if text != original:
        stats["text_fields_changed"] += 1
    return text


def clean_history(history: Any, stats: Counter[str]) -> list[list[str]]:
    if not isinstance(history, list):
        stats["bad_history_type"] += 1
        return []

    cleaned: list[list[str]] = []
    for item in history:
        if not isinstance(item, list) or len(item) != 2:
            stats["bad_history_item"] += 1
            continue
        user = clean_text(str(item[0]), stats)
        assistant = clean_text(str(item[1]), stats)
        if user and assistant:
            cleaned.append([user, assistant])
        else:
            stats["empty_history_item_after_clean"] += 1

    return cleaned


def clean_sample(sample: dict[str, Any], stats: Counter[str]) -> dict[str, Any]:
    cleaned = deepcopy(sample)
    for key in ("instruction", "input", "output", "system"):
        value = cleaned.get(key, "")
        if not isinstance(value, str):
            stats[f"bad_{key}_type"] += 1
            value = str(value) if value is not None else ""
        cleaned[key] = clean_text(value, stats)

    cleaned["history"] = clean_history(cleaned.get("history", []), stats)
    return cleaned


def sample_key(sample: dict[str, Any]) -> str:
    return json.dumps(
        {
            "instruction": sample.get("instruction", ""),
            "input": sample.get("input", ""),
            "output": sample.get("output", ""),
            "history": sample.get("history", []),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def validate(samples: list[dict[str, Any]]) -> Counter[str]:
    stats: Counter[str] = Counter()
    required = ("instruction", "input", "output", "system", "history")
    for sample in samples:
        for key in required:
            if key not in sample:
                stats[f"missing_{key}"] += 1
        if not isinstance(sample.get("history", []), list):
            stats["history_not_list"] += 1
        if not sample.get("instruction"):
            stats["empty_instruction"] += 1
        if not sample.get("output"):
            stats["empty_output"] += 1
    return stats


def dirty_counts(samples: list[dict[str, Any]]) -> Counter[str]:
    patterns = {
        "jd_service_no": re.compile(r"京东客服\s*[0-9０-９\*]+\s*号|客服\s*[0-9０-９\*]+\s*号"),
        "placeholder_service_no": re.compile(r"京东(?:家电)?客服\s*\[[^\]]+x\]\s*号|客服\s*\[[^\]]+x\]\s*号"),
        "star_masks": re.compile(r"\*{2,}"),
        "digits_around_placeholder": re.compile(r"[0-9０-９]+\s*\[[^\]]+x\]\s*[0-9０-９]+"),
        "mixed_around_placeholder": re.compile(r"[0-9０-９\*]+\s*\[[^\]]+x\]\s*[0-9０-９\*]+"),
        "raw_order_id": re.compile(r"订单(?:号|编号)?\s*[:：]?\s*[A-Za-z0-9][A-Za-z0-9\-]{5,}"),
        "phone_like": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        "long_number": re.compile(r"(?<![\d年月日点时分秒\-])\d{8,}(?![\d年月日点时分秒\-])"),
        "sep_or_null": re.compile(r"!@@@!|<sep>|\bNULL\b", re.IGNORECASE),
    }
    counts: Counter[str] = Counter()
    for sample in samples:
        text = "\n".join(
            [
                sample.get("instruction", ""),
                sample.get("input", ""),
                sample.get("output", ""),
                sample.get("system", ""),
                json.dumps(sample.get("history", []), ensure_ascii=False),
            ]
        )
        for name, pattern in patterns.items():
            if pattern.search(text):
                counts[name] += 1
    return counts


def update_dataset_info(dataset_info_path: Path, dataset_name: str, output_path: Path) -> None:
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


def write_report(
    report_path: Path,
    input_path: Path,
    output_path: Path,
    before_count: int,
    after_count: int,
    clean_stats: Counter[str],
    before_dirty: Counter[str],
    after_dirty: Counter[str],
    validation: Counter[str],
    examples: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# ecommerce_customer_sft_v6_cleaned 清洗报告",
        "",
        "## 文件",
        "",
        f"- 输入：`{input_path.as_posix()}`",
        f"- 输出：`{output_path.as_posix()}`",
        "",
        "## 样本统计",
        "",
        f"- 清洗前样本数：{before_count}",
        f"- 清洗后样本数：{after_count}",
        f"- 删除空 instruction/output 样本：{clean_stats.get('drop_empty_instruction_or_output', 0)}",
        f"- 删除完全重复样本：{clean_stats.get('drop_exact_duplicate', 0)}",
        "",
        "## 清洗规则命中次数",
        "",
        "| 规则 | 次数 |",
        "| --- | ---: |",
    ]
    for key, value in clean_stats.most_common():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## 脏模式样本数对比", "", "| 模式 | 清洗前 | 清洗后 |", "| --- | ---: | ---: |"])
    for key in sorted(set(before_dirty) | set(after_dirty)):
        lines.append(f"| {key} | {before_dirty.get(key, 0)} | {after_dirty.get(key, 0)} |")

    lines.extend(["", "## 合规检查", "", "| 项 | 数量 |", "| --- | ---: |"])
    if validation:
        for key, value in validation.most_common():
            lines.append(f"| {key} | {value} |")
    else:
        lines.append("| 字段/类型/空值问题 | 0 |")

    lines.extend(["", "## 修改示例", ""])
    for idx, (before, after) in enumerate(examples[:20], 1):
        lines.extend(
            [
                f"### 示例 {idx}",
                "",
                "**Before**",
                "",
                "```text",
                f"instruction: {before.get('instruction', '')}",
                f"output: {before.get('output', '')}",
                f"history: {before.get('history', [])}",
                "```",
                "",
                "**After**",
                "",
                "```text",
                f"instruction: {after.get('instruction', '')}",
                f"output: {after.get('output', '')}",
                f"history: {after.get('history', [])}",
                "```",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--dataset-name", default="ecommerce_customer_sft_v6_cleaned")
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("Input dataset must be a JSON list.")

    before_dirty = dirty_counts(data)
    clean_stats: Counter[str] = Counter()
    cleaned: list[dict[str, Any]] = []
    examples: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()

    for sample in data:
        if not isinstance(sample, dict):
            clean_stats["drop_non_dict_sample"] += 1
            continue
        before = deepcopy(sample)
        after = clean_sample(sample, clean_stats)
        if not after.get("instruction") or not after.get("output"):
            clean_stats["drop_empty_instruction_or_output"] += 1
            continue
        key = sample_key(after)
        if key in seen:
            clean_stats["drop_exact_duplicate"] += 1
            continue
        seen.add(key)
        if before != after and len(examples) < 30:
            examples.append((before, after))
        cleaned.append(after)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if not args.skip_register:
        update_dataset_info(args.dataset_info, args.dataset_name, args.output)

    after_dirty = dirty_counts(cleaned)
    validation = validate(cleaned)
    write_report(
        args.report,
        args.input,
        args.output,
        len(data),
        len(cleaned),
        clean_stats,
        before_dirty,
        after_dirty,
        validation,
        examples,
    )

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "dataset_name": args.dataset_name,
                "before_count": len(data),
                "after_count": len(cleaned),
                "clean_stats": dict(clean_stats),
                "before_dirty": dict(before_dirty),
                "after_dirty": dict(after_dirty),
                "validation": dict(validation),
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
