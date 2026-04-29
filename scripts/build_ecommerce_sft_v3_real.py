#!/usr/bin/env python
"""Build a real-data-first ecommerce SFT dataset.

v2 proved that large synthetic SOP expansion can create a new template style.
This v3 dataset therefore avoids bulk generated answers and uses:
1. cleaned high-signal samples from ecommerce_customer_sft v1;
2. mined JDDC dev question/answer pairs;
3. a small curated SOP supplement only.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from build_ecommerce_sft_v2 import (
    DEFAULT_DATASET_INFO,
    PROJECT_ROOT,
    SYSTEM_PROMPT,
    build_curated_samples,
    cap_repeated_outputs,
    drop_reason,
    has_repeated_sentence,
    normalize_space,
    update_dataset_info,
)


DEFAULT_SOURCE = PROJECT_ROOT / "LLaMA-Factory" / "data" / "ecommerce_customer_sft.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "LLaMA-Factory" / "data" / "ecommerce_customer_sft_v3_real.json"
DEFAULT_DEV_QUESTION = (
    PROJECT_ROOT
    / "jddc2019-3th-retrieve-model-master"
    / "jddc2019-3th-retrieve-model-master"
    / "data"
    / "dev_question.txt"
)
DEFAULT_DEV_ANSWER = (
    PROJECT_ROOT
    / "jddc2019-3th-retrieve-model-master"
    / "jddc2019-3th-retrieve-model-master"
    / "data"
    / "dev_answer.txt"
)
DEFAULT_REPORT = PROJECT_ROOT / "项目一" / "数据处理" / "ecommerce_customer_sft_v3_real_report.md"

BAD_ANSWER_PATTERNS = [
    r"NULL$",
    r"系统提示",
    r"还有.*帮",
    r"有什么.*帮助",
    r"稍等",
    r"正在.*查",
    r"感谢.*支持",
    r"生活愉快",
    r"评价",
    r"再见",
    r"^好的[呢哈]?$",
    r"^是的[呢哦]?$",
    r"^您好[，。]?$",
]


def clean_turn(text: str) -> str:
    text = text.replace("!@@@!", "\n")
    text = text.replace("***", "[信息x]")
    text = re.sub(r"https?://\S+", "[链接x]", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def choose_answer(answer_line: str) -> str | None:
    candidates = [clean_turn(item) for item in answer_line.split("<sep>")]
    candidates = [item for item in candidates if item and item != "NULL"]

    good: list[str] = []
    for candidate in candidates:
        if len(candidate) < 12:
            continue
        if any(re.search(pattern, candidate) for pattern in BAD_ANSWER_PATTERNS):
            continue
        if has_repeated_sentence(candidate):
            continue
        good.append(candidate)

    if not good:
        return None

    # Prefer answers that contain actionable details but avoid very long noisy replies.
    good.sort(key=lambda item: (("申请" in item) + ("提供" in item) + ("页面" in item) + ("订单" in item), -abs(len(item) - 70)), reverse=True)
    return good[0]


def parse_context_history(body: str, max_history: int = 4) -> list[list[str]]:
    context_match = re.search(r"<context>(.*?)</context>", body, re.S)
    if not context_match:
        return []

    context = context_match.group(1)
    pairs = re.findall(r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)", context, re.S)
    history: list[list[str]] = []
    for question, answer in pairs:
        clean_question = clean_turn(question)
        answer_choice = choose_answer(answer) or clean_turn(answer.split("<sep>")[0])
        if clean_question and answer_choice and len(answer_choice) >= 8:
            history.append([clean_question, answer_choice])

    return history[-max_history:]


def mine_dev_samples(question_path: Path, answer_path: Path) -> list[dict[str, Any]]:
    questions_text = question_path.read_text(encoding="utf-8", errors="replace")
    answers_text = answer_path.read_text(encoding="utf-8", errors="replace")
    question_sessions = re.findall(r"<session ([^>]+)>\s*(.*?)\s*</session \1>", questions_text, re.S)
    answer_sessions = dict(re.findall(r"<session ([^>]+)>\s*(.*?)\s*</session \1>", answers_text, re.S))

    samples: list[dict[str, Any]] = []
    for session_id, body in question_sessions:
        questions = [clean_turn(item) for item in re.findall(r"<Q\d+>(.*?)</Q\d+>", body, re.S)]
        answer_lines = [line.strip() for line in answer_sessions.get(session_id, "").splitlines() if line.strip()]
        history = parse_context_history(body)

        rolling_history = list(history)
        for index, (question, answer_line) in enumerate(zip(questions, answer_lines), 1):
            answer = choose_answer(answer_line)
            if not question or not answer:
                continue
            if drop_reason({"instruction": question, "input": "", "output": answer, "history": rolling_history}) is not None:
                continue

            samples.append(
                {
                    "instruction": question,
                    "input": "",
                    "output": answer,
                    "system": SYSTEM_PROMPT,
                    "history": rolling_history[-4:],
                    "metadata": {
                        "source": "jddc_dev_mined",
                        "session_id": session_id,
                        "turn": index,
                    },
                }
            )
            rolling_history.append([question, answer])

    return samples


def write_report(
    report_path: Path,
    source_count: int,
    filtered_count: int,
    dev_count: int,
    curated_count: int,
    final_samples: list[dict[str, Any]],
    drop_counts: Counter[str],
    cap_dropped: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = [normalize_space(sample["output"]) for sample in final_samples]
    sources = Counter(sample.get("metadata", {}).get("source", "unknown") for sample in final_samples)
    exact_dup_rate = 1 - len(set(outputs)) / len(outputs)

    lines = [
        "# ecommerce_customer_sft_v3_real 数据处理报告",
        "",
        "## 处理目标",
        "",
        "v3_real 取消大批量 generated SOP 扩展，改为真实数据优先，避免回答风格被合成模板带偏。",
        "",
        "## 样本规模",
        "",
        f"- 原始 v1 样本数：{source_count}",
        f"- v1 过滤保留：{filtered_count}",
        f"- JDDC dev 挖掘：{dev_count}",
        f"- 少量 curated SOP：{curated_count}",
        f"- 最终样本数：{len(final_samples)}",
        f"- 精确重复 output 比例：{exact_dup_rate:.2%}",
        "",
        "## 来源分布",
        "",
        "| 来源 | 数量 |",
        "| --- | ---: |",
    ]
    for source, count in sources.most_common():
        lines.append(f"| {source} | {count} |")

    lines.extend(["", "## v1 删除原因统计", "", "| 原因 | 数量 |", "| --- | ---: |"])
    for reason, count in drop_counts.most_common():
        lines.append(f"| {reason} | {count} |")

    lines.append(f"| exact_output_cap | {cap_dropped} |")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "该版本样本数少于 v2，但不会把批量合成 SOP 当作主体。它更适合用来验证复读问题是否来自模板化合成数据。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dev-question", type=Path, default=DEFAULT_DEV_QUESTION)
    parser.add_argument("--dev-answer", type=Path, default=DEFAULT_DEV_ANSWER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--dataset-name", default="ecommerce_customer_sft_v3_real")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-per-exact-output", type=int, default=3)
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()

    source_samples = json.loads(args.source.read_text(encoding="utf-8"))
    drop_counts: Counter[str] = Counter()
    filtered: list[dict[str, Any]] = []
    for sample in source_samples:
        reason = drop_reason(sample)
        if reason:
            drop_counts[reason] += 1
            continue
        cleaned = dict(sample)
        cleaned["system"] = SYSTEM_PROMPT
        filtered.append(cleaned)

    filtered, cap_dropped = cap_repeated_outputs(filtered, args.max_per_exact_output)
    dev_samples = mine_dev_samples(args.dev_question, args.dev_answer)
    curated = build_curated_samples()
    # Drop the generated bulk from build_curated_samples; keep only manually listed samples.
    curated = [sample for sample in curated if sample.get("metadata", {}).get("source") == "curated_sop_v2"]

    final_samples = filtered + dev_samples + curated
    final_samples.sort(key=lambda item: (item.get("metadata", {}).get("source", ""), item.get("instruction", "")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final_samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.skip_register:
        update_dataset_info(args.dataset_info, args.dataset_name, args.output)

    write_report(args.report, len(source_samples), len(filtered), len(dev_samples), len(curated), final_samples, drop_counts, cap_dropped)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "dataset_name": args.dataset_name,
                "source_samples": len(source_samples),
                "filtered_from_v1": len(filtered),
                "dev_mined": len(dev_samples),
                "curated": len(curated),
                "final_samples": len(final_samples),
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
