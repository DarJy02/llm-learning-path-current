import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
    "{problem}"
)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "value", "text", "solution", "answer"):
            if key in value:
                return as_text(value[key])
    if isinstance(value, list):
        return "\n".join(as_text(item) for item in value if as_text(item)).strip()
    return str(value).strip()


def first_existing(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def extract_problem(row: dict[str, Any]) -> str:
    value = first_existing(
        row,
        (
            "problem",
            "question",
            "prompt",
            "input",
            "instruction",
            "query",
        ),
    )
    if value:
        return as_text(value)

    messages = first_existing(row, ("messages", "conversations"))
    if isinstance(messages, list):
        for msg in messages:
            role = msg.get("role") or msg.get("from")
            if role in ("user", "human"):
                return as_text(msg)
    return ""


def extract_answer_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    generations = first_existing(
        row,
        (
            "generations",
            "responses",
            "completions",
            "solutions",
            "reasoning_traces",
        ),
    )
    correctness = first_existing(
        row,
        (
            "correct",
            "is_correct",
            "verified",
            "verification_result",
            "generation_correctness",
            "correctness",
        ),
    )

    if isinstance(generations, list):
        for idx, generation in enumerate(generations):
            text = as_text(generation)
            if not text:
                continue
            correct = None
            if isinstance(correctness, list) and idx < len(correctness):
                correct = bool(correctness[idx])
            elif isinstance(generation, dict):
                flag = first_existing(
                    generation,
                    (
                        "correct",
                        "is_correct",
                        "verified",
                        "verification_result",
                    ),
                )
                if flag is not None:
                    correct = bool(flag)
            candidates.append({"text": text, "correct": correct})

    single_solution = first_existing(
        row,
        ("solution", "answer", "output", "response", "completion"),
    )
    text = as_text(single_solution)
    if text:
        candidates.append({"text": text, "correct": True})

    deduped = []
    seen = set()
    for item in candidates:
        text = item["text"]
        if text and text not in seen:
            deduped.append(item)
            seen.add(text)
    return deduped


def pick_sft_answer(candidates: list[dict[str, Any]]) -> str:
    correct = [item["text"] for item in candidates if item.get("correct") is True]
    pool = correct or [item["text"] for item in candidates]
    if not pool:
        return ""
    return min(pool, key=len)


def pick_dpo_pair(candidates: list[dict[str, Any]]) -> tuple[str, str] | None:
    correct = [item["text"] for item in candidates if item.get("correct") is True]
    incorrect = [item["text"] for item in candidates if item.get("correct") is False]

    if correct and incorrect:
        return min(correct, key=len), max(incorrect, key=len)

    texts = [item["text"] for item in candidates]
    if len(texts) < 2:
        return None

    shortest = min(texts, key=len)
    longest = max(texts, key=len)
    if shortest == longest:
        return None

    # If all traces are verified/correct, use DPO to prefer concise reasoning.
    if len(longest) >= 1.35 * len(shortest):
        return shortest, longest

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--subset", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-sft", type=int, default=1000)
    parser.add_argument("--max-dpo", type=int, default=1000)
    parser.add_argument("--max-scan", type=int, default=20000)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--sft-out", required=True)
    parser.add_argument("--dpo-out", required=True)
    args = parser.parse_args()

    streaming = not args.no_streaming
    ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=streaming)
    print(f"Loaded {args.dataset}/{args.subset}/{args.split} (streaming={streaming})")
    if hasattr(ds, "column_names") and ds.column_names:
        print(f"Columns: {ds.column_names}")
    elif hasattr(ds, "features") and ds.features:
        print(f"Features: {list(ds.features.keys())}")

    sft_rows = []
    dpo_rows = []

    for index, row in enumerate(ds):
        if index >= args.max_scan:
            break

        problem = extract_problem(row)
        if not problem:
            continue

        prompt = PROMPT_TEMPLATE.format(problem=problem)
        candidates = extract_answer_candidates(row)

        if len(sft_rows) < args.max_sft:
            answer = pick_sft_answer(candidates)
            if answer:
                sft_rows.append(
                    {
                        "instruction": prompt,
                        "output": answer,
                    }
                )

        if len(dpo_rows) < args.max_dpo:
            pair = pick_dpo_pair(candidates)
            if pair:
                chosen, rejected = pair
                dpo_rows.append(
                    {
                        "conversations": [
                            {
                                "from": "human",
                                "value": prompt,
                            }
                        ],
                        "chosen": {
                            "from": "gpt",
                            "value": chosen,
                        },
                        "rejected": {
                            "from": "gpt",
                            "value": rejected,
                        },
                    }
                )

        if len(sft_rows) >= args.max_sft and len(dpo_rows) >= args.max_dpo:
            break

    sft_out = Path(args.sft_out)
    dpo_out = Path(args.dpo_out)
    sft_out.parent.mkdir(parents=True, exist_ok=True)
    dpo_out.parent.mkdir(parents=True, exist_ok=True)

    sft_out.write_text(json.dumps(sft_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    dpo_out.write_text(json.dumps(dpo_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote SFT rows: {len(sft_rows)} -> {sft_out}")
    print(f"Wrote DPO rows: {len(dpo_rows)} -> {dpo_out}")

    if not sft_rows:
        print("Warning: no SFT rows generated. Inspect dataset columns above and adjust extract_problem/extract_answer_candidates.")
    if not dpo_rows:
        print("Warning: no DPO rows generated. This can happen if the dataset has only one trace per problem.")


if __name__ == "__main__":
    main()
