import pyarrow.parquet as pq
import json
import os
import random

# Paths - run from scripts/ directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PARQUET_PATH = os.path.join(ROOT_DIR, "项目二", "快速验证", "train-00001-of-00002.parquet")
OUTPUT_PATH = os.path.join(ROOT_DIR, "LLaMA-Factory", "data", "text2sql_mix_sft_local_10000.json")
MAX_SAMPLES = 10000

SYSTEM_PROMPT = "你是一个 Text-to-SQL 助手。请根据数据库 schema 和用户问题生成可执行 SQL。"
INSTRUCTION = "根据数据库 schema 和用户问题生成 SQL。只输出 SQL，不要解释。"


def build_input(dialect, difficulty, schema_context, question):
    return (
        f"SQL Dialect: {dialect}\n"
        f"Difficulty: {difficulty}\n\n"
        f"Schema:\n{schema_context}\n\n"
        f"Question:\n{question}"
    )


def main():
    print(f"Reading parquet: {PARQUET_PATH}")
    table = pq.read_table(PARQUET_PATH)
    total = table.num_rows
    print(f"Total rows: {total}")

    # Work directly with pyarrow columns (no pandas needed)
    cols = {name: table.column(name) for name in table.column_names}

    # Pick 10000 random indices
    indices = sorted(random.sample(range(total), MAX_SAMPLES))
    print(f"Sampled {len(indices)} rows")

    records = []
    for i in indices:
        record = {
            "instruction": INSTRUCTION,
            "input": build_input(
                str(cols["dialect"][i].as_py()),
                str(cols["difficulty"][i].as_py()),
                str(cols["schema_context"][i].as_py()),
                str(cols["instruction"][i].as_py()),
            ),
            "output": str(cols["sql"][i].as_py()),
            "system": SYSTEM_PROMPT,
            "history": [],
            "metadata": {
                "id": str(cols["id"][i].as_py()),
                "source": str(cols["source"][i].as_py()),
                "dialect": str(cols["dialect"][i].as_py()),
                "difficulty": str(cols["difficulty"][i].as_py()),
            },
        }
        records.append(record)

        if (len(records) % 2000) == 0:
            print(f"  processed {len(records)}...")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Written {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
