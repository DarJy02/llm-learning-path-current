# text2sql_mix_sft 数据处理报告

## 数据来源

- Hugging Face dataset: DanielRegaladoCardoso/text-to-sql-mix-v2
- split: train
- eval_split: validation
- train output: LLaMA-Factory/data/text2sql_mix_sft_local_1000.json
- eval output: 项目二/评测集/text2sql_mix_eval_local_100.jsonl

## 处理参数

- max_train: 1000
- max_eval: 100
- seed: 42
- select_only: True
- dialects: generic
- difficulties: easy,medium

## 输出规模

- train records: 1000
- eval records: 100
- total records: 1100

## dialect 分布

- generic: 1100

## difficulty 分布

- easy: 970
- medium: 130

## source Top 20

- clinton-text2sql: 292
- kaxap-llama2: 180
- nstext2sql-wikisql: 176
- sql-create-context: 165
- nstext2sql-sql_create_context: 153
- nstext2sql-nvbench: 41
- nstext2sql-mimicsql_data: 35
- nstext2sql-squall: 13
- nstext2sql-spider: 9
- nstext2sql-css: 9
- nstext2sql-atis: 7
- nstext2sql-sede: 5
- nstext2sql-criteria2sql: 5
- nstext2sql-eicu: 4
- nstext2sql-advising: 2
- nstext2sql-pesticide: 1
- nstext2sql-scholar: 1
- nstext2sql-restaurants: 1
- nstext2sql-mimic_iii: 1

## LLaMA-Factory 注册名

```text
text2sql_mix_sft
```

## 样本格式

```json
{
  "instruction": "根据数据库 schema 和用户问题生成 SQL。只输出 SQL，不要解释。",
  "input": "SQL Dialect: generic\nDifficulty: easy\n\nSchema:\nCREATE TABLE table_204_579 (\n    id number,\n    \"code\" text,\n    \"district\" text,\n    \"headquarters\" text,\n    \"population (as of 2011)\" number,\n    \"area (km2)\" number,\n    \"density (/km2)\" number\n)\n\nQuestion:\ntell me the number of districts with an area over 5000 .",
  "output": "SELECT COUNT(\"district\") FROM table_204_579 WHERE \"area (km2)\" > 5000",
  "system": "你是一个 Text-to-SQL 助手。请根据数据库 schema 和用户问题生成可执行 SQL。",
  "history": [],
  "metadata": {
    "id": "clinton-91f0b2a85c324edb",
    "source": "clinton-text2sql",
    "dialect": "generic",
    "difficulty": "easy"
  }
}
```
