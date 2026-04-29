# Text-to-SQL 执行路线

## 阶段 0：方向封存与切换

项目一电商客服方向暂时封存，不再继续消耗训练资源。

封存结论：

```text
公开客服数据质量不足。
更大模型不能从根本上解决数据噪声和主观评测问题。
项目一保留为探索记录和反思材料。
```

项目二新方向：

```text
Text-to-SQL 高效微调与执行评测
```

## 阶段 1：数据准备

首选数据：

```text
text-to-sql-mix-v2
```

项目中已新增准备脚本：

```text
scripts/prepare_text2sql_mix_sft.py
```

该脚本会从 Hugging Face 下载 `DanielRegaladoCardoso/text-to-sql-mix-v2`，过滤出 SELECT/WITH 类 SQL，并转成 LLaMA-Factory 的 Alpaca 格式。

正式第一版命令：

```bash
python scripts/prepare_text2sql_mix_sft.py --split train --eval-split validation --max-train 10000 --max-eval 200
```

输出：

```text
LLaMA-Factory/data/text2sql_mix_sft.json
项目二/评测集/text2sql_mix_eval.jsonl
项目二/数据处理/text2sql_mix_sft_report.md
```

LLaMA-Factory 数据集注册名：

```text
text2sql_mix_sft
```

抽样策略：

```text
训练集：从原始 train split 随机抽样。
评测集：从原始 validation split 随机抽样。
最终测试：保留原始 test split，暂时不参与训练和调参。
```

目标字段：

```text
question / natural language query
schema / database context
sql / target query
difficulty / source
```

转换成 LLaMA-Factory 格式：

```json
{
  "instruction": "根据数据库 schema 和用户问题生成 SQL。只输出 SQL，不要解释。",
  "input": "Schema:\n...\n\nQuestion:\n...",
  "output": "SELECT ...",
  "system": "你是一个 Text-to-SQL 助手，需要生成可执行 SQL。",
  "history": []
}
```

早期不要一次吃全量数据，先做：

```text
100 条 smoke test
1000 条小训练
10000 条正式第一版
```

## 阶段 2：SFT 训练

主模型：

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

项目中已新增训练配置：

```text
LLaMA-Factory/examples/train_qlora/qwen25_coder_7b_text2sql_t4_qlora.yaml
```

命令：

```bash
cd LLaMA-Factory
llamafactory-cli train examples/train_qlora/qwen25_coder_7b_text2sql_t4_qlora.yaml
```

训练配置：

```text
stage: sft
finetuning_type: lora
quantization_bit: 4
quantization_method: bnb
template: qwen
cutoff_len: 1024
batch_size: 1
gradient_accumulation: 16
learning_rate: 1e-4
epoch: 1
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: q_proj,v_proj
fp16: true
bf16: false
```

注意：

1. T4 不要开 bf16。
2. 7B 在 T4 上必须优先使用 4bit QLoRA。
3. 如果 schema 太长导致爆显存，先降低 `cutoff_len`，不要马上换模型。
4. 第一轮只追求跑通，不追求指标极限。

## 阶段 3：评测设计

评测分四层：

```text
1. 格式合法率：是否只输出 SQL。
2. SQL 解析率：是否能被 SQL parser 解析。
3. Schema 命中：表名、字段名是否来自给定 schema。
4. 执行正确率：执行结果是否和参考 SQL 一致。
```

早期可以先做轻量评测：

```text
Exact match
SQL 关键词覆盖
表名字段名合法性
人工抽样 A/B
```

后续再做真正执行评测。

## 阶段 4：Badcase 分类

badcase 记录文件建议：

```text
项目二/评测结果/badcase_log.md
```

分类：

```text
表名错误
字段名错误
JOIN 关系错误
WHERE 条件遗漏
聚合函数错误
GROUP BY 错误
ORDER BY / LIMIT 错误
输出解释而不是 SQL
SQL 不可解析
```

每条 badcase 记录：

```text
id
question
schema 摘要
reference_sql
model_sql
错误类型
原因分析
修复方向
```

## 阶段 5：DPO 扩展

SFT 跑通后再做 DPO。

DPO 数据构造：

```text
chosen: 正确 SQL / 可执行 SQL
rejected: 常见错误 SQL
```

rejected 来源：

```text
SFT 模型生成的错误 SQL
规则扰动生成的错误 SQL
表名替换
字段替换
条件删除
JOIN 删除
聚合函数替换
```

DPO 目标：

```text
让模型偏好可执行、字段正确、逻辑完整的 SQL。
```

## 阶段 6：简历包装

项目名称：

```text
基于 Qwen2.5-Coder-7B 的 Text-to-SQL 高效微调与执行评测
```

简历写法初稿：

> 基于公开 Text-to-SQL 数据构建自然语言到 SQL 的指令微调数据，使用 Qwen2.5-Coder-7B-Instruct 进行 4bit QLoRA 高效微调。设计 SQL 合法率、schema 命中率、执行正确率和 badcase 分类评估体系，分析模型在表名字段选择、JOIN 关系、条件过滤和聚合推理上的错误模式，并为后续 DPO 偏好对齐构造 chosen/rejected 数据。
