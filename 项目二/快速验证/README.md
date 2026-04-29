# 项目二快速验证：Text-to-SQL 本地 SFT Smoke Test

目标：在本地 RTX 4070 Laptop 8GB 显存上，用很小模型和 1000 条数据快速验证流程是否合理。

这不是最终实验配置。正式训练仍然使用：

```text
Qwen/Qwen2.5-Coder-7B-Instruct + 4bit QLoRA
```

本地快速验证使用：

```text
Qwen/Qwen2.5-Coder-0.5B-Instruct + LoRA SFT
```

## generic + easy/medium 是什么意思

`text-to-sql-mix-v2` 里有不同 SQL 方言和难度标签。

```text
generic: 通用 SQL 方言，先避开 MySQL/PostgreSQL/SQLite 等方言差异。
easy/medium: 简单和中等难度，先避开特别复杂的嵌套查询和多表推理。
```

第一轮快速验证的目标不是挑战最难数据，而是确认：

```text
数据能下载
数据能转换
LLaMA-Factory 能识别数据集
模型能开始训练
输出目录能正常落盘
```

## 1. 依赖说明

你本地如果已经能跑 LLaMA-Factory 训练，不需要重新 `pip install -e .`。

本地数据转换只需要：

```bash
pip install pandas pyarrow
```

如果 Python 的 `datasets` 依赖冲突，先执行：

```bash
pip install -U datasets huggingface_hub pyarrow s3fs botocore
```

或者临时：

```bash
pip install "urllib3<2"
```

## 2. 生成 1000 条本地快速验证数据

在项目根目录执行：

```bash
python scripts/prepare_text2sql_mix_sft.py \
  --local-train-glob "项目二/快速验证/train-*.parquet" \
  --local-eval-glob "项目二/快速验证/validation-*.parquet" \
  --max-train 1000 \
  --max-eval 100 \
  --output-name text2sql_mix_sft_local_1000.json \
  --eval-name text2sql_mix_eval_local_100.jsonl \
  --report-name text2sql_mix_sft_local_1000_report.md
```

会生成：

```text
LLaMA-Factory/data/text2sql_mix_sft_local_1000.json
项目二/评测集/text2sql_mix_eval_local_100.jsonl
项目二/数据处理/text2sql_mix_sft_local_1000_report.md
```

LLaMA-Factory 数据集注册名：

```text
text2sql_mix_sft_local_1000
```

## 3. 本地快速训练

```bash
cd LLaMA-Factory
llamafactory-cli train ../项目二/快速验证/qwen25_coder_0_5b_text2sql_local_sft.yaml
```

输出目录：

```text
LLaMA-Factory/saves/Qwen2.5-Coder-0.5B-Instruct/lora/text2sql_local_1000_sft
```

## 4. 如果本地显存爆了

按顺序改：

```text
cutoff_len: 512 -> 384
gradient_accumulation_steps: 8 -> 16
max_samples: 1000 -> 200
```

## 5. 验证成功的标准

看到这些文件就算流程通了：

```text
adapter_config.json
adapter_model.safetensors
trainer_log.jsonl
train_results.json
eval_results.json
```
