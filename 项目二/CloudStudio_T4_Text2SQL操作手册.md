# CloudStudio T4 Text-to-SQL 操作手册

## 推荐环境

```text
GPU: T4 16G
model: Qwen/Qwen2.5-Coder-7B-Instruct
method: 4bit QLoRA
task: Text-to-SQL SFT
```

## 为什么用 QLoRA

T4 只有 16G 显存，7B 模型如果用普通 fp16 LoRA 很容易爆显存。

QLoRA 的作用：

```text
模型主体 4bit 加载，节省显存。
只训练 LoRA 小参数。
让 T4 能训练 7B 级模型。
```

## CloudStudio 基础操作

进入 T4 环境后：

```bash
nvidia-smi
```

进入项目：

```bash
cd LLaMA-Factory
```

安装依赖：

```bash
pip install -e .
pip install -r requirements/metrics.txt
pip install bitsandbytes modelscope datasets pyarrow pandas
```

如果下载慢：

```bash
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements/metrics.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install bitsandbytes modelscope datasets pyarrow pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
```

使用 ModelScope：

```bash
export USE_MODELSCOPE_HUB=1
```

## 数据下载与转换

回到项目根目录，也就是能看到 `scripts/` 和 `LLaMA-Factory/` 的目录：

```bash
cd ..
```

先跑一个 100 条 smoke test：

```bash
python scripts/prepare_text2sql_mix_sft.py \
  --split train \
  --eval-split validation \
  --max-train 100 \
  --max-eval 20 \
  --output-name text2sql_mix_sft.json \
  --eval-name text2sql_mix_eval.jsonl \
  --report-name text2sql_mix_sft_report.md
```

确认能生成文件后，再跑正式第一版：

```bash
python scripts/prepare_text2sql_mix_sft.py \
  --split train \
  --eval-split validation \
  --max-train 10000 \
  --max-eval 200 \
  --output-name text2sql_mix_sft.json \
  --eval-name text2sql_mix_eval.jsonl \
  --report-name text2sql_mix_sft_report.md
```

会生成：

```text
LLaMA-Factory/data/text2sql_mix_sft.json
项目二/评测集/text2sql_mix_eval.jsonl
项目二/数据处理/text2sql_mix_sft_report.md
```

当前 LLaMA-Factory 注册名：

```text
text2sql_mix_sft
```

说明：

```text
--max-train 10000 不是说数据集只有 10000 条。
脚本会从 Hugging Face 的 train split 中随机抽 10000 条作为本地训练集。
脚本会从 validation split 中随机抽 200 条作为固定评测集。
test split 暂时不动，留作最终测试。
```

如果下载数据时报 `urllib3` / `botocore` / `s3fs` 相关错误，优先执行：

```bash
pip install -U datasets huggingface_hub pyarrow s3fs botocore
```

或者临时降级：

```bash
pip install "urllib3<2"
```

## 命令行训练

进入 LLaMA-Factory：

```bash
cd LLaMA-Factory
```

开始训练：

```bash
llamafactory-cli train examples/train_qlora/qwen25_coder_7b_text2sql_t4_qlora.yaml
```

## WebUI 推荐填写

Model 页：

```text
Model name: Qwen2.5-Coder-7B-Instruct
Model path: Qwen/Qwen2.5-Coder-7B-Instruct
Finetuning method: lora
Checkpoint path: 空
Quantization bit: 4
Quantization method: bnb
Chat template: qwen
Flash attention: disabled
Trust remote code: 勾选
```

Train 页：

```text
Stage: SFT
Dataset: text2sql_mix_sft
Learning rate: 1e-4
Epoch: 1
Cutoff length: 1024
Max samples: 10000
Batch size: 1
Gradient accumulation: 16
Val size: 0.05
LR scheduler: cosine
Warmup steps: 30
FP16: 勾选
BF16: 不勾
Gradient checkpointing: 勾选
Optimizer: paged_adamw_8bit
```

LoRA 页：

```text
LoRA rank: 8
LoRA alpha: 16
LoRA dropout: 0.05
LoRA modules: q_proj,v_proj
Create new adapter: 勾选
Use rslora: 不勾
Use DoRA: 不勾
Use PiSSA: 不勾
```

输出目录：

```text
saves/Qwen2.5-Coder-7B-Instruct/lora/text2sql_sft_t4_qlora
```

## 显存爆了怎么办

按顺序调整：

```text
cutoff_len: 1024 -> 768
gradient_accumulation: 16 -> 32
max_samples: 10000 -> 1000 先跑通
epoch: 1 保持不变
```

不要一上来改成 `lora_target=all`。

## 当前缺口

已完成：

```text
数据下载脚本
数据转换脚本
dataset_info.json 注册
训练 yaml
```

还需要补：

```text
评测脚本
badcase 模板
```
