# CloudStudio T4 SFT 操作手册

本文档用于在 CloudStudio 的 T4 16G 显存环境中跑项目一的电商客服 SFT。

## 推荐路线

T4 上优先跑：

```text
Qwen/Qwen3-4B-Instruct-2507
+ 4bit QLoRA
+ ecommerce_customer_sft_5000
+ qwen3_nothink
```

原因：

1. 比原来的 Qwen3-1.7B 更大，项目观感更强。
2. T4 16G 显存能承受 4B 的 4bit QLoRA。
3. 当前使用 5000 条 demo5 数据，先用 4B 做主线比硬冲 8B 更稳。

## 需要上传或同步的文件

最少需要这些：

```text
LLaMA-Factory/
scripts/
项目一/
```

关键文件：

```text
LLaMA-Factory/data/ecommerce_customer_sft_5000.json
LLaMA-Factory/data/dataset_info.json
LLaMA-Factory/examples/train_qlora/qwen3_4b_ecommerce_sft_t4_qlora.yaml
```

如果只跑 SFT，不需要上传之前的 SFT adapter。

## CloudStudio 终端操作

进入 CloudStudio 的 T4 环境后，打开终端。

先确认 GPU：

```bash
nvidia-smi
```

进入项目：

```bash
cd LLaMA-Factory
```

安装环境：

```bash
pip install -e ".[torch,metrics,bitsandbytes]"
```

如果上面失败，使用更保守的安装：

```bash
pip install -e .
pip install bitsandbytes jieba rouge-chinese nltk
```

登录或设置模型源。如果 Hugging Face 下载慢，可以优先使用 ModelScope：

```bash
export USE_MODELSCOPE_HUB=1
```

开始训练：

```bash
llamafactory-cli train examples/train_qlora/qwen3_4b_ecommerce_sft_t4_qlora.yaml
```

## 如果显存爆了

按这个顺序改配置：

1. `cutoff_len: 512` 改成 `384`
2. `lora_target: q_proj,v_proj` 保持不动，不要改成 `all`
3. `gradient_accumulation_steps: 16` 可以改成 `32`
4. `num_train_epochs: 2.0` 改成 `1.0` 先跑通

## 如果训练太慢

T4 训练 4B QLoRA 慢是正常的。先跑一个小样本测试：

```yaml
max_samples: 100
num_train_epochs: 1.0
```

确认能跑后，再改回正式设置。

## 训练完成后看什么

重点看：

```text
saves/Qwen3-4B-Instruct-2507/lora/ecommerce_sft_5000_t4_qlora
```

里面应有：

```text
adapter_config.json
adapter_model.safetensors
trainer_log.jsonl
training_loss.png
all_results.json
eval_results.json
train_results.json
```

SFT 跑完后再做 Chat 测试，不要只看 loss。
