# DPO 阶段操作记录

更新时间：2026-04-27

## 当前状态

项目主线暂时不做更多消融实验，优先跑通完整链路：

```text
Qwen3-1.7B post-trained checkpoint
-> 电商客服 SFT
-> DPO 偏好对齐
-> Chat 测试与 badcase 记录
```

当前已完成：

- SFT 数据：`ecommerce_customer_sft_v7_quality`
- SFT 数据条数：741
- SFT 模型：`Qwen/Qwen3-1.7B`
- SFT template：`qwen3_nothink`
- SFT adapter：`LLaMA-Factory/saves/Qwen3-1.7B-Thinking/lora/train_04-27-21-Qwen3-1.7B-sft-r4-all`
- SFT checkpoint：`LLaMA-Factory/saves/Qwen3-1.7B-Thinking/lora/train_04-27-21-Qwen3-1.7B-sft-r4-all/checkpoint-88`
- SFT 训练轮数：1 epoch
- SFT 训练步数：88
- SFT train_loss：3.2081
- SFT eval_loss：3.0480

说明：当前主线重点不是追求最低 loss，而是解决客服复读、模板化和答非所问问题。

## DPO 数据

已新增 DPO 偏好数据：

- 数据集名：`ecommerce_customer_dpo_v1`
- 文件：`LLaMA-Factory/data/ecommerce_customer_dpo_v1.json`
- 构建脚本：`scripts/build_ecommerce_dpo_v1.py`
- 构建报告：`项目一/数据处理/ecommerce_customer_dpo_v1_report.md`
- 数据条数：300

类别分布：

| 类别 | 条数 |
| --- | ---: |
| 物流配送 | 121 |
| 售后退换 | 76 |
| 发票 | 48 |
| 价保优惠 | 40 |
| 人工投诉 | 8 |
| 安装预约 | 7 |

DPO 构造原则：

- `chosen`：来自 v7_quality 中较干净、可执行、能回答客户问题的客服回答。
- `rejected`：构造为模板复读、空泛安抚、只让等待、答非所问的低质量回答。
- 对齐目标：偏好“先解决问题、给出条件/路径/材料/时效”的回答，压低“亲亲/还有其他/耐心等待/页面为准”的模板化倾向。

## WebUI DPO 配置

### Model 页

```text
Language: en
Model name: Qwen3-1.7B-Thinking
Model path: Qwen/Qwen3-1.7B
Finetuning method: lora
Checkpoint path:
  LLaMA-Factory/saves/Qwen3-1.7B-Thinking/lora/train_04-27-21-Qwen3-1.7B-sft-r4-all
Quantization bit: none
Chat template: qwen3_nothink
RoPE scaling: none
Booster: auto
```

注意：这里的 `Checkpoint path` 必须填 SFT adapter，不要填 Base，不要填旧的 `Qwen3-1.7B-Base` 训练产物。

### Train 页

```text
Stage: DPO
Dataset: ecommerce_customer_dpo_v1
Learning rate: 5e-6
Epoch: 1
Max samples: 300
Cutoff length: 512
Batch size: 1
Gradient accumulation: 8
Val size: 0.1
LR scheduler: cosine
Warmup ratio: 0.1
Logging steps: 5
Save steps: 50
LoRA rank: 4
LoRA alpha: 16
LoRA dropout: 0.05
LoRA target: all
Create new adapter: false
pref_loss: sigmoid
pref_beta: 0.1
```

输出目录建议：

```text
LLaMA-Factory/saves/Qwen3-1.7B-Thinking/lora/dpo_v1_from_sft_v7_r4
```

### DPO 参数解释

```text
pref_loss = sigmoid
```

表示使用标准 DPO loss，让模型提高 chosen 相对 rejected 的概率。

```text
pref_beta = 0.1
```

表示偏好优化强度。当前数据只有 300 条，建议先用 0.1，避免把模型过度拉偏。

## 训练后检查

DPO 跑完后先不看 loss 自嗨，重点做 Chat 验证：

1. 是否还会一直吐 token 到 max。
2. 是否还重复“您好/亲亲/还有其他可以帮您吗”。
3. 是否能先回答客户问题，而不是先堆套话。
4. 是否能稳定给出操作路径、材料、条件和时效。
5. 是否过度变冷淡，完全没有客服语气。

建议推理参数：

```text
temperature: 0.7
top_p: 0.8
repetition_penalty: 1.12
presence_penalty: 1.5
max_new_tokens: 128 或 192
```

## 面试表述

可以这样讲：

> SFT 后模型已经具备基础电商客服问答能力，但在小数据场景下仍容易学到模板化客服话术。为了解决“空泛安抚、重复收尾、只让等待”的问题，我构建了 300 条电商客服偏好数据，chosen 使用可执行的业务回答，rejected 使用模板化或无效回答，并在 SFT adapter 基础上继续进行 DPO。DPO 的目标不是注入新知识，而是把输出风格从“像客服”进一步对齐到“能解决客户问题”。
