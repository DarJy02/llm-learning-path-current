# 项目三：数学推理 SFT + DPO/GRPO/RLVR

## 项目定位

本项目确定为：

> 基于 Qwen 系列小模型的数学推理微调项目：参考 Kaggle AIMO / AIMO2、Open-R1 和 NuminaMath 路线，完成 SFT + DPO/GRPO/RLVR，并分析训练阶段对数学正确率、推理长度和答案格式的影响。

这不是学习 LLaMA-Factory 的项目，而是一个真实数学推理训练项目。LLaMA-Factory、Open-R1、AIMO2 这些只是可复用工具和参考方案。

## 这些项目之间的关系

| 名称 | 它是什么 | 在本项目里的位置 |
|---|---|---|
| Kaggle AIMO / AIMO2 | 数学奥赛推理竞赛 | 目标场景和评测灵感 |
| imagination-research/aimo2 | AIMO2 第二名方案 | 重点参考：SFT + DPO + 推理策略 |
| project-numina/aimo-progress-prize | AIMO 第一阶段冠军方案 | 重点参考：CoT SFT + TIR SFT |
| huggingface/open-r1 | DeepSeek-R1 路线开源复现 | 重点参考：SFT + GRPO/RLVR |
| OpenR1-Math-220k | 数学推理训练数据 | SFT / DPO 数据核心来源 |
| NuminaMath-TIR | 工具辅助推理数据 | 可选：训练代码辅助解题能力 |

结论：这些不是同一个项目，但方向高度一致。我们要做的是把它们拼成自己的项目路线。

## 当前推荐路线

### 第一阶段：SFT

目标：让 Qwen 小模型学会数学推理格式和长 CoT 输出。

推荐数据：

- `open-r1/OpenR1-Math-220k`
- 可选补充：`AI-MO/NuminaMath-CoT`

核心观察：

- 正确率是否提升。
- 输出长度是否明显变长。
- `\boxed{}` 最终答案格式是否更稳定。
- 小模型是否学会分步骤推理。

### 第二阶段：DPO

目标：在 SFT 基础上优化回答质量，特别是减少冗长、错误或不可解析的输出。

推荐做法：

- 从 `OpenR1-Math-220k` 同题多条 reasoning trace 中构造 chosen/rejected。
- chosen：答案正确、格式可解析、长度适中。
- rejected：答案错误、过长、不可解析，或虽然正确但明显冗长。

这个阶段最贴近 AIMO2 方案。

### 第三阶段：GRPO/RLVR

目标：用可验证 reward 进一步优化模型。

可用 reward：

- 最终答案是否正确。
- 是否包含 `\boxed{}`。
- 答案是否能被解析。
- 输出是否超长。

这个阶段最贴近 Open-R1 / DeepSeek-R1 路线。

## 工具选择

### 继续用 LLaMA-Factory 吗？

需要，但不是唯一选择。

建议分工：

| 阶段 | 推荐工具 | 原因 |
|---|---|---|
| SFT | LLaMA-Factory 或 Open-R1 | LLaMA-Factory 更顺手；Open-R1 更贴近 R1 复现 |
| DPO | LLaMA-Factory | 配置简单，AIMO2 方案也用了 LLaMA-Factory 分支 |
| GRPO/RLVR | Open-R1 优先 | Open-R1 原生提供 `grpo.py`，更适合数学可验证 reward |
| 评估/采样 | 自己写脚本 + vLLM/lmdeploy | 方便做多采样、自一致投票、长度统计 |

所以：如果第一版先做 `SFT + DPO`，直接继续用 LLaMA-Factory 就可以；如果要做 `GRPO/RLVR`，建议引入 Open-R1。

### 微调 Qwen 是在这里搞吗？

`项目三` 作为项目总目录，负责放：

- 项目说明。
- 数据处理脚本。
- LLaMA-Factory 配置。
- Open-R1 配置。
- 实验记录。
- 评估结果。

真实训练代码可以有两种方式：

1. 继续复用上一级已有的 `../LLaMA-Factory`。
2. 在 `项目三` 下克隆 `open-r1`，专门用于 GRPO/RLVR。

推荐结构：

```text
项目三/
  README.md
  模型选择.md
  数据与参考项目.md
  实验路线.md
  configs/
    llamafactory/
    open-r1/
  scripts/
  data/
  outputs/
  logs/
```

## 第一版目标

不要一上来复现 Kaggle 冠军。第一版目标定为：

> 在 Qwen 小模型上完成数学推理 SFT + DPO，并用 MATH-500 / AIME 风格题评估正确率、输出长度和答案格式。

第二版再做：

> 引入 Open-R1 的 GRPO/RLVR，用规则 reward 优化最终答案正确率。

## 关键参考链接

- Open-R1: https://github.com/huggingface/open-r1
- AIMO2 第二名方案: https://github.com/imagination-research/aimo2
- AIMO 第一阶段冠军方案: https://github.com/project-numina/aimo-progress-prize
- OpenR1-Math-220k: https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- NuminaMath-TIR: https://huggingface.co/datasets/AI-MO/NuminaMath-TIR
- MATH-500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- Kaggle AIMO: https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize
