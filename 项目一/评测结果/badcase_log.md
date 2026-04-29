# Badcase 与推理问题记录

本文档记录项目一在训练、推理、评测阶段发现的问题，用于后续 badcase 分析、实验复盘和 README 汇总。

## 2026-04-26 Chat 推理重复输出

### 问题现象

在 LLaMA-Factory WebUI 的 `Chat` 页进行客服问答推理时，模型会反复输出同一段或相似句式，直到达到 `Maximum new tokens = 256` 才停止。

典型表现：

text

Copy

```text
请您提供订单号，我帮您查询。请您提供订单号，我帮您查询。请您提供订单号，我帮您查询……
```

### 影响

1.  回答冗长，用户体验差。
2.  自动指标和人工评测会被污染。
3.  如果不同 rank 的模型重复程度不同，A/B 对比会失真。
4.  说明当前推理停止条件或解码参数需要统一校准。

### 可能原因

1.  `Qwen2.5-1.5B` 是 Base 模型，不是 Instruct 模型。Base 模型本身没有充分对齐聊天停止行为，容易不知道何时结束。
2.  `Maximum new tokens` 设置为 256 时，如果模型没有主动输出结束符，就会一直生成到上限。
3.  `temperature` 或 `top_p` 偏高时，输出更容易发散或陷入重复。
4.  `repetition_penalty` 未设置或过低时，模型没有足够惩罚重复片段。
5.  训练数据中客服常用句式重复度较高，例如“请您提供订单号”“以页面显示为准”“请您保持手机畅通”，模型容易学成模板循环。
6.  如果 `Chat template`、`Enable thinking` 或 adapter 加载不一致，模型可能无法稳定识别助手回复结束位置。

### 解决方案

推理评测统一使用更保守的生成参数：

text

Copy

```text
Maximum new tokens: 128 或 160Temperature: 0.1 - 0.2Top-p: 0.7 - 0.8Repetition penalty: 1.1 - 1.2Enable thinking: 关闭Skip special tokens: 开启Escape HTML tags: 开启Chat template: qwen
```

客服场景优先推荐：

text

Copy

```text
Maximum new tokens: 160Temperature: 0.2Top-p: 0.8Repetition penalty: 1.15
```

### 评测处理规则

正式评测中，如果回答出现明显重复，记录为 badcase：

text

Copy

```text
badcase_type = 重复输出severity = 中/高
```

如果重复导致回答不可用，人工评分中：

text

Copy

```text
准确性：最多 2 分可执行性：最多 2 分客服语气：最多 2 分幻觉/稳定性：标记为不稳定
```

### 后续观察

需要分别观察 `r4 / r8 / r16 / r32` 是否都出现重复：

text

Copy

```text
如果所有模型都重复：优先调整推理参数和模板。如果只有某个 rank 重复：记录为该 rank 的稳定性问题。如果 Base 重复严重而 SFT 不重复：说明 SFT 改善了客服生成稳定性。
```

### 解决办法

text

Copy

```text
1.数据 v2 里减少纯收尾问句，尤其是独立 output。2.训练样本的 output 尽量是“完整业务答复”，自然结束。3.高频模板限频，同一句最多保留几条。4.可以在回答风格里保留温柔话术，但放在业务解决方案前后，不要让它单独成为答案。5.推理时先用较短 max_new_tokens=128/160 和 repetition_penalty=1.15/1.2，作为保护栏。6.一句话：它确实是“没稳定学会句子/轮次结束”，但不是 template 错而是 Base 模型的续写习惯 + 数据里大量客服收尾模板，让 <|im_end|> 的概率压不过继续复读。
```