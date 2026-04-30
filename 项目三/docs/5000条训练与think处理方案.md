# 5000 条训练与 think 处理方案

## 结论

你想本地先训 5000 条是可以的，而且比 500/1000 更像一个正式项目。

但关键不是“随机拿 5000 条”，而是：

> 从当前 parquet 里筛出 5000 条长度合适、答案完整、带正确 reasoning 的样本。

否则很多样本会被 `cutoff_len=4096` 截断，尤其是 `<think>...</think>` 很长时，可能把最终答案截没了。

## 当前分片长度情况

当前文件：

```text
项目三/data/openr1-math-220k/default/default-00000-of-00010.parquet
```

统计结果：

| 条件 | 可用题数 |
|---|---:|
| 有至少一条正确 generation | 6,534 |
| 最短正确 generation <= 8,000 字符 | 2,030 |
| 最短正确 generation <= 10,000 字符 | 2,772 |
| 最短正确 generation <= 12,000 字符 | 3,387 |
| 最短正确 generation <= 16,000 字符 | 4,287 |
| 最短正确 generation <= 20,000 字符 | 4,959 |
| 最短正确 generation <= 24,000 字符 | 5,408 |

所以：

- 只用很短样本，凑不到 5000。
- 要凑 5000 条，就必须接受一部分较长 reasoning。
- 如果 `cutoff_len=4096`，一部分样本大概率会被截断。
- 如果 `cutoff_len=8192`，更适合 5000 条训练，但显存压力更大。

## cutoff_len 怎么处理

### 不推荐：直接随机 5000 + cutoff_len 4096

原因：

- `<think>` 可能很长。
- 最终答案通常在输出末尾。
- 如果被截断，模型可能看不到 `\boxed{}` 或最终答案。
- 训练会变成“学习一段未完成推理”，质量会下降。

### 推荐方案 A：4096 + 长度过滤

适合：

- 本地显存较小。
- 先保证训练质量。

做法：

- 用 tokenizer 计算完整样本 token 长度。
- 只保留 `prompt + response <= 4096` 的样本。
- 如果不足 5000，就接受 3000-4000 条，不硬凑。

优点：

- 不容易训到被截断的坏样本。
- 本地更稳。

缺点：

- 可能不够 5000。
- 更偏短推理样本。

### 推荐方案 B：8192 + 5000 条

适合：

- 本地显存 24GB 左右，或愿意慢慢跑。
- 想保留更多 R1 风格 reasoning。

配置建议：

```yaml
cutoff_len: 8192
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
finetuning_type: lora
quantization_bit: 4
```

优点：

- 更适合这批长 reasoning 数据。
- 更容易保留 `<think>` 和最终答案。
- 更接近数学推理训练。

缺点：

- 显存和训练时间明显增加。
- 如果环境不支持 4bit/flash attention，可能比较慢。

### 推荐方案 C：分阶段训练

这是我最推荐的本地方案：

```text
第一轮：3000 条，cutoff_len=4096，筛短样本
第二轮：5000 条，cutoff_len=8192，保留长 reasoning
```

如果第一轮结果正常，再跑第二轮。

## think 怎么处理

当前数据里的 `generations` 已经包含：

```text
<think>
...
</think>
```

所以第一版建议：**保留 think 标签**。

原因：

- 这是 OpenR1 数据的原始 reasoning 风格。
- 它能让模型学习“先推理，再给最终答案”的格式。
- 你的项目本来就是数学推理复现，保留 `<think>` 更贴近 R1/AIMO 路线。

## 必须用“能 think 的模型”吗？

不必须。

`<think>` 本质上是文本格式，不是某个神秘开关。

模型生成 token 的逻辑仍然是：

```text
根据前文预测下一个 token
```

如果训练数据大量是：

```text
<think>
推理过程
</think>
最终答案是 \boxed{...}
```

模型就会学会生成这种格式。

所以：

| 模型 | 是否能处理 think |
|---|---|
| Qwen2.5-Math | 可以通过 SFT 学会输出 `<think>` 风格 |
| DeepSeek-R1-Distill-Qwen | 已经很熟悉 `<think>` 风格 |
| Qwen3 Thinking | 原生支持 thinking/no-thinking 模式 |

我们现在用：

```text
Qwen/Qwen2.5-Math-1.5B-Instruct
```

它不是原生 Qwen3 thinking 模型，但可以通过 SFT 学会“显式 reasoning 输出”。

## template: qwen 和 think 的关系

`template: qwen` 只是 Qwen2.5 的对话模板。

它负责把样本包装成类似：

```text
user: 题目
assistant: 回答
```

而 `<think>...</think>` 是 assistant 回答里的普通文本。

所以：

```yaml
template: qwen
```

不代表“不能 think”。

它只是说明：

> 用 Qwen2.5 的聊天格式训练，至于 assistant 输出里有没有 `<think>`，取决于你的训练数据。

## 训练数据怎么筛

### SFT 样本

每题选择：

```text
正确 generation 中最短的一条
```

并且要求：

- 有 `<think>`。
- 有最终答案。
- 最好有 `\boxed{}` 或明确 final answer。
- token 长度不超过当前 `cutoff_len`。

### DPO 样本

优先构造：

```text
chosen = 正确且较短的 generation
rejected = 错误 generation
```

如果同题没有错误 generation，但有多条正确 generation，则构造：

```text
chosen = 正确且更短的 generation
rejected = 正确但过长的 generation
```

这样 DPO 的目标就是：

> 在保持正确性的同时减少过长输出。

这和 AIMO2 第二名方案的思路一致。

## 当前建议训练设置

如果你坚持 5000 条，本地建议：

```text
SFT: 5000
DPO: 3000-5000 pair
cutoff_len: 8192 优先
SFT epoch: 1
DPO epoch: 1
```

如果显存不够：

```text
SFT: 3000-4000
DPO: 2000-3000 pair
cutoff_len: 4096
SFT epoch: 1
DPO epoch: 1
```

不要用：

```text
5000 随机样本 + cutoff_len 4096 + 任由截断
```

## 对提升的预期

5000 条训练更可能看到：

- `<think>` 格式更稳定。
- `\boxed{}` 输出更稳定。
- 输出过程更像 R1。
- pass@1 可能小幅提升。
- DPO 后输出长度可能下降。

但不要预期：

```text
Qwen2.5-Math-1.5B 经过 5000 条 SFT 后数学能力暴涨
```

更真实的项目结论应该是：

```text
SFT 让模型学会 R1 风格推理输出；
DPO 在保持正确率的前提下改善输出长度和可解析率。
```
