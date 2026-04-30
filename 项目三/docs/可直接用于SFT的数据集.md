# 可直接用于 SFT 的数学推理数据集

## 结论

网上确实有已经整理好的 SFT 数学推理数据，不一定非要自己从 `OpenR1-Math-220k` 的多 generation 结构里筛。

但是要分清楚：

- 整理好的 SFT 数据：更方便，适合快速训练。
- OpenR1-Math-220k 原始数据：更灵活，适合自己构造 SFT/DPO。

如果当前目标是先跑出一个像样的 SFT，本地可以优先用整理好的 SFT 数据。

## 推荐 1：Fast-Math-R1-SFT

链接：

- https://huggingface.co/datasets/RabotniKuma/Fast-Math-R1-SFT

规模：

```text
7,900 rows
约 222 MB
```

来源：

- OpenR1 Math
- openr1_hard
- Light-R1-SFTData stage2

数据集作者说明它做过这些处理：

- 从 OpenR1 Math 中采样高难度样本。
- 合并 openr1_hard。
- 合并 Light-R1 stage2 数据。
- 去重。
- 选择正确 generation 中 token 长度最短的那条。
- 形成 problem - R1 trace - answer 格式。

适合用途：

- 直接做 SFT。
- 本地 5000 条训练。
- 复现 AIMO/R1 风格数学推理。

优点：

- 已经帮你做了“多 generation 里筛正确解”的工作。
- 规模刚好适合本地。
- 包含 R1 风格 reasoning trace。

缺点：

- 不一定天然适合 DPO。
- 如果后续要做 DPO，仍然需要 OpenR1-Math-220k 这种多 generation 数据。

## 推荐 2：Light-R1-SFTData

链接：

- https://huggingface.co/datasets/qihoo360/Light-R1-SFTData

适合用途：

- 参考 AIMO2 第二名路线。
- 做 stage1/stage2 SFT。

关键信息：

- AIMO2 第二名方案提到使用了 Light-R1 stage2 data。
- 数据集介绍里也说明 stage2 是从 76k 数据中筛出的更难样本，约 3k。
- 后续还基于 SFT stage2 的响应构造 DPO pair。

优点：

- 和 AIMO2 第二名路线关系更近。
- 数据规模小，适合本地实验。

缺点：

- 数据量较小。
- 需要看具体字段后再接入 LLaMA-Factory。

## 推荐 3：Dataset-SFT-Math

链接：

- https://huggingface.co/datasets/96kevinli29/Dataset-SFT-Math

特点：

- 是一个数学 SFT mixture。
- 来源包含 NuminaMath-CoT、OpenR1 Math 等。

适合用途：

- 扩充 SFT 数据。
- 做不同数据源对比实验。

优点：

- 混合数据源，覆盖面可能更广。

缺点：

- 和 AIMO2 路线的直接关系不如 Fast-Math-R1-SFT / Light-R1-SFTData。

## 推荐 4：OpenMathInstruct-2

链接：

- https://huggingface.co/datasets/nvidia/OpenMathInstruct-2

适合用途：

- 大规模数学 SFT。
- 服务器实验。

优点：

- 数据量大。
- 是成熟数学 instruction tuning 数据。

缺点：

- 对当前本地第一版太大。
- 不一定是 R1 `<think>` 风格。

## 当前最推荐选择

如果你现在想少折腾、直接 SFT：

```text
RabotniKuma/Fast-Math-R1-SFT
```

理由：

- 规模 7900，刚好适合本地。
- 已经筛过正确 R1 trace。
- 和 OpenR1 / AIMO 风格接近。
- 可以直接拿 5000 条做 SFT。

如果你要严格贴 AIMO2 第二名路线：

```text
qihoo360/Light-R1-SFTData
```

理由：

- AIMO2 第二名提到了 Light-R1 stage2 data。
- 更接近他们的 SFT 数据来源。

如果你要做 DPO：

```text
继续使用 open-r1/OpenR1-Math-220k
```

理由：

- 原始数据有多 generation。
- 有 correctness 标记。
- 可以构造 chosen/rejected pair。

## 建议组合

第一版：

```text
SFT: Fast-Math-R1-SFT 5000 条
DPO: OpenR1-Math-220k 本地分片构造 3000-5000 pair
```

第二版：

```text
SFT: Light-R1-SFTData stage2 + Fast-Math-R1-SFT
DPO: OpenR1-Math-220k 构造 pair
```

这样既省了 SFT 清洗工作，又保留了 DPO 的研究空间。
