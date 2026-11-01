# Day 6：Attention 为什么占内存，分块后如何算出相同结果

前置：[Day 5 配置与内存](DAY05_ARCHITECTURES.md)。
个人进度仍记录为 Day 5；本页是下一步材料，不自动认定已通过。

## 1. 今天只回答一个核心问题

普通 Attention 常先算整张 score，再 softmax，再与 V 相乘：

```text
Q × K.T → score [B,H,T,T] → probabilities [B,H,T,T] → ×V → 输出
```

完整 prefill 的每个位置都要与一组可见位置比较，因此出现两条长度轴。
即使 Mask 上三角为禁止，朴素实现通常仍分配矩形张量，不能自动减半。

以 B=1、H=16、T=4096、每元素2字节为例，单张 score 是512 MiB；
T=8192 则为2048 MiB。实际 dtype 或是否物化由实现决定。
这不是模型权重、不是 KV Cache，也不是完整训练/推理峰值。

**问题不是“能否只算一小块”，而是“算小块时怎样合并到正确的归一化结果”。**

## 2. 先看一行，而不是整个模型

已经给定一个 Query 的分数与对应 Value（均为教学数字）：

```text
scores = [1, 2, 3, 4]
values = [10, 20, 30, 40]
```

答案是：

```text
output = softmax(scores) · values ≈ 34.9265
```

为了稳定，先减去最大分数4：

```text
unnormalized weights = [exp(-3), exp(-2), exp(-1), 1]
denominator = 1.5530
numerator = 10×exp(-3)+20×exp(-2)+30×exp(-1)+40
output = numerator / denominator
```

softmax 的概率必须按所有可见 key 一起归一化，而不是每块各自归一化就结束。

## 3. 为什么“每块独立 softmax，再平均”是错的

把输入拆成 `[1,2]` 和 `[3,4]`：

```text
第1块独立 softmax 的加权结果 ≈ 17.3106
第2块独立 softmax 的加权结果 ≈ 37.3106
简单平均 ≈ 27.3106
```

这不等于34.9265。因为两块被错误地分配了同样的总权重；
真实分数更大的第2块应该分到更多权重。

类似合并两份统计时，只知道“各自平均值”不够，还需要合适的权重。
这里要保存的不是每块最终概率，而是一个可重缩放的累积状态。

## 4. 每一行保留三个量

```text
m：见过的最大 score
l：以 m 为基准的指数和（分母；小写字母 l，不是层数 L）
o：以 m 为基准的加权 Value 和（分子，通常是一个向量）
```

先处理第1块 scores=[1,2]、values=[10,20]：

```text
m = 2
l = exp(1-2)+exp(2-2) = 1.3679
o = 10×exp(1-2)+20×exp(2-2) = 23.6788
```

再处理第2块，最大值变成4。旧的 l/o 用2作基准，新块用4作基准，
不能直接相加，先换成同一基准：

```text
旧基准2 → 新基准4
旧累积量统一乘 alpha = exp(2-4)

新的 l = alpha×旧l + exp(3-4)+exp(4-4)
       ≈ 1.5530

新的 o = alpha×旧o + 30×exp(3-4)+40×exp(4-4)

最终 output = 新o / 新l ≈ 34.9265
```

换基准的意义可直接看一个元素：

```text
exp(1-2) × exp(2-4) = exp(1-4)
```

没有“神奇压缩概率”，只是重写了同一套指数加权计算。

## 5. 公式只记这一组

新块 score 为 s、对应 Value 为 V：

```text
m_new = max(m_old, max(s))
alpha = exp(m_old - m_new)
p = exp(s - m_new)                  # 暂未归一化的块内权重
l_new = alpha*l_old + sum(p)
o_new = alpha*o_old + p @ V
output = o / l                     # 所有块完成后
```

实现细节：

- 初始 m=-inf、l=0、o=0；至少要有一个有效 key，避免全 Mask 行产生 NaN。
- Mask 禁止的 score 按 -inf 处理，对指数和的贡献为0。
- m/l/o 常用 FP32 累积，输出可恢复原 dtype；浮点实现不保证逐位相同。
- 实际 Attention 的 o 是每个 query/head 一行向量，不只是这里的标量。
- 分块仍计算 dense score，只是不同时保存完整 T×T，主算术复杂度并未变成线性。

## 6. 在 Jupyter 怎么学

先打开 `learning/day06_online_softmax.ipynb`：

1. 先预测普通答案和错误平均答案。
2. 逐块观察 m/l/o，核对换基准公式。
3. 改块宽为1、2、3，检查结果是否仍对齐。
4. 将 score 增大到1000附近，观察稳定性。
5. 运行分块 Attention 对照 SDPA。

再打开 `learning/day06_course.ipynb`，看统一实验和 JSON 报告。
`learning/online_attention.py` 是可读的 PyTorch 参考，不能因“不写整张 score”
就宣称有生产 FlashAttention 的 GPU 性能。真正融合 GPU 实验在 Day 20–21。

## 7. 面试主问与追问

**主问**：FlashAttention 为什么能减少显存读写？

**答案骨架**：按 Q/K/V 分块，使用 online softmax 累积 m/l/o，
避免完整 score/probabilities 写回显存，并利用片上复用与算子融合。
代价与收益受 tile、dtype、硬件和输入规模影响；不是免费减少全部计算。

**追问一**：KV Cache 也因此不需要了吗？
答：不。KV 是历史请求状态，score 是本轮中间结果；优化的是不同对象。

**追问二**：分块参考等价就证明 FlashAttention 训练正确吗？
答：不。这里只证明支持输入上的 forward；训练 backward、dropout、
异长、GQA、不同 Mask 等需要另外的实现与测试。

**追问三**：为什么要保存最大值而不直接累加 exp？
答：指数可能溢出；更新最大值并重缩放旧量，可以稳定地合并不同块。

**今日验收**：不用背公式也能解释 `exp(1-2)×exp(2-4)=exp(1-4)`，
并说清分母 l 和分子 o 为什么必须一起乘 alpha。

参考：[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)；
[Triton 官方 Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。
