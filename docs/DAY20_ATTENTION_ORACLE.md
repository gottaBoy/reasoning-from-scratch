# Day 20：Attention基准，先确认比较的是同一个问题

前置：[Day 6分块数学](DAY06_ONLINE_SOFTMAX.md)、[Day 19](DAY19_TILED_MATMUL.md)。

## 1. 图解与数字例子

比较两份“账本”要让资料、可见范围和精度一样：

```text
同Q/K/V
   ├─ 清晰PyTorch分块参考 → O_ref
   ├─ SDPA后端             → O_sdpa
   └─ 支持范围内Triton     → O_gpu
                ↓ 全部输出误差与负例
```

T=17、D=32、Q tile=16时有两个Query tile，第二块只含1个真实Query；
Key tile=32时也需要尾部mask。
单token Decode是1×T_kv矩形，不是T×T的缩小版方阵。

## 2. 概念与计算契约

PyTorch `blocked_attention`是数学参考，允许用offset表达简单矩形因果关系，
但要求相同head数、K/V同shape及相同D；不是通用GQA接口。
当前Triton只支持同shape的 `[B,H,T,D]` 等长causal prefill，
D为16/32/64，无Cache、padding、backward、GQA。

本课CPU路径可验证数学；只有CUDA路径才额外执行教学Triton kernel。
SDPA可能选择不同后端，不能看API名就认定用了FlashAttention。

## 3. Jupyter与源码路线

打开 [Day 20 Notebook](../learning/day20_course.ipynb)。
先阅读report.scope和device，再看数值/计时结果。
定位 `online_attention.blocked_attention` 与 `triton_ops.attention` 的输入契约。
自己尝试T=1/17/65、支持D，检查禁止位置及未来token扰动。

若要比较Decode，使用支持该形状的数学参考与SDPA；
不要把非方阵传给当前Triton后放宽阈值让它“通过”。

## 4. 调试与优化顺序

1. 固定Q/K/V、Mask语义、scale、dtype和dropout=0。
2. 先清晰oracle，查score、分母与每头输出，再比GPU。
3. 全Mask行、padding等不支持情况明确拒绝或不列入支持范围。
4. 预热后比较同一工作负载，记录实际后端与同步口径。
5. 只在正确的矩阵规模上调整tile，性能和误差分别报告。

当前参考的causal顺序确保每行先看到合法Key；不要把这点推广到任意稀疏mask的稳定性保证。

## 5. 自测与参考答案

**问：两个自己的实现相同就够了吗？**
答：不够，可能共享同一错误；需要独立oracle、因果性与边界负例。

**问：CPU报告passed是否证明GPU kernel运行了？**
答：不是，应看device和具体字段，GPU项目要有真实调用证据。

**问：为什么不直接复用方形causal mask做Cached Decode？**
答：Query有历史位置偏移，非方阵Mask的对齐约定必须核对。

**问：本kernel能直接训练吗？**
答：不能，forward-only，不提供完整backward和autograd契约。

## 6. 验收与资料

- [ ] 列三种实现的支持范围，并保存同输入误差与一个不支持例。
- [ ] 将prefill/decode、数学路径/GPU路径分开报告。

深入：[PyTorch SDPA](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)；
[Fused Attention教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。
下一课：[Day 21](DAY21_FUSED_KERNELS.md)。
