# Day 17：Triton的program与行算子，一张工单处理一块数据

前置：[Day 15–16](DAY16_SHARED_REDUCTION.md)、[RMSNorm](DAY08_RMSNORM_ROPE.md)。

## 1. 图解与数字例子

Vector Add的N=1003、BLOCK=256：

```text
program0 → offsets 0..255
program1 → offsets 256..511
program2 → offsets 512..767
program3 → offsets 768..1023，只存到1002
```

一个Triton program不等于一个CUDA thread。
行宽127做RMSNorm时，载入128个lane形状，最后一个补0，
**均方分母仍是127，不是128**，最终只写127个有效值。

## 2. 概念与计算契约

`program_id`选tile，`arange`构建块内索引，load/store mask管合法地址，
constexpr控制编译期专门化。改变shape或constexpr可能触发新JIT。
本wrapper只支持连续CUDA FP32/FP16/BF16、同device/dtype、forward-only；
要求grad、CPU输入、非连续布局会明确拒绝，不静默搬设备或修布局。

`row_op`输入二维 `[rows,width]`，宽度1..8192；
RMSNorm可给每特征weight，内部FP32累积。输出shape/dtype保持输入契约。

## 3. Jupyter与源码路线

打开 [Day 17 Notebook](../learning/day17_course.ipynb)，先确认DEVICE=cuda。
CPU报告requires_cuda，不把跳过写成通过。
定位 `triton_ops._add/_rows` 及wrapper的_check，先看合法输入如何传入kernel。
参考实验验证1003长度加法和3×127行RMSNorm。

自己追加width=1/7/127/1003与三种dtype，
gamma不能只用全1；边界和精度按实际计算契约分别验证。
编译观察补做 [IR讲义](TRITON_IR.md) 与
[真实产物Notebook](../learning/day17_triton_ir.ipynb)。

## 4. 调试与优化顺序

1. 先手算program grid、offset和末尾mask，再运行。
2. 用PyTorch/高精度参考逐元素对齐，查RMS分母与weight广播。
3. 非连续、不同dtype或requires_grad等负例必须按契约拒绝。
4. 预热后同步计时；把JIT首次耗时单独报告。
5. 有profiling证据后再改BLOCK/warps等参数；别把IR导出当成加速证据。

当前默认tile由wrapper选择；要调参需显式扩展受控入口并重新对齐，不能声称所有选项已可配置。

## 5. 自测与参考答案

**问：program等于thread吗？**
答：不是，它描述tile计算，底层线程/warp映射由编译器与配置决定。

**问：127补到128后，为什么不能除128？**
答：补位只是计算布局，不是多了一个真实输入特征。

**问：输入有requires_grad怎么办？**
答：当前wrapper拒绝；要训练需独立backward/autograd设计，不能直接detach掩盖需求。

**问：第一次调用很慢说明kernel很慢吗？**
答：可能包含JIT等开销，应区分启动与预热后稳态。

## 6. 验收与资料

- [ ] 画program索引图，说明所有mask和有效宽度。
- [ ] 至少记录一个合法边界和一个拒绝用例。

深入：[Triton Vector Add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)；
[LayerNorm教程](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)。
下一课：[Day 18](DAY18_STABLE_ROW_OPS.md)。
