# Day 21：融合Attention与SwiGLU，少搬中间结果不是少算一切

前置：[Day 20正确性基线](DAY20_ATTENTION_ORACLE.md)、[Day 6 m/l/o](DAY06_ONLINE_SOFTMAX.md)。

## 1. 图解与数字例子

像加工工序直接传给下一道，不把每个半成品送回大仓库：

```text
Attention：Q/K局部分数 → m/l/o更新 → 下一块 → 最终O
SwiGLU：  gate → SiLU ─┐
          up ──────────┴→ 相乘 → 输出
```

gate=2、up=3时，输出为 `SiLU(2)*3 ≈ 5.2848`。
这里的SiLU可以输出负数或大于1，不是路由概率。
融合SwiGLU本例只融合SiLU与乘法，不包含E→F的两次投影及F→E的GEMM。

## 2. 概念与计算契约

Attention固定BQ=16、BK=32，逐Key块更新FP32最大值、分母、累积输出，
不把完整T×T score/probabilities写出。
wrapper只支持连续CUDA、同dtype/device、等长causal prefill及D=16/32/64。
SwiGLU接受gate/up同shape，forward-only；不替你自动生成训练梯度。

“exact”指稠密Attention数学定义，不指浮点逐位相同。
减少IO、launch和部分中间分配可能受益，但资源压力也会抵消，不能预先规定加速比。

## 3. Jupyter与源码路线

打开 [Day 21 Notebook](../learning/day21_course.ipynb)，需要CUDA。
看实际Triton/SDPA误差与同步wall-clock，别只读status。
定位_attention的Q驻留、K/V循环、scale/denom/acc，
再看_swiglu中load、FP32激活、乘法与store。

先预测5.2848，运行后把gate改负数；实际大实验用现有正确性测试覆盖边界。
本课参考没有生产autotune、多GPU、backward或完整FFN融合。

## 4. 调试与优化顺序

1. 相同输入先对齐，不以放宽容差换取“更快”。
2. 检查tile尾部、因果可见性、最大值变更时分子分母同步缩放。
3. JIT/预热不计入稳态样本，记录实际shape与后端。
4. 观察寄存器、共享内存、occupancy和launch成本，定位主因。
5. 单变量改tile或实现，复跑全支持边界；允许优化后更慢并保留结果。

报告当前函数名与支持表，比把它命名为“生产FlashAttention”更重要。

## 5. 自测与参考答案

**问：为什么融合SwiGLU不等于融合整个FFN？**
答：三个线性投影GEMM不在该逐元素kernel里，它只接收已算出的gate/up。

**问：融合越多越快吗？**
答：不一定，资源占用与调度自由度下降可能超过减少IO/launch的收益。

**问：优化kernel最少交付什么？**
答：支持契约、独立正确性、负例/边界、精度、同步计时、环境与适用范围。

**问：m/l/o能替代KV Cache吗？**
答：不能，m/l/o是当前Query的累积状态，KV是供不同Query读取的历史表示。

## 6. 验收与资料

- [ ] 解释局部中间数据在哪里消费，哪些完整数组不再写回。
- [ ] 交付对齐与计时报告，明确forward-only与缺失功能。

深入：[Triton Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)；
[FlashAttention原论文](https://arxiv.org/abs/2205.14135)。
下一课：[Day 22](DAY22_SERVING_BOUNDARY.md)。
