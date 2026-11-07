# Day 18：Softmax与LayerNorm，数学等式不等于数值算法

前置：[Day 17行算子](DAY17_TRITON_ROWS.md)、[Online Softmax](DAY06_ONLINE_SOFTMAX.md)。

## 1. 图解与数字例子

大分数像超量程仪表，直接exp可能溢出：

```text
[1000,1001] → 先减最大值 → [-1,0] → exp → 归一化
结果约 [0.268941,0.731059]
```

行宽3补到4做softmax，补位若用0而真实分数为 `[-5,-6,-7]`，
假补位exp(0)反而占大权重；应把无效分数改为负无穷。
LayerNorm常数行 `[2,2,2]` 减均值后为0，非affine输出为0；
RMSNorm常数行通常不是零向量。

## 2. 概念与计算契约

```text
Softmax：exp(x-max(x))/sum(exp(x-max(x)))
LayerNorm：(x-mean(x))/sqrt(mean((x-mean(x))²)+eps)
```

本kernel的LayerNorm是非affine版本，不含gamma/beta。
有效宽度WIDTH参与均值/方差分母，padding不算真实特征。
高均值低方差时 `E[x²]-E[x]²` 易消去误差，源码先中心化再算平方均值。
极端数值仍有有限精度边界，不能把“减最大值”说成保证所有NaN/Inf输入可用。

## 3. Jupyter与源码路线

打开 [Day 18 Notebook](../learning/day18_course.ipynb)，需要实际CUDA。
定位_rows的MODE=1与else，逐行标注有效load、padding处理、归约、store。
参考实验对齐3×127的softmax与PyTorch非affine LayerNorm。

独立追加常数、全零、大幅值、宽度1及1003，记录dtype与容差。
本wrapper没有任意attention mask接口，全Mask行策略不能从本实验自动推断。
softmax不仅要检查行和，还要检查各元素与错误padding负例。

## 4. 调试与优化顺序

1. 检查load补值与tl.where顺序是否符合运算。
2. 用独立FP64公式观察误差，避免两个实现共享同一个错误参考。
3. 检查WIDTH不被BLOCK替代，方差只计有效位置。
4. 比较短行/长行，分清launch、访存、归约成本。
5. 优化block/warps之前先保留基线，并重测常数与极端输入。

## 5. 自测与参考答案

**问：为什么softmax补位不是0？**
答：0分数仍有exp(0)权重，可能扭曲有效概率。

**问：概率和接近1就够了吗？**
答：不够，错分布也可和为1，应逐元素对齐。

**问：本LayerNorm有可学习gamma吗？**
答：没有，比较参考时用非affine形式，不能省略差异。

**问：RMSNorm和LayerNorm遇到常数正向量有何不同？**
答：LayerNorm先减均值成0；RMSNorm不减均值，通常保留非零方向。

## 6. 验收与资料

- [ ] 说明padding如何影响分母与最大值。
- [ ] 交付不同幅值/宽度/dtype的误差表，不只随机一次。

深入：[Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)；
[PyTorch LayerNorm](https://docs.pytorch.org/docs/2.10/generated/torch.nn.LayerNorm.html)。
下一课：[Day 19](DAY19_TILED_MATMUL.md)。
