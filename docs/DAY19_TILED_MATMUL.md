# Day 19：MatMul分块，搬来的材料要多用几次

前置：[Day 17](DAY17_TRITON_ROWS.md)、[Day 14计时](DAY14_MODEL_BASELINE.md)。

## 1. 图解与数字例子

把矩阵乘看成小块积木：

```text
A [M,K] × B [K,N] → C [M,N]
每个program负责C的一块BM×BN
沿K轴反复搬A/B小块 → acc += A_tile @ B_tile
```

M=65、N=49、K=33，tile=32时，输出grid=3×2。
K循环两次，第二次只有1个有效K元素，其余补0。
最后M tile只剩1行，最后N tile剩17列；必须分别mask读取与写出。

## 2. 概念与计算契约

主要乘加FLOPs近似2MNK，不是2MNT。
本wrapper接受连续CUDA二维同dtype矩阵，tile只支持16/32/64；
BK固定32，BM/BN取tile，acc用FP32，dot明确使用IEEE输入精度设置。
大tile增加复用机会，也可能增加寄存器/共享内存压力、尾部浪费、减少并行块。

不能把本例叫完整autotune，不能默认Tensor Core路径或TF32设置对所有dtype相同。

## 3. Jupyter与源码路线

打开 [Day 19 Notebook](../learning/day19_course.ipynb)，需要CUDA。
定位_matmul的rows/cols/kk、A/B地址、两个load mask、acc累加与store mask。
参考实验比较16/32/64并计时，oracle是实际输入转CPU FP64后矩阵乘。

先在纸上画65×49输出tile，另测17×33乘33×19。
三种tile必须先全部对齐，再读时间报告；记录median、device、dtype及精度模式。

## 4. 调试与优化顺序

1. 用小整数矩阵核对A行跨度K、B行跨度N，避免stride写错。
2. 检查M/N/K三种尾部，不只改一个非整齐维度。
3. K每轮是累加而不是覆盖；比较独立高精度oracle。
4. 预热后同步计时，多尺寸看launch-bound与compute-bound区别。
5. 用profiling定位资源瓶颈，再单变量试tile；不挑最有利的一个shape宣传。

TF32等全局状态会影响PyTorch FP32参考，独立CPU FP64 oracle可避免把参考模式变更当kernel错误。

## 5. 自测与参考答案

**问：tile=32时65×49要几块？**
答：ceil(65/32)×ceil(49/32)=3×2，不是向下取整。

**问：为什么K尾部补0？**
答：对矩阵乘求和0是中性贡献，越界数据不能参与。

**问：tile越大越快吗？**
答：不一定，资源占用、并行度和尾部利用率会改变。

**问：与cuBLAS一个小例子相比更快，能替代它吗？**
答：不能，要覆盖支持范围、精度、规模和多种硬件，本课只是教学kernel。

## 6. 验收与资料

- [ ] 写出3×2 grid与两次K循环，指出每个边界mask。
- [ ] 输出不同tile的正确性及同步计时，不承诺正加速。

深入：[Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)；
[PyTorch数值精度](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)。
下一课：[Day 20](DAY20_ATTENTION_ORACLE.md)。
