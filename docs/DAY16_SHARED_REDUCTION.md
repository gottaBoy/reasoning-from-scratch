# Day 16：Reduction，先小组汇总，再合并总和

前置：[Day 15索引](DAY15_CUDA_INDEXING.md)。本程序固定每block 256线程。

## 1. 图解与数字例子

8个数字求和可像折叠纸条：

```text
[1,2,3,4,5,6,7,8]
第1轮：前4项分别加后4项 → [6,8,10,12]
第2轮：前2项加后2项    → [16,20]
第3轮：第1项加第2项    → [36]
```

每轮后大家确认上一轮写完，再进入下一轮，这就是block barrier的用途。
N=257时，两个block先产生两个partial sums，再由CPU合并。
**本课没有把跨block的全局归约伪装成同一个GPU block里的同步。**

## 2. 概念与计算契约

每个线程先把自己的数写shared数组；越界线程贡献0，但仍参加同步。
stride每轮减半，对应示例128/64/.../1。
`__syncthreads()`只协作当前block，不能完成整个grid的普通全局barrier。
sum填0，max通常填负无穷；补充值取决于归约运算。

浮点加法顺序改变可能产生误差，不能用数学结合律要求bitwise相同。
程序用CPU double求参考，GPU算block partial、CPU合并，分别标明精度和执行位置。

## 3. Jupyter与源码路线

沿用 [Day 15的编译步骤](DAY15_CUDA_INDEXING.md)，打开
[Day 16 Notebook](../learning/day16_course.ipynb)。
同一binary验证Vector Add和Shared Memory Reduction；
设置CUDA_LESSON_BINARY后才执行run_day，缺失时为requires_cuda_compiler。

定位block_sum中的shared写入、每个barrier、stride循环与partial写回。
先在纸上跟踪8项例子，实际程序仍按256线程运行，不随纸上示例随意改配置。

## 4. 调试与优化顺序

1. 所有线程先写shared位置，保证没有读取未初始化元素。
2. 越界线程填0，而不是在barrier前提前退出。
3. 循环内只有加法条件化，block barrier仍在一致控制流中。
4. 检查partial数组长度=block数，CPU合并涵盖全部partial。
5. 之后再研究warp shuffle、分层归约与二次GPU kernel；这些不是当前实现已完成的功能。

共享内存可能有bank conflict、容量与同步成本，不能只按存储名字认定必快。
性能比较需区分最终CPU传输合并时间与GPU局部归约。

## 5. 自测与参考答案

**问：最后block只有一个有效元素，其他线程能直接return吗？**
答：本程序的同步模式下不应这样做；无效线程补0并参与协作。

**问：一个block算完就能读另一个block结果吗？**
答：没有隐式执行次序或全局barrier；要用有定义的跨kernel/全局协作方案。

**问：为什么CPU double与FP32 partial不要求逐位一样？**
答：运算精度和加法次序不同，应报告合理容差与输入。

**问：当前最终合并在哪？**
答：CPU，不是完整全GPU reduction。

## 6. 验收与资料

- [ ] 手算折叠过程并解释每个同步点。
- [ ] 运行边界，标明GPU局部和CPU全局两个阶段。

深入：[CUDA同步与共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)；
[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。
下一课：[Day 17](DAY17_TRITON_ROWS.md)。
