# Day 15：CUDA索引，让每位工人只处理自己的元素

前置：[Day 14基线](DAY14_MODEL_BASELINE.md)。从CPU数学进入GPU执行，不从优化参数开始。

## 1. 图解与数字例子

把grid看成整张工单，block是小组，thread是组内工人：

```text
每组256人，任务257个元素
组0：全局位置0..255
组1：全局位置256..511，但只有位置256有效
```

索引 `i=blockIdx.x*blockDim.x+threadIdx.x`。
第1组第0人处理位置256，第1组第1人算出257，必须不读不写。
多启动的255位工人不是错误，缺少尾部Mask才是越界风险。

## 2. 概念与计算契约

`blocks=ceil(N/256)`，一维Vector Add每个有效线程读取a[i]/b[i]并写out[i]。
warp在本NVIDIA环境中有32个线程，block可以含多个warp。
CUDA thread block与KV页block不是同一个概念；后者是存储单位。
相邻线程访问相邻地址有利于合并事务，实际效果仍依赖对齐和硬件。

本例 `vector_reduce.cu` 是独立C++程序，不依赖PyTorch扩展ABI；
它没有自动测生产吞吐，也不是本机所有CUDA扩展都兼容的证明。

## 3. Jupyter与源码路线

打开 [Day 15 Notebook](../learning/day15_course.ipynb)，定位
[vector_reduce.cu](../learning/kernels/vector_reduce.cu) 的vector_add。
脚本无编译器路径时报告requires_cuda_compiler，不计通过。

在**宿主机**已有nvcc的终端编译，先检查实际compute capability；
以下sm_121仅对应当前Spark记录，不是所有GPU：

```bash
nvcc -arch=sm_121 -O2 learning/kernels/vector_reduce.cu -o /tmp/course-vector-reduce
docker compose -f docker-compose.spark.yml cp /tmp/course-vector-reduce reasoning:/workspace/logs/course-vector-reduce
```

Notebook运行前在setup之后、run_day之前设置：

```python
os.environ["CUDA_LESSON_BINARY"] = "/workspace/logs/course-vector-reduce"
```

程序测试N=1/255/256/257/1003/65537。某些环境的容器没有nvcc，
不能把宿主机路径直接当容器路径；用上面的复制步骤。

## 4. 调试与优化顺序

1. 先手算N=257的grid和尾部有效线程。
2. 检查cudaMalloc/Memcpy/launch/synchronize返回值，避免只看kernel启动成功。
3. 实际运行所有边界，必要时在支持环境用Compute Sanitizer检查非法访问。
4. 保持索引正确后再调整launch和访存；本reduction部分固定256，不可只改launch使两者不匹配。
5. 计时应另建同工作负载基准；当前JSON只有正确性，不含带宽结果。

## 5. 自测与参考答案

**问：257个元素为什么启动512个线程？**
答：两个完整256线程block，每个线程用i<N判断是否有有效工作。

**问：warp与block一样吗？**
答：不一样，warp是硬件执行分组，block是程序定义的协作/共享内存单位。

**问：有nvcc就代表kernel能运行吗？**
答：不能，还需设备、驱动、适用架构和实际launch验证。

**问：为什么不要直接删除尾部判断？**
答：最后block可能读取/写入越界地址，即使一次未崩溃也不正确。

## 6. 验收与资料

- [ ] 画出257元素的线程表并运行六组边界。
- [ ] 记录编译命令、实际GPU与可执行文件路径；缺依赖如实记状态。

深入：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)；
[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。
下一课：[Day 16](DAY16_SHARED_REDUCTION.md)。
