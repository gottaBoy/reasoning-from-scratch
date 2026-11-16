# P2-01：CUDA Vector Add，线程负责哪一个数？

P1-07已完成小模型组装和定义范围内的验收。现在进入GPU执行层，
不改Attention公式，不做共享内存归约，不追求加速比。

## 1. 文本、图像、音频、视频：相同的是特征逐元素相加

先把两个同shape的特征张量看成两排数：

```text
a   = [ 1,  2,  3]
b   = [10, 20, 30]
out = [11, 22, 33]
```

| 模态 | 可类比的相加场景 | 本练习处理什么 |
| --- | --- | --- |
| 文本 | 一个token的残差特征 + 更新特征 | 每个特征位置的两个浮点数 |
| 图像 | 一个patch的残差特征 + 更新特征 | 每个特征位置的两个浮点数 |
| 音频 | 一帧音频的残差特征 + 更新特征 | 每个特征位置的两个浮点数 |
| 视频 | 一个时空token的残差特征 + 更新特征 | 每个特征位置的两个浮点数 |

这里不是把token ID相加，也不是直接拼接两张图片或两个音视频文件。
在这份独立CUDA练习里，输入已经是连续的一维FP32数组，`N`是浮点元素总数，
不是token数、patch数、秒数或视频帧数。多模态模型只作特征运算的类比。

## 2. 为什么需要全局索引

固定一个block有256个线程：

```text
block 0：thread 0..255 → 全局元素   0..255
block 1：thread 0..255 → 全局元素 256..511
block 2：thread 0..255 → 全局元素 512..767
```

三个量：

```text
blockIdx.x  = 当前block编号
blockDim.x  = 每个block的线程数，本关固定256
threadIdx.x = 当前block内的线程编号
```

所以全局索引为：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

只用 `threadIdx.x` 会让不同block重复处理前256个数，
后面的数无人处理。这里的CUDA block是线程协作单位，不是KV Cache的存储页。

## 3. 255、256、257为什么都要测

```text
N=255 → 1个block → 256个线程 → 最后1个无效
N=256 → 1个block → 256个线程 → 全部有效
N=257 → 2个block → 512个线程 → 后面255个无效
```

上取整由host端计算：

```cpp
int blocks = (n + 255) / 256;
```

kernel只允许有效线程访问数据：

```cpp
if (i < n) out[i] = a[i] + b[i];
```

先手算：`N=257`、`blockIdx.x=1`、`threadIdx.x=17`，
则 `i=273`，这位线程必须不读、不写。

## 4. 个人文件与验收职责

打开 `learning/my_vector_add.cu`，只填两个位置：

- V1：计算全局索引。
- V2：在范围内读取a/b并写out，不修改a/b。

独立验收程序 `learning/kernels/vector_add_check.cu` 提供main，
负责分配、拷贝、launch、同步和CPU FP64数值对照；不要把main写进个人kernel文件。
参考kernel另放在 `learning/kernels/vector_add.cu`，不把参考通过算作个人实现通过。

验收覆盖 `N=1/255/256/257/1003/65537`：

- 逐元素结果正确且有限，输出预填NaN，空kernel不能冒充通过。
- 输入a/b逐值不变。
- 输出前后guard及逻辑尾部不被写入，漏掉边界判断会失败。
- 所有CUDA分配、拷贝、launch及同步错误明确失败。
- 可要求Compute Sanitizer memcheck；缺工具不会静默跳过。

guard检查补充逻辑边界，不能证明所有越界读或所有并发错误都不存在；
memcheck通过也不能替代数值和输入保护检查。

## 5. 在宿主机运行

本轮宿主机已有CUDA Toolkit，不依赖PyTorch，不需要重建Docker。
GB10的实际compute capability为12.1，因此本机使用 `sm_121`；
更换设备时先核对架构，不把此值照搬到其他GPU。

```bash
# 先验证参考链路
python3 -m learning.practice_vector_add --arch sm_121 --sanitize

# 再验收自己的kernel
python3 -m learning.practice_vector_add --arch sm_121 \
  --candidate learning/my_vector_add.cu --sanitize
```

报告使用新的UTC文件名写到 `logs/practice/`，包含源码/harness hash、
编译命令、实际capability、各组错误计数及memcheck日志，不覆盖旧证据。
未安装nvcc、没有GPU、编译错误或显式要求的memcheck失败都不算通过，
不会改用CPU相加掩盖CUDA失败。
本机本轮并行运行两份memcheck时曾出现一次cudaMalloc内存不足，串行重跑通过；
上面两条命令按顺序运行。资源错误保留报告，不放宽正确性检查。

## 6. 只改一个错误做验证

先正确实现，再分别做两个临时错误版本：

1. 去掉 `i < n`，观察非整齐N的guard错误。
2. 把全局索引改成 `threadIdx.x`，观察 `N > 256` 的遗漏。

恢复正确实现后复跑全部大小。本关不包含吞吐、带宽或kernel计时，
也不进入P2-02的shared memory reduction。

官方资料：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)、
[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。
