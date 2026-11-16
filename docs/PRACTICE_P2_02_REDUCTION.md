# P2-02：共享内存归约，8个线程怎样合作算出36？

上一关每个线程各写一个相加结果。本关每个block合作产生一个结果：

```text
输入N个数 → 每256个数交给一个block → 每block输出一个partial
                                           ↓
                                  CPU把所有partial相加
```

先只练求和，不写均值、RMSNorm、warp shuffle或第二个GPU归约kernel。

## 1. 先看清每个线程的读写

手算时假设只有8个线程。每个线程先往本block共享数组放一个数：

```text
线程编号： t0 t1 t2 t3 t4 t5 t6 t7
数组下标：  0  1  2  3  4  5  6  7
数组数值：  1  2  3  4  5  6  7  8
```

加载后先同步，保证读其他线程的格子之前，加载阶段已经完成。

第一轮stride=4，只有前4个线程做加法：

```text
t0：values[0] = values[0] + values[4] = 1+5 = 6
t1：values[1] = values[1] + values[5] = 2+6 = 8
t2：values[2] = values[2] + values[6] = 3+7 = 10
t3：values[3] = values[3] + values[7] = 4+8 = 12
```

下一轮只需要前4个格子，原来的8个格子仍然存在。同步后：

```text
stride=2：
t0：values[0] = 6+10 = 16
t1：values[1] = 8+12 = 20

再同步，stride=1：
t0：values[0] = 16+20 = 36
```

第二轮t0会读第一轮t2写出的10，这就是两轮之间需要同步的具体原因。
加法放在 `tid < stride` 条件里；本实现的block同步放在条件外。

## 2. 正式练习固定256线程

实际launch固定每block 256线程，因此stride从128开始：

```text
128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
```

对N=257：

```text
block 0：加载input[0..255] → partial[0]
block 1：线程0加载input[256]，其他255个格子填0 → partial[1]
CPU：partial[0] + partial[1]
```

尾部线程仍需初始化自己的格子，否则别的线程会读到未定义的值。
加0不改变求和结果；本算法不使用 `if (i >= n) return` 代替初始化和协作。
每个block拥有自己的共享数组，本block的同步不负责跨block汇总。

## 3. 四种模态怎样对应

| 模态 | 可以用来练习求和的一组特征 |
| --- | --- |
| 文本 | 某个token的一组通道数值 |
| 图像 | 某个patch的一组通道数值 |
| 音频 | 某个时间帧的一组通道数值 |
| 视频 | 某个时空token的一组通道数值 |

本练习只收到一维浮点数组，不知道这些数来自哪种模态。
这里是特征求和的类比，不是在求token ID总和、叠加文件或实现完整多模态模型。
将来做RMSNorm时还需平方、求均值、epsilon和缩放；本关先不加入这些步骤。

## 4. 只填写个人文件R1–R5

打开 `learning/my_block_sum.cu`：

1. R1：声明每block共用的 `values[256]`。
2. R2：计算tid和全局i；每线程加载一个数，尾部填0。
3. R3：加载后同步一次。
4. R4：循环折半加法，每一轮后同步。
5. R5：仅线程0把 `values[0]` 写到 `partial[blockIdx.x]`。

保留函数名、签名和固定256线程契约，个人文件不需要main。
`learning/kernels/block_sum_check.cu`负责分配、launch、读取和CPU参考；
`learning/kernels/block_sum.cu`是独立的完整参考，不替换个人文件。

## 5. 在宿主机验收

现有GB10使用sm_121，换GPU先核对架构。宿主机命令不需要PyTorch：

```bash
# 参考链路：只确认验收环境
python3 -m learning.practice_reduction --arch sm_121 --sanitize

# 个人实现：填写后执行
python3 -m learning.practice_reduction --arch sm_121 \
  --candidate learning/my_block_sum.cu --sanitize
```

每次生成新报告，数值通过后依次运行memcheck、racecheck、synccheck。
三项工具运行的是同一组输入；缺失或没有清洁日志不会静默记为通过。
数值失败时保留失败结果，不继续运行sanitizer。

六组大小为1/255/256/257/1003/65537，每组分别用全1、带正负号的二进制分数、
非二进制分数测试。验收逐block比较CPU FP64参考，不能只看最后总和。
partial容差为 `1e-4 + 1e-5 * abs(expected)`；
CPU最终合并的容差取各partial容差之和。
还会检查输出guard、输入不修改和输出有限性，未写的partial预填NaN。

本轮参考6/6及三项sanitizer均通过，个人框架0/6，尚未完成实现。
原始失败和通过报告保存在 `logs/practice/`。
首次参考执行曾在cudaMalloc报内存不足，串行重试通过；资源失败报告仍保留，
不放宽容差，不关闭其他用户进程。

## 6. 完成后的复盘

先说清：哪几个线程做加法、下一轮读取谁的结果、同步点在哪里，
以及最终跨block相加发生在CPU。
再用临时副本分别改错尾部填零、partial索引或同步位置，验证错误能被发现。
候选代码可能产生同步死等，执行有超时，不把未结束或sanitizer失败记为通过。
本关只验功能和上述边界，不提供带宽或加速比。

官方资料：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)、
[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。
