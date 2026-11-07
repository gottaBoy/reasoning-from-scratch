# Day 6：Attention 为什么占内存，分块后如何算出相同结果

前置：[Day 5 配置与内存](DAY05_ARCHITECTURES.md)。
2026-11-07用户确认开始Day 6；当前状态与个人验收见
[进度记录](LEARNING_PROGRESS.md) 和 [Day 6学习记录](DAY06_STUDY_LOG.md)。
本页参考答案不代替个人推导与实验。

字母与shape统一查阅 [课程符号与张量形状规范](NOTATION.md)。
以下先给当前公式的速查，不要求先记完整规范。
特殊符号的中英文读法与整句朗读见该规范第7节；本页先读手算，再看第6节的性能收益。

## 1. 今天只回答一个核心问题

### 1.1 先把每个字母说清楚

| 字母 | 英文 | 这里表示什么 |
| --- | --- | --- |
| `Q` | Query | 用来查询的特征向量 |
| `K` | Key | 用来与Query匹配的特征向量 |
| `V` | Value | 按匹配权重汇集的内容向量 |
| `B` | Batch size | 一批中的序列数 |
| `H` | Query heads | 本图的头数，严格写法为 `Hq` |
| `T_q` | Query sequence length | 本轮Query位置数，即分数表的行数 |
| `T_kv` | Key/value sequence length | 可查询的K/V位置数，即分数表的列数 |
| `D_k/D_v` | Head dimensions | 每头Q/K、V各有多少个特征数 |
| `S` / `P` / `O` | Scores / Probabilities / Output | 分数、归一化权重、各头的加权输出 |

`[B,H,T,T]` 中的两个 `T` 不是一个多余的重复：
一个是“哪个Query位置在问”，一个是“与哪个Key位置比较”。
仅在本节完整Self-Attention Prefill中，二者相等，`T_q=T_kv=T`。
这里的 `T` 是 **sequence length 的惯用符号**，表示位置轴长度；
它不是 `Tensor` 的缩写，也不是某个单独 token。无 padding 时，位置数通常等于
token 数；严格含义仍是张量中这一轴的槽位数。

用一句话记忆：

```text
token 是内容单位，position 是排队位置，T 是位置数量，Tensor 是装它们的数组。
```

Attention 的分数表可以想成“提问者 × 资料卡片”：

```text
T_q  = 有多少个Query位置在提问，即分数表行数
T_kv = 有多少个Key/Value位置可查询，即分数表列数
```

这个“位置”不只适用于文字：

```text
文字：一个位置 ≈ 一个 text token
图片：一个位置 ≈ 一个 image patch
音频：一个位置 ≈ 一个 audio time frame
视频：一个位置 ≈ 一个时空 patch / tubelet
```

例如 `224×224` 图片切成 `16×16` patch，有 `14×14=196` 个图片序列元素；
8帧同样图片的视频则约有 `8×196=1568` 个视频序列元素。
多模态拼接时 `T_total` 是各模态序列长度和，再加实现需要的特殊 token。
如果文字 Query 查询图片 K/V，则可以出现 `T_q=文字长度`、`T_kv=图片patch数`。

完整 Prefill 中每个输入位置都同时扮演两种角色，所以 `T_q=T_kv=T`；
单 token Decode 中通常只有新 token 提问，历史 Cache 加新 token 提供资料，
所以 `T_q=1`，而 `T_kv` 等于历史长度加1。

### 1.2 再读完整数据流

普通 Attention 的朴素实现常先算整张 score，再 softmax，再与 V 相乘。
下面假设K/V已经按Query heads对齐；GQA的紧凑Cache头数仍是 `Hkv`：

```text
Q     [B,Hq,T_q,D_k]
K     [B,Hq,T_kv,D_k]
V     [B,Hq,T_kv,D_v]

S = (Q @ K.transpose(-2,-1)) / sqrt(D_k) + Mask
                     ↓
             [B,Hq,T_q,T_kv]
                     ↓ softmax(dim=-1)，沿Key位置归一化
P            [B,Hq,T_q,T_kv]
                     ↓ @ V，按权重汇集内容
O            [B,Hq,T_q,D_v]
```

`@` 是矩阵乘法；数学中写成 `K^T` 的转置在这里只交换最后两轴，
不是四维PyTorch张量的 `K.T`。本节不考虑dropout，Mask允许位置加0、禁止位置加负无穷。
`O` 尚未合并头与输出投影，不是词表logits或生成的文字。

因此，原来的简图可以准确地读成：

```text
匹配Q/K → 缩放并Mask → 分数表S → softmax → 权重表P → 加权V → 每头输出O
                      [B,H,T,T]          [B,H,T,T]         [B,H,T,D_v]
```

例如 `B=1,H=2,T=3`，代表1条序列、2个头，每个头各有一张3行3列的分数表，
总共18个分数。Decode时若历史已有8个位置，再输入1个token，
则 `T_q=1,T_kv=9`，分数表是 `[B,Hq,1,9]`，不能继续写成 `[B,Hq,T,T]`。

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

### 4.1 专业名称，不是英文缩写展开

`m/l/o` 是数学变量，不存在像 MLA 那样唯一规定的英文全称。
本课程按职责采用以下规范名称：

| 符号 | 建议英文名称 | 中文名称 | 形状 |
| --- | --- | --- | --- |
| `m` | Running Row Maximum | 当前行累积最大值 | 标量 |
| `l`，数学常写 `ℓ` | Running Softmax Normalizer | Softmax累积归一化因子 | 标量 |
| `o` | Unnormalized Attention Output Accumulator | 未归一化注意力输出累加器 | `[D_v]`向量 |

`l/ℓ` 读作 **ell**，不是数字1，也不是 `log` 的缩写。
`o` 读作字母 **oh**，不是数字0；它尚未归一化，不等于最终输出 `O`。
官方Triton教程中类似状态使用 `m_i/l_i/acc`，不能只按变量名猜职责。
不同版本论文也可能直接维护归一化输出；要先核对更新式。

### 4.2 这三个量到底记录了什么

针对某一个头的某一个Query，已经处理的有效Key位置集合为 `J`：

```text
m = max_{j∈J} s_j
l = ∑_{j∈J} exp(s_j - m)
o = ∑_{j∈J} exp(s_j - m) * V_j
最终输出 = o / l
```

`s_j` 是已经完成缩放、Mask的分数，不是未缩放的原始点积。
`j∈J` 读“j属于集合J”；`∑` 读“对这些j求和”；`V_j` 读“V下标j”。
V是向量时，标量权重乘它的每个分量，再分别累加。

`l` 是当前基准下的分母，不是所有历史分数的普通加和。
如需自然对数归一化量，则 `logsumexp(s)=m+ln(l)`，不是 `l` 本身。
最大值基准改变后，`l` 不保证单调增加；V可以为负，`o` 也不必为正。

### 4.3 用前面两块数字手算

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

- 初始 m=-inf、l=0、o=0。行内最终至少要有一个有效Key；如果一开始遇到全Mask块，
  还必须特殊处理或跳过该行的无效更新，否则 `-inf-(-inf)` 会产生NaN。
  不能只凭“最后总有有效Key”就认为每步更新安全。
- Mask 禁止的 score 按 -inf 处理，对指数和的贡献为0。
- m/l/o 常用 FP32 累积，输出可恢复原 dtype；浮点实现不保证逐位相同。
- 实际 Attention 的 o 是每个 query/head 一行向量，不只是这里的标量。
- 分块仍计算 dense score，只是不同时保存完整 T×T，主算术复杂度并未变成线性。
- 每个新Query需要独立的m/l/o；这些是本轮计算状态，不像KV Cache那样直接供下个Query复用。

## 6. 这样计算，对FlashAttention到底有什么好处

### 6.1 先看最终需要什么，而不是中间算过什么

对于一行Attention，最终只需一个 `[D_v]` 向量：

```text
输出 = ∑ exp(s_j-m)*V_j / ∑ exp(s_j-m)
```

因此可以“读一块，汇总一块”，不必为最终输出长期保留每个Key的独立概率。
`m`保证数值稳定，`l`保证跨所有块统一归一化，`o`保留输出所需的加权内容。
三者一起使**分块的全局softmax加权汇集**可行，而不是给每块同等权重。

这里没有保存足以恢复所有历史概率的信息；若任务要求返回完整Attention map，
就需要额外存储或重新计算。也不是把历史KV压成m/l/o：换一个Query，匹配分数就会变化。

### 6.2 朴素实现与融合分块实现的搬运路径

```text
朴素、分开执行的示意：
Q/K → 矩阵乘法 → 完整S写入全局内存
完整S读回 → softmax → 完整P写入全局内存
完整P和V读回 → 矩阵乘法 → O写入全局内存

融合分块的示意：
Q块 + K/V块
    ↓ 加载并复用片上数据
局部分数 → 更新m/l/o → 继续下一块K/V
    ↓
完成归一化 → 写出O，以及实现所需的辅助信息
```

这是解释数据流，不规定所有版本相同的循环顺序、加载次数或写回策略。
FlashAttention各版本会调整并行划分与状态保存方式；
核心是不把完整 `T_q*T_kv` 的S/P数组物化到全局内存。

“片上”是GPU的寄存器、共享内存等有限资源；“全局内存”容量较大，
访问成本和带宽特征与片上存储不同。论文把其测试硬件的全局存储称为HBM，
不要据此把所有GPU、包括Spark，都称为同一种物理显存。
Q/K/V和最终输出仍然需要存放、加载，不能把本图理解成零显存读写。

### 6.3 三种收益要分开说

| 收益 | 为什么成立 | 不代表什么 |
| --- | --- | --- |
| 减少大中间数组 | 无须在全局内存保存完整S/P | 不代表权重、KV、输入输出消失 |
| 减少中间数据IO | 局部分数用于更新后可被覆盖；避免完整S/P的反复写回和读回 | 不代表Q/K/V只读一次或所有场景固定加速 |
| 更适合融合计算 | 在GPU kernel内部结合分块矩阵乘法、归一化和加权汇集 | 不代表Python循环本身更快 |

真实速度还取决于矩阵乘法效率、tile形状、寄存器压力、共享内存、
并行度、dtype及输入规模；速度需要独立Benchmark。这里不把论文加速比当本机结果。

### 6.4 内存数字：一张大表与一行累积状态

假设 `B=1,Hq=16,T_q=T_kv=T`，只计算单层一个FP16分数数组：

```text
S_bytes = B * Hq * T_q * T_kv * 2
```

| T | 单张完整S的理论大小 |
| --- | ---: |
| 4096 | 512 MiB |
| 8192 | 2048 MiB，即2 GiB |
| 16384 | 8192 MiB，即8 GiB |

若P也单独物化且同为FP16，它的大小与S相同；
但不能一概相加就宣称进程峰值，因为dtype、内存复用和实现不同。

当 `D_v=128`，一行的m/l/o只包含：

```text
1个m + 1个l + 128个o分量 = 130个数
若全为FP32：130*4 = 520 bytes
```

这个**单行累积状态**不随 `T_kv` 增长，但不是完整工作内存：
还需Q/K/V块、局部分数、输出及其他临时资源。若同时处理多个Query行，
每行都有自己的状态；全部输出仍随 `B*Hq*T_q*D_v` 增长。

例如选 `BLOCK_M=64,BLOCK_N=128`：
局部分数 `[64,128]` 若用FP32，理论数组为32 KiB；
64行m/l/o的理论FP32状态为32.5 KiB。
这只是逻辑元素计数，不是实际每个SM的分配量、寄存器数或可运行tile承诺。
不能拿“整层512 MiB”和“单行520 bytes”相除就称为整模型节省倍数。

## 7. 收益的边界：哪些事情没有改变

### 7.1 没有把稠密Attention变成线性计算

普通完整Prefill的Q/K匹配和对V的加权汇集，主矩阵运算量仍约为：

```text
O(B * Hq * T_q * T_kv * (D_k + D_v))
```

这是大O复杂度，字母O不是前文的输出张量。
`T_q=T_kv=T` 时，对序列长度仍是二次关系。
因果实现可跳过不合法的块，减少常数工作，但不改变稠密问题的该渐近阶。
减少数据搬运、允许适当重算，可能比单纯减少FLOPs更重要。

### 7.2 没有取消KV Cache，也不承诺Decode固定提速

KV Cache保存历史请求状态，FlashAttention减少本轮中间结果的存储与IO。
GQA减少独立KV heads，MLA改变KV表示；这些优化对象不同，可以组合。
单token Decode通常是 `[B,Hq,1,T_kv]`，本来就不是完整方形分数表；
此时历史KV读取、batch及并行划分很重要，不能照搬长Prefill的收益比例。

### 7.3 “Exact”不等于浮点逐位一致

这里是相同的稠密Attention数学定义，不是按低秩或稀疏模式丢弃匹配关系。
浮点加法顺序、dtype和kernel变化仍可能带来误差，按合理容差对齐。
完整softmax通常也会减去最大值；**减最大值本身不是FlashAttention独有的加速秘诀**。

### 7.4 训练仍要处理反向传播

真实训练实现需保存必要的输入、输出或归一化辅助状态，
并在backward按块重算所需局部概率，避免长期保存完整S/P。
这不等于零激活内存，也不是每次反向都重跑整个模型。
本页CPU教学实验只验证forward，不覆盖生产backward、dropout、全Mask行策略。

### 7.5 数学参考不等于融合GPU kernel

本仓库 `online_softmax_chunks` 最后仍返回完整概率向量，适合理解归一化；
它不是证明FlashAttention内存收益的实现。
`blocked_attention` 不物化完整S/P，但使用多个PyTorch操作与Python循环：
即使放到CUDA上，中间张量也不自动留在单一kernel的片上存储中。
其教学实现还要求K/V同形状、Q/K/V每头维度相同，不支持通用GQA接口。

PyTorch SDPA可能选择不同后端；调用成功不代表本次一定用了FlashAttention。
实际融合GPU实验与计时在Day 20–21，本页不提供实测加速比。

## 8. 在 Jupyter 怎么学

先打开 `learning/day06_online_softmax.ipynb`：

1. 先预测普通答案和错误平均答案。
2. 逐块观察 m/l/o，核对换基准公式。
3. 改块宽为1、2、3，检查结果是否仍对齐。
4. 将 score 增大到1000附近，观察稳定性。
5. 运行分块 Attention 对照 SDPA。
6. 用二维Value向量验证o不是单个数，并检查 `m+ln(l)` 与logsumexp对齐。
7. 不分配大张量，只用算术验证512 MiB等估算，分清单层、单行与tile的统计范围。
8. 先自己回答自测，再查看Notebook里的参考答案。

再打开 `learning/day06_course.ipynb`，看统一实验和 JSON 报告。
`learning/online_attention.py` 是可读的 PyTorch 参考，不能因“不写整张 score”
就宣称有生产 FlashAttention 的 GPU 性能。真正融合 GPU 实验在 Day 20–21。

## 9. 面试主问与追问

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

**追问四**：m/l/o的英文名称是什么？
答：本课程规范名是Running Row Maximum、Running Softmax Normalizer、
Unnormalized Attention Output Accumulator。它们是变量角色名，不是官方缩写展开。

**追问五**：o只有一个数吗？下一个Query能否直接复用？
答：通常o为 `[D_v]` 向量。不同Query分数不同，应分别维护状态；
不能把m/l/o当成历史KV Cache。

**追问六**：分块后平均丢了什么？
答：不同块在全局分母中的相对权重；先把每块归一化为总权重1，会错误地抹掉差异。

**追问七**：为什么旧l和旧o必须一起乘alpha？
答：二者都使用旧最大值作为指数基准；只有一起换到新基准，才能保持同一个加权比值。

**追问八**：是否一定更快？如何证明？
答：不一定。先验证同一输入的数值，再单独测warmup后的GPU时间与峰值内存，
确认实际后端、dtype、Mask、shape一致；不能用CPU循环或公式估算作为加速证据。

**今日验收**：不用背公式也能解释 `exp(1-2)×exp(2-4)=exp(1-4)`，
并说清分母 l 和分子 o 为什么必须一起乘 alpha；画出完整S/P不再写回全局内存的路径，
说明单行状态、KV Cache与整个进程内存不是一回事。

收尾记录见 [Day 6学习记录](DAY06_STUDY_LOG.md)；用户已要求继续
[Day 7：请求生命周期与调度地图](DAY07_SCHEDULER_MAP.md)。
Day 6关注一次Attention内部怎样分块汇集，Day 7关注多条请求怎样等待、运行和回收。
进入下一天不自动勾选本页独立验收。

## 10. 一手参考

- [FlashAttention原论文：分块、IO分析与反向重算](https://arxiv.org/html/2205.14135v2)
- [Triton官方Fused Attention：m_i/l_i/acc更新及最终归一化](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [PyTorch 2.10 SDPA：后端选择与数值差异](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)

本页的数值和数组大小为独立教学推导；引用的生产实现不等于本仓库已验证所有对应能力。
