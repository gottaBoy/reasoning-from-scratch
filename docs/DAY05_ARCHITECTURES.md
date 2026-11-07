# Day 5：Llama、Qwen、DeepSeek、MoE 和 MLA

Day 5 的目标是建立“配置到数据流”的阅读能力，不是今天复刻三个生产模型。
先区分：模型名称是家族，具体 checkpoint/版本才决定真实配置；比较时必须记录
模型版本、参数 dtype、实现仓库和是否做过量化。
全文的 `E/L/Hq/Hkv/D/F`、配置名称对应及跨框架符号差异统一见
[课程符号与张量形状规范](NOTATION.md)。

## 1. 先用同一张表比较

| 观察项 | Llama 类稠密模型 | Qwen3 0.6B（本仓库） | DeepSeek-V2/V3 类 MoE/MLA |
| --- | --- | --- | --- |
| 主路径 | 每个 token 经过同样的 FFN | 每个 token 经过同样的 FFN | Attention 之外，token 路由到部分 experts |
| Query/KV | 常见 MHA 或 GQA，查具体 config | `16` Query heads、`8` KV groups | MLA 用压缩 latent 表示，不能只按普通 GQA 解释 |
| FFN | 常见 SwiGLU 变体 | `1024 → 3072 → 1024` | MoE：router 选择 top-k experts；共享 expert/FFN 细节按版本查看 |
| 位置编码 | 常见 RoPE | RoPE，`rope_base=1_000_000` | MLA 中位置编码与压缩/解压数据流一起分析 |
| KV 状态 | 通常保存每层 K/V | 本仓库保存紧凑 KV groups | MLA 目标之一是减少 KV 传输/缓存压力 |
| 今天的任务 | 读 config 和 Attention/FFN | 对照已有实现和形状 | 画教学级 Router/latent KV 图，不写生产 kernel |

不要从“Qwen 使用 GQA”推断所有 Qwen 都使用同一个 head 数，也不要从“DeepSeek
使用 MLA”推断所有版本的 KV 张量都能直接套用本仓库的
`[B, n_kv_groups, T, head_dim]`。

## 2. Day 5 的阅读顺序

### 2.1 Llama/Qwen：先看稠密 Transformer 骨架

用本仓库 Qwen3 作为可执行参照：

```text
Embedding
→ [RMSNorm → Attention → Residual → RMSNorm → SwiGLU FFN → Residual] × layers
→ Final Norm
→ LM Head
```

### 2.1.1 先澄清：两套名称，不是除法

原清单里的 `hidden_size / emb_dim` 表示“外部配置名称 / 本仓库名称”，不是求商。
为避免与公式里的除号混淆，统一改成对应箭头：

```text
hidden_size          ↔ emb_dim
num_hidden_layers    ↔ n_layers
num_attention_heads  ↔ n_heads
num_key_value_heads  ↔ n_kv_groups
intermediate_size    ↔ hidden_dim
head_dim             ↔ head_dim
rope_theta           ↔ rope_base
```

特别容易混淆的是两个带 hidden 的名字：

```text
hidden_size = 1024       # 外部配置：主路径的宽度
hidden_dim  = 3072       # 本仓库：FFN 内部的宽度
```

**在这里 `hidden_size` 对应 `emb_dim`，不是 `hidden_dim`。**
这是不同代码的命名约定，不是所有项目的统一规则。

### 2.1.2 先记“宽、层、头、组、每头、加工宽、位置”七件事

把一个 token 看成表格的一行。Block 不断改写这一行，进入不同计算分支时，
这一行临时变成不同宽度：

| 统一符号 | 英文名称 | 直觉问题 | 本仓库字段 | Qwen3 0.6B 示例 |
| --- | --- | --- | --- | --- |
| `E` | Embedding / Model Dimension | 进入/离开 Block，每行有多少个数？ | `emb_dim` | 1024 |
| `L` | Number of Transformer Layers | 这样的 Block 串联多少次？ | `n_layers` | 28 |
| `Hq` | Number of Query Heads | Q 分成多少个查询头？ | `n_heads` | 16 |
| `Hkv` | Number of Key/Value Heads | K 和 V 各有多少个独立头？ | `n_kv_groups` | 8 |
| `D` | Head Dimension | 每个头包含多少个数？ | `head_dim` | 128 |
| `F` | Feed-Forward Intermediate Dimension | FFN 中间分支展开到多宽？ | `hidden_dim` | 3072 |
| `θ` | RoPE Frequency Base | RoPE 各对特征转多快的频率参数？ | `rope_base` | 1_000_000 |

FFN全称为Feed-Forward Network（前馈神经网络）；E/F在论文中也常写作
`d_model/d_ff`。这些是局部符号约定，不要求每个字母都有唯一的英文缩写展开。

这里把 Q/K/V head dimension 相同作为本实现的前提。其他模型的 Q/K/V 维度
可能不同，尤其 MLA 需要另外分析。

不要先背英文名字，先回答这七个中文问题。
口诀：**主路看 E，深度看 L，Q 头 Hq，KV 头 Hkv，每头 D，FFN 看 F，位置看 θ。**

把它们放在同一条宽度流中：

```text
主干 x：      [B,T,E]
                │
   Attention：  E → Q:[Hq×D_k]，K:[Hkv×D_k]，V:[Hkv×D_v]
                │
   合并头：     Hq×D_v → 输出投影 → E
                │
   FFN：        E → F（gate/up两条）→ 相乘 → E（down）
```

这里的关系分三类：

```text
必然由张量拼接得到：
Q width = Hq×D_k
K width = Hkv×D_k
V width = Hkv×D_v
Attention输出投影输入 = Hq×D_v

实现接口要求：
主干输入输出必须回到 E，才能做 residual 相加
FFN 必须 E → F → E，才能接回主干

模型设计选择：
D 是否等于 E/Hq
F 取多大（本配置恰好 F=3E）
```

用 Qwen3 0.6B 代入：

```text
E=1024，Hq=16，Hkv=8，D=128，F=3072

Q：1024 → 16×128=2048 → 16个Query heads
K/V：1024 → 8×128=1024 → 8个KV heads
Attention合并：16×128=2048 → 输出投影 → 1024
FFN：1024 → 3072 → 1024
```

所以 `E` 是“主干道路宽度”，`H` 是“分几条车道”，`D` 是“每条车道宽度”，
`F` 是“FFN 临时扩建的加工厂宽度”。`T` 只表示有多少个位置经过这条道路，
不会直接改变这些投影权重的 shape。

### 2.1.3 把这些数字放进同一张图

只观察一个 token，先省略 Batch 和时间轴：

```text
主路径 x：1024个数（E）
    |
    | Attention 分支
    |-- Q 投影 → 2048个数 → 分成16个头，每头128个数
    |-- K 投影 → 1024个数 → 分成 8个头，每头128个数
    |-- V 投影 → 1024个数 → 分成 8个头，每头128个数
    |                 |
    |              加权汇集
    |                 |
    |         16个头 × 128 = 2048个数
    |                 |
    |            输出投影
    |                 v
    |             1024个数 → 与主路径相加
    |
更新后的主路径：1024个数（E）
    |
    | FFN 分支
    |-- gate 投影 → 3072个数（F）→ SiLU --|
    |-- up   投影 → 3072个数（F）--------|→ 逐元素相乘
    |                                            |
    |                                        down 投影
    |                                            v
    |                                        1024个数 → 与主路径相加
    v
输出仍然是1024个数（E），交给下一个 Block
```

上述 Block 串联 **28 次（L）**。RoPE 在 Attention 中旋转 Q/K 的成对特征，
由位置和 θ 决定相位，不把 128 维变成 1_000_000 维，也不旋转 1_000_000 度。
θ 不是最大上下文长度；本配置的上下文长度另由 `context_length=40960` 指定。

### 2.1.4 三个公式从“几组，每组几个数”得到

```text
Q 投影总宽度       = Hq  × D = 16 × 128 = 2048
K 投影紧凑总宽度   = Hkv × D =  8 × 128 = 1024
V 投影紧凑总宽度   = Hkv × D =  8 × 128 = 1024
每组 KV 服务几个 Q = Hq / Hkv = 16 / 8 = 2
```

K 和 V 各自都是 1024，不是 K+V 总共 1024。
最后一条公式不表示把 head_dim 再除以 2，而是：

```text
KV0 给 Q0、Q1 用
KV1 给 Q2、Q3 用
...
KV7 给 Q14、Q15 用
```

这里共享的是同一组 K/V 表示，各 Query head 仍有自己的 Query。

### 2.1.5 为什么不能总用 `E / Hq` 算 head_dim

关键动作是 **先线性投影，再拆头**，不是直接把原始 x 平分：

```text
x 的1024维
    ↓ 可学习的 W_query 投影，允许改变宽度
Q 的2048维
    ↓ 只是将2048个数重新分组
16个头 × 每头128维
```

代码对应：

```python
self.d_out = num_heads * head_dim
self.W_query = nn.Linear(d_in, self.d_out, bias=False)
```

因此 `1024/16=64` 算术上没错，但不符合这份配置显式 `head_dim=128` 的定义。
只有目标实现确实规定 `head_dim=E/Hq` 时，才可以这样推导。
本仓库是在 `head_dim is None` 时采用这个默认值。

**分叉图可以保留，但要分清分叉的对象。** 正确关系是
`X → Attention → 第一次残差相加 → Y → FFN → 第二次残差相加 → Z`，
不是Attention和FFN同时接收X后只相加一次。
完整的两次残差分叉图见 [统一规范](NOTATION.md) 第1节。

把“投影”和“拆头”拆成两个动作，误解就容易消除：

| 动作 | 做什么 | 能否改变每位置的元素数 | 有无可学习权重 |
| --- | --- | --- | --- |
| Linear投影 | 对输入特征做加权组合 | 可以，例如1024变2048 | 有，Q权重为 `[2048,1024]` |
| view/reshape拆头 | 对已经算好的结果重新分组 | 不可以，2048只能重排成16×128等合法形状 | 无 |

不是“把原来的1024个数切成16份”，而是：

```text
输入：[B,T,1024]
  ↓ Linear(1024,2048)，计算得到2048个新特征
Q：  [B,T,2048]
  ↓ view(B,T,16,128)，数值不变，只改变分组
Q：  [B,T,16,128]
  ↓ transpose(1,2)，交换位置轴和头轴
Q：  [B,16,T,128]
```

**恒成立的是 `D=Q投影宽度/Hq`；只有Q投影宽度等于E时，才有 `D=E/Hq`。**
每个头都可以读取整个1024维输入，并非只从输入中分走64个特征。

用4个数手算，令 `E=4,Hq=2,D=3`：

```text
x = [a,b,c,d] = [1,2,3,4]
    ↓ 教学Linear(4,6)，人为定义6个线性组合
Q = [a+b,b+c,c+d,a-c,b-d,a+b+c+d] = [3,5,7,-2,-2,10]
    ↓ 拆成2个头
头0：[3,5,7]
头1：[-2,-2,10]
```

这里每头3维，`D=6/2=3`，而 `E/Hq=4/2=2`。
线性投影提供不同的特征组合，并不凭空创造更多独立信息。
对应的 [Notebook](../learning/day05_config_memory.ipynb) 可运行此手算例，
并直接比较仓库Attention显式D与默认D的投影shape。

**继续学习**：理解拆头后，读 [Attention汇集、合并、输出投影手算](ATTENTION_HAND_CALC.md)，
打开 [对应Notebook](../learning/day05_attention_handcalc.ipynb)。
它用两个Query头共享一组K/V，说明为何头输出仍可不同，以及如何从4维合并结果回到3维主干。

### 2.1.6 用实际 Shape 检查，而不是只记数字

当 `B=1,T=8`：

| 位置 | Shape | 最后一维含义 |
| --- | --- | --- |
| 主路径 | `[1,8,1024]` | E，每个 token 的内部表示 |
| Q 投影后 | `[1,8,2048]` | Hq×D |
| Q 拆头并换轴 | `[1,16,8,128]` | D，每个 Q head 的特征数 |
| K/V 拆头，扩展前 | `[1,8,8,128]` | D，每个 KV head 的特征数 |
| gate/up 各自的输出 | `[1,8,3072]` | F，FFN 中间特征 |
| FFN down 输出 | `[1,8,1024]` | E，回到残差主路径 |

K/V Shape 中两个 `8` 分别是 KV heads 和 token 数，数值相同但不是同一概念。
主路径 1024 与 K 投影宽 1024 也只是数值碰巧相同，来源分别为 E 和 Hkv×D。
验证时可把 T 改成 5，立刻看清哪条轴代表序列。

换一个示例检验自己是否真正会推导：

```text
E=48, L=3, Hq=6, Hkv=2, D=16, F=120

Q 宽度 = 96，K 宽度 = 32，V 宽度 = 32
每组 KV 服务 3 个 Q heads
FFN：48 → 两条各120的分支 → 相乘 → 48
```

如果只记住“Qwen3 是 2048 和 1024”，换例子就不会算，说明还没建立概念。

### 2.1.7 面试时如何回答与自测

**主问**：`hidden_size`、`intermediate_size`、`head_dim` 有什么区别？

**答案要点**：分别是残差主路径宽 E、FFN 内部宽 F、每个 Attention head 的宽 D；
Q 投影宽由 Hq×D 决定，不一定等于 E。不同代码的字段名称要按实现映射。

**追问**：把 `n_layers` 从 28 改成 40，会直接改变单层 Q 宽度吗？
答：不直接改变；在其他字段不变时，增加串联深度，不改变单层的 Hq×D。
但总参数、总 Cache 层数、计算量等会随层数增加。

练习时只改变一个量，并先预测：

- T 从 8 改成 5：token 轴变化，projection width 不变。
- Hkv 从 8 改成 4：K/V 投影和紧凑缓存的头轴变化，Q 宽不变；
  参数 shape 已改变，不能直接加载原 checkpoint。
- F 从 3072 改成 4096：改变 FFN 内部宽，不直接改变 Attention heads。
- θ 改变：RoPE 的频率设置改变，Q/K shape 不变。

另一个跨项目命名陷阱：某些实现将 `num_key_value_groups` 用来表示
`Hq/Hkv`（这里为 2），而本仓库 `n_kv_groups` 表示 `Hkv`（这里为 8）。
**相似名字不代表相同含义，最终检查张量和源码。**

### 2.1.8 最后再换模型家族

先按以上统一符号 E/L/Hq/Hkv/D/F/θ 写一张配置表，
再换一个具体 Llama 版本逐项核对。不要直接按品牌名记忆配置。
对普通、相同 Q/K head_dim 的 GQA，关注：

```text
Q projection width = n_heads × head_dim
K/V compact width  = n_kv_groups × head_dim
GQA sharing ratio  = n_heads / n_kv_groups
```

如果配置没有显式 `head_dim`，先检查实现是否使用
`hidden_size / n_heads`；不要自行假设。

### 2.1.9 再补一层：投影权重矩阵到底有多大

不要把“有 2048 个输出特征”理解成只有 2048 个权重。
这里的 `nn.Linear(in_features, out_features, bias=False)` 保存的权重形状为：

```text
W：[输出宽度, 输入宽度]
y = x @ W.T
```

例如 Q 投影：

```text
x：[B,T,1024]
W_query：[2048,1024]
x @ W_query.T → [B,T,2048]
```

可以理解成输出的每一个数，都由输入的 1024 个数按一组学到的系数组合得到；
输出有 2048 个，就需要 2048 组这样的系数。不是简单复制或重新切分原数字。

Qwen3 0.6B 中，每个 Block 的主要线性层为：

| 投影 | PyTorch 权重形状 `[out,in]` | 无 bias 参数量 |
| --- | --- | ---: |
| Q | `[2048,1024]` | 2,097,152 |
| K | `[1024,1024]` | 1,048,576 |
| V | `[1024,1024]` | 1,048,576 |
| Attention 输出投影 | `[1024,2048]` | 2,097,152 |
| gate | `[3072,1024]` | 3,145,728 |
| up | `[3072,1024]` | 3,145,728 |
| down | `[1024,3072]` | 3,145,728 |

普通、相同 Q/K/V head_dim 的 dense GQA，主要投影参数可记成：

```text
Attention 投影：2×E×Hq×D + 2×E×Hkv×D
SwiGLU FFN：3×E×F
```

这不含 Norm、Embedding、LM Head 等参数，不能把它直接称为模型总参数量。
FFN 的 3 来自 gate、up、down 三套权重，不是因为 F 恰好等于 3E。

### 2.1.10 最容易混的三种内存

| 内存对象 | 存的是什么 | 对 T 的关系 | 是否按请求变化 |
| --- | --- | --- | --- |
| 参数权重 | W_query、FFN 等学到的系数 | 固定模型时不随 T 改变 | 通常由请求共用 |
| KV Cache | 每层各位置算出的 K/V | 随保留的 T 线性增长 | 是，请求级状态 |
| Attention score | 本轮每个 Q 与可见 K 的分数 | full prefill 时 T² | 是，本轮临时结果 |

同样“每个数 2 字节”的前提下，假设 B=1、T=4096：

```text
普通 GQA 的全部层 KV：
2 × L × B × T × Hkv × D × 2 bytes
= 2 × 28 × 1 × 4096 × 8 × 128 × 2
= 448 MiB

单层 full-prefill score：
B × Hq × T × T × 2 bytes
= 1 × 16 × 4096 × 4096 × 2
= 512 MiB
```

第一个 2 表示 K 和 V 两份；最后的 2 才是每个元素的字节数。
MiB 用 `1024² bytes`，不要把它与十进制 MB 混写。

若 T 从 4096 翻倍至 8192，在相同前提下：

```text
权重参数：不变
KV Cache：448 → 896 MiB（2倍）
单层 score：512 → 2048 MiB（4倍）
```

这里一个是“全部层持久 Cache”，一个是“单层临时 score”，不能简单将二者都乘 L，
也不能直接相加就宣称是完整进程峰值。真实还涉及 allocator、临时 buffer、实际 dtype、
量化/分页开销。FlashAttention 的重要价值正是避免物化完整 T×T score；
并不是 KV Cache 自动消失了。

### 2.1.11 一次只改一个字段

以下假设其他显式配置不变，不自动重算 D：

| 改动 | 直接变化 | 不直接变化 |
| --- | --- | --- |
| T 翻倍 | 输入位置、KV 长度、prefill score 面积 | 线性层权重 shape |
| L 增加 | Block 数、总参数、总 KV 层数 | 单层 Q/K/V 宽度 |
| Hq 增加 | Q/out 投影、Query heads、共享比例 | 紧凑 KV heads |
| Hkv 减少 | K/V 投影、紧凑 Cache 头轴、共享比例 | Query heads |
| D 增加 | Q/K/V 宽度、Cache 每头宽度 | 残差 E（经 out 投影还原） |
| F 增加 | FFN 三套矩阵、中间特征 | Attention 的 Hq/Hkv/D |
| θ 改变 | RoPE 的频率/数值行为 | 以上线性矩阵 shape |

改了参数 shape 就不能无条件加载旧 checkpoint；
shape 没变也不代表数学行为没变，例如修改 θ。

在 Jupyter 打开 [配置与内存实验](../learning/day05_config_memory.ipynb)，
先预测再执行。工具 [config_math.py](../learning/config_math.py) 只做配置算术，
不分配完整长序列 score，也不下载/加载模型。

### 2.2 MoE：把“一个 FFN”替换成“路由后的多个 FFN”

稠密 FFN 是：

```text
x → 一个 FFN → y
```

教学级 MoE 是：

```text
                 ┌─ Expert 0 ─┐
x → Router ──────┼─ Expert 1 ─┼→ 加权合并 → y
                 └─ Expert 2 ─┘
```

Router 对每个 token 产生 expert 分数：

```text
router_logits = router(x)        # [tokens, num_experts]
scores, ids = topk(router_logits, k)
```

把你熟悉的普通 FFN 想成“一家加工站”，MoE 则是“多家加工站，加一个派单员”：

```text
一个 token 的 x
    ↓ Router：为各家加工站打分
选择 Expert 1、3（示例 top-k=2）
    ├→ Expert 1 的 FFN → 输出 y1
    └→ Expert 3 的 FFN → 输出 y3
最终输出 = w1×y1 + w3×y3
```

这个 gate 选择“调用哪些完整 FFN”；SwiGLU 的 gate 只调节“一个 FFN 内的通道”。
两者可以同时出现：MoE 中每个 expert 本身也可使用 SwiGLU。

然后只有选中的 experts 处理该 token，最后按 router 权重合并。
真实 MoE 还涉及 capacity、token dispatch、负载均衡、通信和辅助 loss；
Day 5 只画数据流和实现一个 CPU 教学版本，不把它当作生产 MoE。

重点回答：

- 为什么可以节省每个 token 的激活计算？
- 为什么总参数量可以很大，但每个 token 只激活部分参数？
- 为什么 router 负载不均会让吞吐下降？
- expert 权重和 KV Cache 是两种不同的内存问题吗？

### 2.3 MLA：先理解“压缩 latent”，再看 RoPE 放在哪里

如果这节不够直观，先读 [MLA：两个数怎样生成不同头的 K/V](MLA_INTUITION.md)，
再打开 [可执行对齐实验](../learning/day05_mla_intuition.ipynb)。
先记：**每层每个历史 token 各保存一份短 latent，多头用不同投影使用它，
再单独处理位置 Key；不是整个历史压成一条摘要。**

普通 Attention 的直觉是：

```text
x → Q, K, V
K/V 沿序列保存或传输
```

教学级 MLA 数据流先画成：

```text
x
├─ q path → query 相关表示
└─ kv path → latent cKV ─→ 解压/组合 ─→ K/V 相关表示
                         ↑
                    位置编码路径
```

MLA 的具体投影、latent rank、位置编码分解和推理时吸收矩阵必须按对应版本代码
阅读。今天只需要能解释：它不是把 `n_kv_groups` 改成另一个整数，而是改变
KV 表示、缓存和计算的数据流。

可以把普通 KV 想成“每个历史位置保留较宽的记录”，MLA 想成
“训练出更紧凑的生成表示，并让后续计算直接使用它”。但这不是通用压缩软件：

- 不是对任意现有 K/V 直接做无损压缩。
- 压缩/恢复投影是模型架构和训练的一部分。
- RoPE 的位置分支及推理时矩阵吸收决定实际缓存什么，必须看具体实现。
- MoE 主要改变“找哪些 FFN 加工”，MLA 主要改变“Attention 的 KV 怎样表示”；
  二者不是互相替代的方案。

比较 MLA 和 GQA：

| 问题 | GQA | MLA |
| --- | --- | --- |
| 主要做法 | 多个 Q head 共享较少的 K/V heads | 将 KV 相关状态压缩到 latent，再按实现解压/吸收 |
| Cache 直觉 | `[B, KV heads, T, D]` | 需要查看实现保存 latent 还是等价状态 |
| 读取重点 | head 数、`repeat_interleave`、Cache shape | latent rank、位置编码分支、prefill/decode 数据流 |
| Day 5 产出 | 配置推导 | 教学级流程图和问题清单 |

## 3. 今天的具体实验

在 [Day 5 Notebook](../learning/day05_architecture_map.ipynb) 中运行：

1. 用 Qwen3 配置推导 Q/K/V projection width 和 GQA sharing ratio。
2. 用同一个输入画出 Dense FFN 和教学级 MoE 的数据流。
3. 修改 `num_experts` 和 `top_k`，统计每个 expert 被选中的 token 数。
4. 检查 router 不均匀时，为什么最大 expert load 决定 dispatch 的上界。
5. 画出普通 KV 与 MLA latent KV 的对比草图，标注“这里尚未实现生产 MLA”。

不要用随机 MoE 的输出质量评价路由器；本次只验证形状、路由计数和边界条件。

## 4. Day 5 记录模板

```text
模型/版本：
配置文件路径：
实现仓库：
hidden_size：
n_heads：
n_kv_groups / num_key_value_heads：
head_dim：
intermediate_size：
GQA sharing ratio：
FFN 类型：
MoE/MLA：
KV 状态形状或 latent 描述：
我确认的事实：
仍需查证的问题：
```

## 5. Day 5 验收

- [ ] 能从一个 config 推导 Q/K/V 投影宽度和 GQA ratio。
- [ ] 能解释 Dense FFN 和 MoE 的数据流差异。
- [ ] 能说明 router、top-k、expert load 和 dispatch 的关系。
- [ ] 能用自己的话说明 MLA 与“简单减少 KV heads”不是一回事。
- [ ] 有一张 Llama/Qwen/DeepSeek 版本化配置差异表。
- [ ] 明确哪些内容是本仓库实测，哪些只是外部源码/论文阅读。

## 6. Day 5 自测参考答案

先独立回答，再展开这一节。以下答案区分了数学推导、当前仓库实测和外部阅读，
不能把外部 MLA 论文结论写成“本仓库已实现”。

### 1. 为什么 Qwen3 的 Q projection width 是 2048，但 compact K/V width 是 1024？

Qwen3 0.6B 的配置是：

```text
Hq = 16
Hkv = 8
D = 128
E = 1024
```

Q 需要为 16 个 Query head 各产生 128 个数：

```text
Q width = Hq × D = 16 × 128 = 2048
```

K 和 V 只需要为 8 个独立的 KV groups 各产生 128 个数：

```text
K width = Hkv × D = 8 × 128 = 1024
V width = Hkv × D = 8 × 128 = 1024
```

这不是说 K/V 比 Q“少一半信息”，而是 GQA 让每两个 Query head 共享同一组
K/V。计算 Attention 前，K/V 可以按组扩展给 Query heads 使用；Cache 保存的是
扩展前的紧凑版本。本仓库源码和 Day 5 Notebook 已实测这些 projection shape。

面试一句话：

> Q head 决定有多少种查询视角，KV group 决定保存多少份可被查询的历史表示；
> GQA 保留 Q 的数量，减少 K/V 的数量。

### 2. 32 个 token、4 个 experts、top-k=2，为什么 dispatch slots 是 64？

每个 token 选择 2 个 expert：

```text
32 tokens × 每个 token 2 个选择 = 64 个 token-expert 分配
```

这里的 64 不是新生成了 64 个 token，也不是 64 个不同 token。
同一个 token 会被复制/派发到两个 expert，各 expert 处理自己的那份输入，
最后 router 再用权重把两个结果合并回这个 token 的一个输出。

```text
token 7 → Expert 1
        → Expert 3
        ↓
      加权合并
        ↓
      token 7 的一个输出
```

如果 top-k=1，就是 32 个 dispatch slots；top-k=4，就是 128 个。
当前 Notebook 实测 `counts.sum() == tokens * top_k`。

### 3. 为什么 expert load 最大的 expert 可能决定这轮 dispatch 速度？

假设 32 个 token、top-k=2，理想平均是每个 expert 16 个 slots：

```text
平均：[16,16,16,16]
```

如果 router 偏斜成：

```text
[50, 8, 4, 2]
```

Expert 0 要处理 50 个 slots，其他 expert 很快空闲，但这一轮不能在 Expert 0
完成前结束。并行 dispatch 的完成时间往往受最重的 shard/expert、通信和容量限制。

所以 top-k 总 slots 固定为 64，不代表每个 expert 工作量相同。
router 负载均衡、capacity、token dropping/padding、跨卡 all-to-all 都会影响真实吞吐。
教学 Notebook 只统计 slots，不模拟通信或 GPU expert kernel。

面试一句话：

> MoE 的稀疏性减少平均激活计算，但负载倾斜会让最忙 expert 成为关键路径，
> 造成空闲设备等待、容量溢出或通信不均衡。

### 4. 为什么 MLA 不能只解释为“把 KV groups 改少”？

GQA 的核心改变是一个简单的 head 数关系：

```text
多个 Query heads → 共享较少的 K/V heads
Cache 仍可直观写成 [B,Hkv,T,D]
```

MLA 改的是 KV 的**表示方式和计算路径**。教学图可以先记：

```text
x → latent KV 表示 → 与位置分支组合/解压/吸收 → Attention 使用
```

因此必须继续查具体实现：

- decode 时到底缓存 latent、位置分支还是某种等价状态？
- RoPE 应用在压缩前、压缩后，还是拆成独立路径？
- 哪些投影可以吸收到 query 或 output 计算中？
- Cache 的真实 shape、dtype 和重建成本是什么？

如果只把 `Hkv` 改小，仍然是在普通 GQA 的数据流里；这不会自动得到 MLA。
当前仓库没有生产 MLA 实现，Day 5 只要求数据流图和问题清单。

### 5. 配置表哪些是源码实测，哪些来自论文阅读？

当前仓库实测/可直接验证：

```text
Qwen3 0.6B：
emb_dim=1024
n_layers=28
n_heads=16
n_kv_groups=8
head_dim=128
hidden_dim=3072
rope_base=1_000_000
Q/K/V projection shape
Cache shape、GQA 扩展、FFN 形状
```

外部模型源码或论文阅读项：

```text
某个具体 Llama checkpoint 的实际 config
DeepSeek-V2/V3 的 MLA latent rank 和完整缓存路径
DeepSeek MoE 的真实 router/capacity/通信实现
生产 vLLM 的 paged kernel、scheduler 和 prefix sharing 细节
```

“源码实测”至少要有配置路径、命令、输入 shape 和输出证据；
“论文阅读”要标出论文/版本，不能用模型家族名称替代具体版本。

## 7. 配置与内存自测参考答案

### 1. Q 投影为什么需要 `2048×1024` 个权重，而不是 2048 个？

因为线性层不是“给每个输出位置放一个数字”，而是每个输出都要看完整输入：

```text
x：[B,T,1024]
Wq：[2048,1024]
x @ Wq.T → [B,T,2048]
```

2048 是输出数量；每个输出有 1024 个输入系数，所以参数量是：

```text
2048 × 1024 = 2,097,152
```

如果有 bias，还要额外加 2048 个 bias；本仓库的 Q/K/V/out projection 使用
`bias=False`。矩阵形状是 `[out_features, in_features]`，不要把 `[2048,1024]`
读成“2048 个头”或“2048 个 token”。

### 2. 为什么 FFN 的主要参数是 `3EF`，而不是 `2EF`？

SwiGLU 有三套线性权重：

```text
gate_proj：E → F，参数 E×F
up_proj：  E → F，参数 E×F
down_proj：F → E，参数 F×E
```

矩阵参数量相加：

```text
E×F + E×F + F×E = 3EF
```

本仓库 Qwen3 的 `E=1024、F=3072`，所以每个 Block 的三个 FFN 矩阵共有：

```text
3 × 1024 × 3072 = 9,437,184
```

`gate_proj` 和 `up_proj` 是同一个输入的两条并行分支，不是共享权重；
如果是普通两层 FFN，没有独立 gate 分支，才常见 `E×F + F×E = 2EF`。
这里没有把 Attention、Norm、Embedding 或 LM Head 算进去。

### 3. T 翻倍时，为什么权重不变、Cache 两倍、prefill score 四倍？

保持模型配置和 batch/dtype 不变，把序列长度从 `T` 变成 `2T`：

**权重不变：**

```text
Wq、Wk、Wv、FFN 矩阵的 shape 只由 E/Hq/Hkv/D/F 决定
```

它们是模型参数，不是每个 token 新创建一份。

**KV Cache 两倍：**

普通 GQA 的 Cache 近似：

```text
2 × L × B × T × Hkv × D × bytes
```

其中第一个 2 是 K 和 V 两份。只有 T 变成 2T，所以 Cache 变成两倍。

**朴素 prefill score 四倍：**

```text
score shape = [B,Hq,T,T]
```

T 同时出现在 Query 轴和 Key 轴：

```text
T×T → (2T)×(2T) = 4T²
```

所以是平方增长。FlashAttention 可以避免物化完整 score，但不会自动把
exact dense attention 的主要算术复杂度变成线性。

### 4. 为什么不能把所有层 Cache 和单层 score 都再乘 28？

因为它们本来就不是同一个统计对象：

**Cache 公式已经包含了所有层：**

```text
2 × L × B × T × Hkv × D × bytes
```

其中 `L=28` 已经乘过了。不能再乘一次 28。

**score 示例只计算了一层：**

```text
B × Hq × T × T × bytes
```

如果要估算“所有层同时都物化一张 score”，理论上才会再乘 28；
但普通逐层 Transformer 通常只需要当前层的 score，前一层的临时 score
可以释放或复用。因此“单层 score”与“全部层 Cache”不能直接相加，
也不能把单层 score 乘 28 后当作真实峰值。

真实峰值还取决于：

- 是否物化完整 score/probabilities；
- kernel 的临时 buffer 和 workspace；
- allocator 保留内存；
- dtype、量化、分页 metadata；
- 是否训练以及 activation/gradient/optimizer。

## 8. Day 5 到 Day 6：先回答一个新问题

既然已经知道 score 是 `[B,Hq,T,T]`，上下文翻倍为什么这张表会变成四倍？
如果不把整张表保存下来，还能算出相同的 softmax 加权结果吗？

这就是下一步 [Day 6：Attention 显存与 Online Softmax](DAY06_ONLINE_SOFTMAX.md)。
从小数字推导开始，再运行现成的分块参考，不需要立刻写 CUDA。

参考入口：

- [Qwen3 本仓库配置与实现](../reasoning_from_scratch/qwen3.py)
- [Llama 官方模型仓库](https://github.com/meta-llama/llama-models)
- [DeepSeek-V2 论文（MLA 与 MoE）](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)
