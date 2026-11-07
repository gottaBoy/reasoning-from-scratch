# 课程符号与张量形状规范

适用范围：本仓库30天课程的公式、图示、Notebook与实验报告。
这是课程内的统一约定，不是所有论文、框架都使用相同字母的行业标准。
阅读外部资料时，先查看其 notation / shape legend，再按本文映射。

## 1. 先看这张速查表

**先直接记住两个名称：**

- `E`：**Embedding Dimension**，嵌入维度；在本课程模型中也是
  **Model Dimension / Residual-Stream Width**，即残差主路径宽度。
- `F`：**Feed-Forward Intermediate Dimension**，前馈网络中间维度；
  其中 **FFN = Feed-Forward Network，前馈神经网络**。

这些字母是本文选择的数学符号，不是必须逐字母展开的官方缩写。
原始Transformer论文分别用 `d_model`、`d_ff` 表示主干和FFN内部维度；
外部实现的Embedding宽度也不一定与主干宽度相同，仍需检查中间是否有投影。

| 符号 | 英文全称 / 惯用名称 | 中文含义 | 本仓库字段 / 常见别名 | 备注 |
| --- | --- | --- | --- | --- |
| `B` | Batch Size | 批大小，即序列数量 | `batch_size` | 不是每条序列的token数 |
| `H` | Number of Attention Heads | 注意力头数的泛称 | `num_heads` | 本课程优先用Hq/Hkv明确是哪种头 |
| `Hq` | Number of Query Heads | Query头数 | `n_heads` / `num_attention_heads` | q对应Query |
| `Hkv` | Number of Key/Value Heads | K和V各自的头数 | `n_kv_groups` / `num_key_value_heads` | kv对应Key/Value，不是两者头数之和 |
| `T_q` | Query Sequence Length | 本轮Query位置数 | `query_length` | 单token Decode通常为1 |
| `T_kv` | Key/Value Sequence Length | 本轮K/V位置数 | `kv_length` | Cache路径包含历史与当前输入 |
| `T` | Sequence Length | 序列位置轴长度 | `seq_len` / `num_tokens` | 不是Tensor；字段含义仍需结合padding等检查 |
| `D_k` | Query/Key Head Dimension | 每头Q/K特征维度 | `head_dim` / `d_k` | Q与K点积的维度必须匹配 |
| `D_v` | Value Head Dimension | 每头V特征维度 | `v_head_dim` / `d_v` | 不必总等于D_k |
| `D` | Head Dimension | 每头特征维度的简写 | `head_dim` / `d_head` | 本课程普通示例令D_k=D_v=D；不保证D=E/Hq |
| `E` | Embedding Dimension / Model Dimension / Residual-Stream Width | 本课程模型的嵌入及主干宽度 | `emb_dim` / `hidden_size` / `d_model` | 这里hidden_size不是本仓库hidden_dim |
| `F` | Feed-Forward Intermediate Dimension | FFN中间特征宽度 | `hidden_dim` / `intermediate_size` / `d_ff` | 不是头数，不规定必须为3E或4E |
| `L` | Number of Transformer Layers | Transformer Block层数 | `n_layers` / `num_hidden_layers` | 大写L，不是softmax分母ell |
| `V_vocab` | Vocabulary Size | 词表大小 | `vocab_size` | vocab是vocabulary的简写；不是Value向量 |
| `s_bytes` | Bytes per Element | 每元素存储字节数 | `tensor.element_size()` | 不同数组需分别确认dtype |

表中的代码名是常见映射，不保证所有项目同名同义。Q/K/V及S/P/O的英文名称见第2节，
m/l/o见第4节，希腊字母与发音见第7节。

对齐本仓库 Qwen3 0.6B 配置时：

```text
E=1024, L=28, Hq=16, Hkv=8, D_k=D_v=D=128, F=3072, V_vocab=151936
```

配置来源为 `reasoning_from_scratch/qwen3.py` 的 `QWEN_CONFIG_06_B`，
不是所有 Qwen3 版本的统一配置。`D` 不一定等于 `E/Hq`，应先查实现。

**最简记忆：B看几条序列，H看几个头，T看几个位置，D看每头几个数。**

### `E、H、D、F` 放在同一张宽度图里

本仓库的Block依次经过Attention和FFN，每个子层各有一次残差相加。
分叉图更适合看清“计算支路与原样保留的主干”。下面保留这种表示，
但不把Attention和FFN画成接收同一个X、最终只相加一次的并行模块：

```text
X [B,T,E]
    │
    ├────────────────────────────────────┐
    ↓                                    │
  RMSNorm                                │
    ↓                                    │
  Q/K/V线性投影 → 拆成各自的头             │ 保留X
    ↓                                    │
  Attention匹配与加权汇集                  │
    ↓                                    │
  合并头 [B,T,Hq*D_v]                     │
    ↓ 输出投影 Hq*D_v → E                │
  A [B,T,E]                              │
    ↓                                    │
    + ←──────────────────────────────────┘
    ↓
Y = X + A   [B,T,E]          第一次残差相加
    │
    ├────────────────────────────────────┐
    ↓                                    │
  RMSNorm                                │
    ├─ gate_proj：E → F → SiLU ─┐         │
    └─ up_proj：  E → F ───────┤         │ 保留Y
                               ↓         │
                          逐元素相乘      │
                               ↓         │
                         down_proj：F → E│
                               ↓         │
                          U [B,T,E]      │
                               ↓         │
                               + ←───────┘
                               ↓
Z = Y + U   [B,T,E]          第二次残差相加
```

Q/K/V投影是并行来源，FFN里的gate/up也是并行来源；
**Attention与FFN在这个Block中则是先后执行的。**
残差相加要求本层两边的shape一致，不要求Attention内部的 `Hq*D_v=E`。

对普通 MHA/GQA，最重要的形状关系是：

```text
输入主干：             [B,T,E]
Q 投影宽度：           Hq × D_k
K 投影紧凑宽度：       Hkv × D_k
V 投影紧凑宽度：       Hkv × D_v
Attention 合并头宽度： Hq × D_v
输出投影：             Hq×D_v → E

FFN gate/up：          E → F
FFN down：             F → E
```

因此：

- `E` 决定残差主路径及子层接回主路径时的接口宽度。
- `Hq/Hkv` 决定分成多少个 Query 头、多少组独立 K/V。
- `D_k/D_v` 决定每个头内部有多少个特征。
- `F` 只决定 FFN 中间“加工车间”的宽度，不是头数，也不是序列长度。
- `T` 只决定一次处理多少个位置；通常不改变这些权重矩阵的 shape。

**不是所有关系都能反推：**

```text
Q width = Hq × D_k                 # 普通多头投影的形状关系
K width = Hkv × D_k
V width = Hkv × D_v
FFN shape = E → F → E              # 这里所讨论FFN的宽度关系

D_k = E / Hq                       # 只有实现明确这样规定时才成立
F = 3E                             # 某些模型的配置选择，不是定律
```

本仓库 Qwen3 0.6B 的具体代入：

```text
E=1024, Hq=16, Hkv=8, D_k=D_v=D=128, F=3072

Q：1024 → 16×128 = 2048 → 拆成16个头
K：1024 →  8×128 = 1024 → 8个KV头
V：1024 →  8×128 = 1024 → 8个KV头
Attention合并：16×128 = 2048 → 输出投影 → 1024

FFN：1024 → 3072（gate/up两条分支）
     3072 → 1024（down）
```

注意这里 `E/Hq=1024/16=64`，但显式配置的 `D=128`。
这正是“每头维度不能盲目用 `E/Hq` 推导”的可执行例子。

### `D`为什么不一定等于`E/Hq`：分的是投影结果，不是原始输入

先省略Batch和位置，每个位置输入为一个E维向量。计算有两步：

```text
第1步：Linear，有权重，会重新计算特征，允许改变宽度
        x [E] @ W_Q.T [E,Hq*D_k] → q_flat [Hq*D_k]

第2步：view/reshape，无权重，只把结果重新分组，不改变元素数
        q_flat [Hq*D_k] → Hq个头，每个D_k维
```

真正可以从“拆头”推出的等式是：

```text
D_k = Q投影输出宽度 / Hq
```

只有另外规定 **Q投影输出宽度=E**，才能得到 `D_k=E/Hq`。
不是先拿E个数平分，再要求模型只能这样设计。
每个头可用自己的投影权重读取整个E维输入，并非只能看到输入中的 `E/Hq` 个数。

先看能手算的例子：`E=4,Hq=2,D_k=3`，人为选一组教学权重：

```text
x = [a,b,c,d] = [1,2,3,4]                  4个输入数
    ↓ Linear(4,6)，W_Q形状[6,4]
q_flat = [a+b,b+c,c+d,a-c,b-d,a+b+c+d]
       = [3,5,7,-2,-2,10]                  6个投影结果
    ↓ 只重新分组，不再计算
头0 = [3,5,7]，头1 = [-2,-2,10]            2个头，每头3个数
```

这里 `D_k=6/2=3`，而不是 `E/Hq=4/2=2`。
新增输出是输入的不同线性组合，不是复制补零，也不是凭空增加独立信息。
同样地，不能用reshape直接把4个输入数变成6个数。

| 比较项 | Q宽度保持E的设计 | 本仓库Qwen3 0.6B配置 |
| --- | --- | --- |
| E / Hq | 1024 / 16 | 1024 / 16 |
| Q投影 | Linear(1024,1024) | Linear(1024,2048) |
| 拆头 | 16个头，每头64维 | 16个头，每头128维 |
| D | 64，等于E/Hq | 128，不等于E/Hq |

二者都是维度上可行的设计，不是在比较模型质量。
本仓库 `GroupedQueryAttention` 只有在 `head_dim=None` 时才由 `E/Hq`
推导每头宽；显式传入 `head_dim=128` 就按它建立投影矩阵。
权重shape改变后不能无条件加载同一checkpoint。
可在 [Day 5配置Notebook](../learning/day05_config_memory.ipynb) 的新增第0节亲手验证。
拆头后的完整计算继续看 [两头共享KV的手算](ATTENTION_HAND_CALC.md)，
从 `[0.75,0.25]` 的权重一直算到输出投影和残差相加。

参数量也能从这张图读出（忽略 bias）：

```text
Q：E × (HqD_k)
K：E × (HkvD_k)
V：E × (HkvD_v)
输出投影：(HqD_v) × E
SwiGLU：3 × E × F
```

所以 `E/H/D/F` 是宽度和分工，`T` 是位置数量，`L` 是 Block 层数；
它们控制不同轴，不能互相替代。

这里的 `T` 通常来自 **sequence length** 的惯用记号。
它表示“这一轴有多少个序列位置”，不是 `Tensor`，也不是 token 的英文首字母。
对于经过 tokenizer 的文本，若没有 padding、packing 或额外位置，一个 token
通常对应一个位置，因此可以把 `T` 直观理解为 token 数。
更严格地说，`T` 数的是张量中的位置槽位；padding 位置也会占槽位，
而 packed/变长批处理可能不再直接用一个矩形 `T` 表示所有有效 token。

### 用三层关系记住 `T`

不要把下面三个概念混成一个：

```text
token    = tokenizer 产生的离散编号，例如 [我] [喜欢] [AI]
position = 序列中的槽位，例如第0、第1、第2个位置
tensor   = 装这些位置表示的多维数组，例如 X
```

例如一句话被 tokenizer 切成 4 个 token：

```text
token：    [我] [喜欢] [学习] [AI]
position：   0    1     2     3
T = 4

X.shape = [B, T, E] = [2, 4, 1024]
```

这里 `T=4` 表示每条序列有 4 个位置；`X` 才是 Tensor。
如果第4个位置是 padding，`T` 仍然是4，只是该位置通常由 padding mask 排除。
所以“`T` 等于 token 数”是无 padding 时的直观近似，不是符号本身的定义。

有些论文会用 `L`、`S` 或 `N` 表示序列长度；符号是作者的局部约定。
本课程选 `T` 便于把 Query/Key 的两个位置轴写成 `T_q/T_kv`。

### 用四种模态理解 `T`

在多模态模型中，`T` 不一定表示“文字 token 数”。
更通用的说法是：

> `T` = 当前序列中有多少个可参与计算的序列元素。

| 模态 | 一个序列元素是什么 | `T` 通常怎么得到 | 典型形状 |
| --- | --- | --- | --- |
| 文字 Text | tokenizer 产生的文字 token | token 数 | `[B,T_text,E]` |
| 图片 Image | 图像 patch / visual token | `patch_rows × patch_cols`，可能再加 CLS 等特殊 token | `[B,T_image,E]` |
| 音频 Audio | 时间窗口或声学帧 | 时间帧数，通常还经过下采样 | `[B,T_audio,E]` |
| 视频 Video | 某一帧的空间 patch，或时空 tubelet | `frame_count × patches_per_frame`，取决于实现 | `[B,T_video,E]` |

#### 文字：一个 token 对应一个位置

```text
文本：       我   喜欢   学习   AI
token位置：   0    1     2    3
T_text = 4
```

在这种最熟悉的场景中，可以近似记成“`T` 就是 token 数”。

#### 图片：先切 patch，再排成序列

假设图片大小为 `224×224`，patch 大小为 `16×16`：

```text
横向 patch 数 = 224 / 16 = 14
纵向 patch 数 = 224 / 16 = 14
T_image = 14 × 14 = 196
```

图片原本是二维网格，进入 Transformer 前通常会被展平成序列：

```text
二维图片 patch 网格 [14,14]
          ↓ flatten
图片序列      [patch0, patch1, ..., patch195]
          ↓ projection
X_image     [B,196,E]
```

`196` 不是图片像素总数，也不是文字 token；它是 patch 数。
如果实现额外加入一个 CLS 或 image summary token，长度可能变成 `197`。

#### 音频：一个时间帧对应一个位置

音频通常先切成连续的时间窗口，再提取每个窗口的特征：

```text
音频波形 → 声学帧0 → 声学帧1 → 声学帧2 → ...
                         T_audio = 帧数
```

例如 16 秒音频、10 ms 的帧移，未考虑边界和下采样时约有：

```text
T_audio ≈ 16 / 0.01 = 1600
```

实际模型可能合并相邻帧，因此进入 Transformer 的 `T_audio` 可能更小。
这里的一个“audio token”通常是一个时间帧特征，不是一个可读的单词。

#### 视频：时间 × 空间一起形成序列

假设视频取 8 帧，每帧 `224×224`，每个空间 patch 为 `16×16`：

```text
每帧 patch 数 = 14 × 14 = 196
T_video = 8 × 196 = 1568
```

可以把它看成：

```text
第0帧的196个patch
第1帧的196个patch
...
第7帧的196个patch
                ↓ flatten
视频序列长度 T_video = 1568
```

如果模型使用 tubelet（一次合并连续多帧），或者做空间/时间下采样，
实际长度会不同。视频的“位置”至少包含时间位置和空间位置，不能只把它
当成普通文字 token 的一维位置。

#### 多模态：各自编码后再拼接

例如一段文字有 `T_text=10`，一张图片有 `T_image=196`：

```text
文字序列： [text0 ... text9]       长度10
图片序列： [image0 ... image195]   长度196

拼接后：
T_total = T_text + T_image = 206
X_total.shape = [B,206,E]
```

模型还需要模态类型 embedding、位置编码或特殊分隔 token，
因此真实长度可能是 `206 + special_tokens`。
如果是 Cross-Attention，也可以不拼接：

```text
文字 Query：T_q = 10
图片 K/V：  T_kv = 196
score.shape = [B,Hq,10,196]
```

这正好说明 `T_q` 和 `T_kv` 为什么要分开写：
Query 可以来自文字，Key/Value 可以来自图片、音频或视频。

注意图片中的 `H` 常表示 image height，视频中的 `W` 可能表示 width；
而 Attention 里的 `Hq` 表示 Query heads。多模态文档应优先写
`H_img/W_img`、`Hq/Hkv`，不要只写一个裸的 `H`。

## 2. 张量名称和运算符

### 2.1 字母是“数组”，不只是英文缩写

| 符号 | 英文 | 在 Attention 中的职责 |
| --- | --- | --- |
| `X` | Hidden states | 当前层输入的隐藏表示 |
| `Q` | Query | 当前各位置用来查询的特征向量 |
| `K` | Key | 各可查询位置用来与 Q 匹配的特征向量 |
| `V` | Value | 各可查询位置被加权汇集的内容向量 |
| `W_Q/W_K/W_V` | Projection weights | 生成 Q/K/V 的模型参数，不是缓存状态 |
| `M` | Additive attention mask | 加到分数上的 Mask；允许为0，禁止为负无穷 |
| `S` | Attention scores | 缩放并加 Mask 后、softmax 前的分数 |
| `P` | Attention probabilities | 对 Key 位置轴做 softmax 后的注意力权重 |
| `O` | Per-head attention output | 各头对 V 加权汇集后的输出，尚未合并头及做输出投影 |

Q/K/V 都是数值向量，不是单词本身，也不是三个额外输入句子。
Self-Attention 中通常从同一组隐藏表示投影得到；Cross-Attention 的来源可以不同。

本规范区分 `S` 和 `P`：旧文或代码中的 `scores` 要结合该行运算判断，
不能因为变量名没改，就把 softmax 前的任意分数当成概率。
这里不考虑 dropout；至少有一个有效 Key 时，`P` 每行非负且沿 Key 轴和为1。
不是每列和为1，也不是所有头的权重加起来为1。

### 2.2 运算符不要靠猜

| 写法 | 含义 | Python / PyTorch 对应 |
| --- | --- | --- |
| `[B,Hq,T_q,D_k]` | 张量的形状，各项是轴长度 | 不是张量中实际存放的四个数 |
| `QK^T` | Q 与转置后的 K 做矩阵乘法 | `Q @ K.transpose(-2, -1)`，假设头已对齐 |
| `A @ B` | 矩阵乘法；高维时按规则作批量矩阵乘法 | Attention 沿最后两轴运算 |
| `A * B` | 代码中的逐元素乘法，允许广播 | 不是矩阵乘法 |
| `A ⊙ B` | 公式中的逐元素乘法 | 对应 `A * B` |
| `sqrt(D_k)` | 每头 Q/K 维度的平方根 | `math.sqrt(D_k)` |
| `softmax(..., dim=-1)` | 沿最后一轴归一化 | 本文分数的最后一轴是 Key 位置 |
| `sum` / `Σ` | 按指定索引求和 | 必须注明对哪一轴求和 |

旧图中的 `×` 需要按语境读：在 `Q × K^T` 中是矩阵乘法，
在内存公式 `B × Hq × T_q × T_kv` 中是标量相乘。
新代码示例优先用 `@` 和 `*` 明确区分。

**转置上标与长度字母是两回事。**
数学的 `K^T` 表示每个头内 `[T_kv,D_k]` 矩阵的转置。
对四维 PyTorch 张量，不要照抄 `K.T`：它不是只交换最后两轴。
应使用 `K.transpose(-2, -1)` 或 `K.mT`。

## 3. 一次 Attention 的完整形状

### 3.1 普通 MHA/GQA，先写清紧凑状态

采用本课程常用轴顺序：`[batch, heads, positions, features]`。
只写普通投影与注意力的逻辑形状；MLA 的特殊表示另见第6节。

```text
X          [B, T_q, E]
Q          [B, Hq,  T_q,  D_k]
K_compact  [B, Hkv, T_kv, D_k]
V_compact  [B, Hkv, T_kv, D_v]
```

使用 Cache 时，`K_compact/V_compact` 包含历史及本轮追加的状态，
不一定只来自本轮的 `X`。
`T_kv` 是这个数组的Key轴长度，不表示每个Query都能读取其中所有位置；
因果Mask、padding或其他Mask仍会限制每一行的可见范围。

- MHA：`Hq=Hkv`。
- MQA：`Hkv=1`。
- 分组 GQA：多个 Q heads 共享一组 K/V；本课程实现要求 `Hq % Hkv == 0`。

计算时，每个 Q head 必须读到它对应的 KV head。
教学代码可以把 K/V 沿头轴重复到 `Hq`；优化实现也可以直接做分组读取，
并不要求真的复制大数组。**紧凑 Cache 的头数仍是 `Hkv`，不是 `Hq`。**

### 3.2 对齐头之后，逐步推导

下面的 `K/V` 指按 Q heads 对齐后的逻辑视图，不代表底层一定物化：

```text
Q        [B, Hq, T_q,  D_k]
K        [B, Hq, T_kv, D_k]
K转置    [B, Hq, D_k,  T_kv]

Q @ K转置
  [T_q,D_k] @ [D_k,T_kv] → [T_q,T_kv]
                ↑
          对这条特征轴求和

S = (Q @ K转置) / sqrt(D_k) + M
S        [B, Hq, T_q, T_kv]
P = softmax(S, dim=-1)
P        [B, Hq, T_q, T_kv]

O = P @ V
  [T_q,T_kv] @ [T_kv,D_v] → [T_q,D_v]
                ↑
          对所有可用Key位置加权求和
O        [B, Hq, T_q, D_v]
```

`M` 只需能广播到分数形状，不要求真的分配四维数组。
布尔 Mask 在不同 API 中可能有相反含义，不能将本文加法 Mask 的约定直接
替换成任意框架的 `True/False`。

用元素索引读同一个过程，`b/h/i/j/d` 分别是序列、Query头、Query位置、
Key位置、Q/K特征的索引。省略 Batch 和头索引后：

```text
S_ij = (Σ_d Q_id K_jd) / sqrt(D_k) + M_ij
P_ij = exp(S_ij) / Σ_j' exp(S_ij')
O_iv = Σ_j P_ij V_jv
```

这里 `v` 是 V 特征索引，不是词表大小。实际稳定 softmax 会先减去行最大值。
`O` 合并头后是 `[B,T_q,Hq*D_v]`，再经输出投影回 `[B,T_q,E]`。
LM Head 最终产生的词表 logits 才是 `[B,T_q,V_vocab]`；
不要把它与 Attention 的分数 `S` 混为一谈。

### 3.3 为什么旧图写两个 T

从空 Cache 开始的完整 Self-Attention prefill，通常有：

```text
T_q = T_kv = T
H = Hq                      # 旧图的简写
S/P 的 shape = [B,H,T,T]
```

两个 `T` 数值相同、职责不同：

```text
行：哪个Query位置在查询？
列：它在与哪个Key位置匹配？
```

例如 `B=1,H=2,T=3`，每个头都有一张3行3列的匹配表。
`S[0,1,2,0]` 表示“第0条序列、第1个头，第2个Query位置对第0个Key位置的分数”。
这是从0开始的索引；整个张量共 `1*2*3*3=18` 个分数。
因果 Mask 禁止读取未来位置，但朴素矩形张量仍可以包含这些被 Mask 的元素。

所以在 `[B,H,T,T]` 中：

```text
第一个 T：Query 位置数，等价于 T_q
第二个 T：Key/Value 位置数，等价于 T_kv
```

完整 Self-Attention Prefill 中两者都来自同一段输入，所以数值相同；
这不表示两个轴“都是 Tensor”，也不表示模型有两份 token。

可以把 Attention 矩阵想成“提问者 × 资料卡片”的表：

```text
每一行：一个 Query 位置在提问
每一列：一个 Key 位置可被查询
每个格子：这一行对这一列的匹配分数
```

因此：

```text
T_q  = 提问者有多少个位置
T_kv = 资料卡片有多少个可查询位置
```

完整 Prefill 中，每个输入位置既是提问者也是资料卡片，所以两者都为 `T`。
单 token Decode 中，只有新 token 是提问者，但历史 Cache 加新 token 都是资料卡片，
因此常见 `T_q=1`、`T_kv=历史长度+1`。

### 3.4 Prefill 和 Decode 不总是方阵

以本课程 Qwen3 配置、单条序列为例，逻辑张量如下：

| 阶段 | Q | 紧凑K（本例V同形状） | 分数S/概率P |
| --- | --- | --- | --- |
| 空Cache，8 token Prefill | `[1,16,8,128]` | `[1,8,8,128]` | `[1,16,8,8]` |
| 已存8位置，再输入1 token | `[1,16,1,128]` | `[1,8,9,128]` | `[1,16,1,9]` |

Decode 的新 token 对应 `T_q=1`，但需查询9个位置，`T_kv=9`。
分块 Prefill、多 token Decode、Cross-Attention 也可能有 `T_q != T_kv`。
使用 Cache 时，因果 Mask 必须考虑 Query 的实际位置偏移，不能不加判断地
对非方阵套用一个从左上角开始的三角 Mask。

## 4. Online Softmax 的局部符号

以下针对一个 Query head 的一个 Query 位置，并以已经处理的有效 Key 为范围。
本课程将 `o` 定义为**未归一化分子**；其他资料可能将同名变量定义为归一化输出。
它们是数学变量，不是有唯一英文展开的缩写；以下英文名用于规范描述职责。

| 符号 | 规范英文名称 | 含义与形状 |
| --- | --- | --- |
| `s` | Block Scores | 当前分块的分数向量 `[block_keys]`；不是 `s_bytes` |
| `m` | Running Row Maximum | 到目前为止的最大分数，标量 |
| `l` / `ℓ` | Running Softmax Normalizer | 以 `m` 为基准的指数和，即分母；标量 |
| `o` | Unnormalized Attention Output Accumulator | 同一基准的指数加权V和，即分子；`[D_v]` |
| `alpha` / `α` | Rescaling Factor | 旧状态换到新基准的系数 `exp(m_old-m_new)`，标量 |
| `p` | Unnormalized Block Weights | 当前块的指数权重 `exp(s-m_new)`；不是完整概率P |
| `output` | Normalized Attention Output | `o/l`，对应O的一行 |

所以 `m/l/o` 是每行的累积状态；`O` 才是各行各头组成的输出张量。
`l`读ell，不是log；自然对数归一化量为 `logsumexp(s)=m+ln(l)`。
官方Triton教程里可对照循环内的 `m_i/l_i/acc`；其指数底数和最终辅助状态的保存
方式需结合代码读取，不能把变量名不加转换地套到本页自然指数公式。
具体换基准手算见 [Day 6讲义](DAY06_ONLINE_SOFTMAX.md)。

## 5. 读源码时的字段与符号冲突

| 外部名称或写法 | 本课程对应 | 易混点 |
| --- | --- | --- |
| `hidden_size` / `emb_dim` | `E` | 本仓库的 `hidden_dim` 反而是FFN宽 `F` |
| `num_attention_heads` / `n_heads` | `Hq` | 不是每头维度 |
| `num_key_value_heads` / 本仓库 `n_kv_groups` | `Hkv` | 某些库的 `num_key_value_groups` 是 `Hq/Hkv` |
| `intermediate_size` / 本仓库 `hidden_dim` | `F` | 不是残差宽度 |
| `head_dim` | 普通示例中为 `D` | 不存在通用的 `D=E/Hq` 保证 |
| PyTorch SDPA文档 `N` | `B` | 它的Batch符号不是本文的头数 |
| PyTorch SDPA文档 `L` / `S` | `T_q` / `T_kv` | 不是本文层数 `L` / 分数张量 `S` |
| PyTorch SDPA文档 `E` / `Ev` | `D_k` / `D_v` | 不是本文残差宽度 `E` |
| PyTorch SDPA文档 `Hq` / `H` | `Hq` / `Hkv` | 它的 `H` 是KV头数，旧课程简写 `H` 是Q头数 |

这里按 PyTorch 2.10 的 SDPA shape legend 核对，与课程现有环境一致。
**专业规范的重点不是某个字母永远只能有一种含义，而是每篇文档明确声明、
局部不歧义，并能对照实际张量。**

报告中建议同时写：

```text
模块：普通GQA，层0
阶段：Cache已有8位置，本轮输入1位置，追加后可用9位置
轴顺序：[batch, heads, positions, features]
配置：B=1,Hq=16,Hkv=8,T_q=1,T_kv=9,D_k=D_v=128
数组：K_compact=[1,8,9,128]，dtype=bfloat16，device=cuda
```

不同实现也会用 `[B,T,H,D]`。轴顺序改变时先声明，不把排列不同误判为模型不同。
变长Batch的矩形 `T` 可能是padding后长度；服务框架也可能打包为总token轴。
因此 `B` 不一定直接等于线上全部活跃请求数，数组元素数也不直接等于有效token数。

## 6. MoE、MLA、内存和Kernel的补充约定

| 符号 | 含义 | 范围或限制 |
| --- | --- | --- |
| `N_tokens` | 参与当前路由的token数 | 等长且无padding时可为 `B*T_q` |
| `N_experts` | routed experts数量 | 不包含另计的shared experts |
| `k_top` | 每个token选择的routed experts数 | 无丢弃/容量padding时，逻辑dispatch slots为 `N_tokens*k_top` |
| `r_kv` | 联合KV latent宽度 | 不是KV头数；旧实验参数名为 `rank` |
| `D_rope` | 独立位置Key宽度 | 指本课程讲解的解耦RoPE MLA路径 |
| `theta` / `rope_base` | RoPE频率基数 | 不是张量宽度，也不是上下文长度 |
| `BLOCK_M/BLOCK_N` | Kernel中常用的Query/Key tile大小 | 是局部分块长度，不是Batch大小；最终以该kernel定义为准 |

普通、各层相同、K/V同dtype的紧凑Cache数组字节数：

```text
L * B * T_kv * Hkv * (D_k + D_v) * s_bytes
```

当 `D_k=D_v=D` 时，才简写成 `2*L*B*T_kv*Hkv*D*s_bytes`。
单层朴素分数数组字节数为 `B*Hq*T_q*T_kv*s_bytes`，这里 `s_bytes`
必须取分数数组自己的dtype，不能默认与Cache相同。
这两者不是同一种内存，以上也不是完整进程峰值。

本课程介绍的紧凑MLA路径，每层每位置保存联合latent和独立位置Key，
逻辑形状为 `[B,T_kv,r_kv]` 和 `[B,T_kv,D_rope]`；
不能再套普通Cache的 `2*Hkv*D`。详见 [MLA讲义](MLA_INTUITION.md)。

## 7. 特殊符号怎么读

以下采用中文技术讨论与英语数学交流中的常用读法，不是现代希腊语发音教学。
中文音译只是认字辅助，不等于准确英语发音；不同英语口音允许有差异。
不要求先背整张表，遇到符号时按“名称、读法、局部含义”三步查。

### 7.1 希腊字母：名字不决定数学意义

| 符号 | 英文名称 / 读音提示 | 中文常见读法 | 本课程常见语境 |
| --- | --- | --- | --- |
| `α` | alpha，AL-fuh | 阿尔法 | Online Softmax的重缩放系数 |
| `β` | beta，BAY-tuh；也有BEE-tuh | 贝塔 | LayerNorm偏置等参数，按实现定义 |
| `γ` | gamma，GAM-uh | 伽马 | Norm的可学习缩放 |
| `δ` / `Δ` | delta，DEL-tuh；大写称capital delta | 德尔塔 / 大写德尔塔 | 小变化量 / 差值，按上下文定义 |
| `ε` / `ϵ` | epsilon，EP-si-lon等 | 艾普西龙 | Norm分母中的小正数，代码常写eps |
| `θ` / `ϑ` | theta，THEE-tuh或THAY-tuh | 西塔 | RoPE频率参数，或另行定义的模型参数 |
| `λ` | lambda，LAM-duh | 拉姆达 | 权重、正则项系数等，需局部定义 |
| `μ` | mu，MYOO或MOO | 缪 | 均值等统计量，需局部定义 |
| `σ` | sigma，SIG-muh | 西格马 | 标准差，或被定义为某个函数 |
| `Σ` | capital sigma | 大写西格马 | 希腊大写字母；旧课程也用它简写求和 |
| `π` | pi，PIE | 派 | RoPE等旋转公式中的圆周率 |
| `τ` | tau，TOW，类似cow韵尾 | 陶 | 温度等参数，需局部定义 |

`ε/ϵ`、`θ/ϑ`分别是同一字母的常见字形变体，不自动代表不同数学对象。
`Σ`是字母，`∑`是求和运算符；本课程旧文中的 `Σexp(...)` 按“对exp项求和”理解。
不要将 `α` 当作拉丁字母a，也不要将 `θ` 当作长度字母T。

### 7.2 拉丁字母、下标和上标

| 写法 | 中文读法 | 英文读法示例 | 注意事项 |
| --- | --- | --- | --- |
| `m` | 字母m，艾姆 | em | 不是其变量职责的缩写展开 |
| `l` / `ℓ` | 字母l，埃尔 | ell | ℓ是l的数学字形，不是希腊字母或数字1 |
| `o` / `O` | 小写o / 大写O，欧 | lowercase oh / capital oh | 不是数字0；本课程二者职责不同 |
| `Q/K/V` | 读对应英文字母，也可说Query/Key/Value | cue / kay / vee | 先区分字母名和张量含义 |
| `s_j` | s下标j | s sub j / s subscript j | 指某个位置的分数 |
| `T_q` | T下标q | T sub q | 指Query长度，不是T乘q |
| `m_old/m_new` | 旧m / 新m | m old / m new | old/new是标签，不是额外乘数 |
| `K^T` / `Kᵀ` | K的转置 | K transpose | 这里上标T不是长度，也不是幂 |
| `x^2` / `x²` | x的平方 | x squared | 此处上标2才表示幂 |
| `m'` | m撇 | m prime | 旧递推式表示更新值，不自动表示导数 |
| `x∈R^n` | x属于n维实向量空间 | x is in R to the n | R常排成 `ℝ`；n是维度 |

### 7.3 运算符、关系符和函数

| 符号或函数 | 中文读法 | 英文读法 | 说明 |
| --- | --- | --- | --- |
| `∑_j a_j` | 对j求和a下标j | sum over j of a sub j | 累加指定范围 |
| `∑_{j=1}^n` | j从1到n求和 | sum from j equals one to n | 上下限给出索引范围 |
| `⊙` | 逐元素乘 / 哈达玛积 | Hadamard product / elementwise product | 与矩阵乘法不同 |
| `@` | 矩阵乘 | matrix multiply / matmul | Python运算符名也可读at |
| `×` / `·` | 乘 / 点乘，按上下文 | times / dot | 必须看对象，不能统一当逐元素乘 |
| `/` | 除以 | divided by / over | `o/l`读o除以l |
| `√x` / `sqrt(x)` | x的平方根 | square root of x | Attention缩放中常见 |
| `exp(x)` / `e^x` | e的x次方 / 指数函数 | exponential of x / e to the x | 不是x的平方 |
| `ln(x)` | x的自然对数 | natural log of x | 本页logsumexp采用自然对数 |
| `max(...)` | 最大值 | maximum of ... | row maximum为行最大值 |
| `≈` | 约等于 | approximately equal to | 不是严格相等 |
| `≠` | 不等于 | not equal to | 与赋值区别 |
| `≤` / `≥` | 小于等于 / 大于等于 | less/greater than or equal to | 常见于Mask条件 |
| `∝` | 正比于 | proportional to | KV内存随上下文增长的讨论中常见 |
| `∈` / `∉` | 属于 / 不属于 | is in / is not in | 集合成员关系 |
| `∞` / `-∞` | 无穷大 / 负无穷 | infinity / negative infinity | 代码常写inf / -inf |
| `→` | 映射到、得到或趋于，按上下文 | maps to / yields / tends to | 数据流图中通常读“得到” |
| `:=` | 定义为 / 赋值为，按上下文 | is defined as / is assigned | 不能与恒等证明混读 |
| `∂L/∂x` | L对x的偏导数 | partial derivative of L with respect to x | 此处L若为loss，必须另行声明，不是层数 |
| `∇_x f` | f对x的梯度 | gradient of f with respect to x | ∇符号名为nabla，也称del |
| `‖x‖₂` | x的二范数 | L two norm of x / Euclidean norm of x | 不是绝对值或逐元素平方 |
| `O(T²)` | T平方阶 | order T squared / big O of T squared | 大O复杂度，不是输出张量O |

`exp2(x)`是2的x次方，不是 `exp(x²)`。
官方kernel可能通过调整score尺度使用以2为底的指数，不能原样混用自然指数的基准状态。
数学排版中的 `x^2` 在Python代码中应写 `x**2`；
Python的 `^` 是按位异或，不是乘方。

### 7.4 将Online Softmax公式完整读出来

```text
m_new = max(m_old, max(s))
读：新m，等于旧m和当前块最大分数中的较大值。

α = exp(m_old - m_new)
读：阿尔法，等于e的“旧m减新m”次方。
英文：alpha equals the exponential of m old minus m new.

ℓ_new = α * ℓ_old + ∑_j exp(s_j - m_new)
读：新ell，等于阿尔法乘旧ell，加上对j求和的e的“s下标j减新m”次方。
英文：ell new equals alpha times ell old, plus the sum over j
      of exp of the quantity s sub j minus m new.

o_new = α * o_old + ∑_j exp(s_j - m_new) * V_j
读：新o，等于阿尔法乘旧o，加上各位置指数权重乘对应V向量的总和。

O_row = o / ℓ
读：这一行最终输出，等于o除以ell。
```

其中j只遍历当前块，旧状态已经包含之前的块。
先把整句话读通，再回答“哪个是基准、哪个是分母、哪个是分子”，不要只背字母音。

## 8. 写作与验收要求

1. 首次出现公式时，链接本规范，并在附近说明本节特有符号。
2. 首次写shape时注明轴顺序；压缩前后、拆头前后、合并头前后分别命名。
3. 区分维度长度与元素索引，例如 `Hq` 是头数，`h` 是某个头的索引。
4. 不省略会影响结论的条件，例如完整Prefill、单tokenDecode、是否有Cache。
5. 计内存时给出对象、层数范围、dtype和单位；`MiB=2^20 bytes`，`GiB=2^30 bytes`。
6. 不承诺所有外部库同名同义；引入其他符号时写映射，不无声换义。

## 一手参考

- [PyTorch 2.10 SDPA：参数、参考运算与Shape legend](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Transformer原论文：3.1节d_model、3.3节d_ff与3.4节Embedding](https://arxiv.org/html/1706.03762v7)
- [PyTorch 2.10 Tensor.T与Tensor.mT定义](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_tensor_docs.py)
- [PyTorch 2.10 Linear：输入输出宽度与权重形状](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Linear.html)
- [PyTorch 2.10 Tensor.view：重排不改变元素数](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.view.html)
- [Triton官方Fused Attention：累积状态、exp2与归一化](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Valdosta State University：希腊字母与常见英语读音](https://barnbaum.valdosta.edu/personal/teaching_MISC/greek.htm)
- [Cambridge Dictionary：alpha英语发音与音频示例](https://dictionary.cambridge.org/pronunciation/english/alpha)
- [本仓库Qwen3配置及实现](../reasoning_from_scratch/qwen3.py)

本文选用自己的维度符号；API行为按以上版本核对，具体模型以实际配置与实现为准。
