# P1-02：把匹配、Mask、softmax、汇集写成你自己的Attention

上一关：[RMSNorm/RoPE](MY_OPS_IMPLEMENTATION.md)。这里进入对应原Day 9的本地实战。
本关只写计算核心，**Q/K/V已经给定**；不创建Linear投影、RoPE或持久Cache。
不要一次加入五个模块，让第一次报错不知道来自哪一步。
**统一查看 [my_ops注释模板](../learning/my_ops_template.py)，只在
`learning/my_ops.py` 中实现**。变量、手算、位置规则、A1–A5和验收命令
集中在模板的 `causal_attention` 部分；模板保持未完成状态，方便复习。
不再单独建立Attention模板或实现文件。
更直观的入口是 [统一模板的中文图解版](../learning/my_ops_template_zh.py)：
把Q/K/V解释为问题、资料索引和资料内容，从17.5手算串起A1–A5，
并解释每次矩阵乘法消去哪条轴；只需任选一版读，实现仍放在同一个 `my_ops.py`。

## 1. 先手算一个输出17.5

只看最后一个Query，它能读取两个位置：

```text
Q = [0, sqrt(2)*ln(3)]       D_k=2
K位置0=[1,0]，K位置1=[0,1]
V位置0=[10]，V位置1=[20]     D_v=1

Q @ K转置 / sqrt(2) = [0,ln(3)]
softmax = [1/4,3/4]
加权V = 1/4*10 + 3/4*20 = 17.5
```

这里D_k与D_v可以不同；头输出最后宽度来自V，而不是Key位置数。
这不是文字语义测试，只是数值算例。

## 2. 五步写完核心

参照 `learning/my_ops_template.py` 的Attention部分，在已有的
`learning/my_ops.py` 中新增并完成以下接口：

```python
def causal_attention(q, k, v, start=0):
    ...
```

模板已提供参数校验和A1–A5注释，作为你实现时的提示。
仅在自己的 `my_ops.py` 中将待填写部分变成实际计算与return；
不要用整个模板覆盖该文件，也不要重新定义或删除上一关已完成的RMSNorm/RoPE。

| 步骤 | 你实现什么 | shape |
| --- | --- | --- |
| A1 | Q/K矩阵乘，除以sqrt(D_k) | `[B,1,T_q,T_kv]` |
| A2 | 构造Query绝对位置与Key位置 | `[T_q]`、`[T_kv]` |
| A3 | key位置大于query位置为禁止，填负无穷 | 广播到分数 |
| A4 | 沿Key轴softmax | `[B,1,T_q,T_kv]` |
| A5 | 权重矩阵乘V | `[B,1,T_q,D_v]` |

本关FP32、同device、单头，输入非空。
负start、Query范围超出Key长度、Q/K维度不匹配、K/V位置数不同都应抛ValueError。
保留autograd，不能在函数内部detach或进入no_grad。

<a id="attention-key-transpose"></a>

### 2.1 A1：k.transpose(-2, -1)表示什么？

`transpose`交换两条轴；负数表示从最后一轴向前数：

| 参数 | 在当前k中指哪条轴 | 含义 |
| --- | --- | --- |
| `-1` | `D_k`，最后一轴 | 一份Key有多少个匹配特征 |
| `-2` | `T_kv`，倒数第二轴 | 有多少份Key/资料 |

所以不是交换最后两个元素，而是：

```text
原来 k： [B,1,T_kv,D_k]
转置后： [B,1,D_k,T_kv]
```

**原来一份资料占一行，转置后一份资料占一列**。Batch/head轴不变。
先省略这两个前导轴，看3份资料、每份2个特征：

```text
原来 K：[3,2]                 K.transpose(-2,-1)：[2,3]

         特征0  特征1                  资料0  资料1  资料2
资料0      1      2           特征0      1      3      5
资料1      3      4           特征1      2      4      6
资料2      5      6
```

这样一个Query就能同时给各份Key打分：

```text
Q：[问题数,特征数] @ K转置：[特征数,资料数]
                     -> [问题数,资料数]

                 [1  3  5]
[1  2]     @     [2  4  6]     = [5  11  17]

对资料0：1*1 + 2*2 = 5
对资料1：1*3 + 2*4 = 11
对资料2：1*5 + 2*6 = 17
```

可在Jupyter里单独观察，不需要完成整份Attention：

```python
import torch

q = torch.tensor([[[[1., 2.]]]])                       # [1,1,1,2]
k = torch.tensor([[[[1., 2.], [3., 4.], [5., 6.]]]])  # [1,1,3,2]
k_t = k.transpose(-2, -1)                             # [1,1,2,3]
raw_scores = q @ k_t                                 # [1,1,1,3]
print(k_t[0, 0])              # [[1,3,5], [2,4,6]]
print(raw_scores[0, 0])       # [[5,11,17]]
```

这里是**未缩放、未应用Mask的点积分数**，不是最终Attention输出。
完整A1仍要除以`sqrt(D_k)`，随后才做Mask、softmax和加权V。
`transpose`自身不修改数值或原来的`k`，也不会替你完成点积；
不要用`reshape`代替交换行列。公式中的`K.T`是二维数学简写，
当前四维代码使用`k.transpose(-2,-1)`，只交换最后两条轴。

<a id="attention-mask-broadcasting"></a>

### 2.2 多维张量中的 Mask 是怎样广播的？

把当前 Attention 的分数看成四维索引：

```text
scores[b, h, query_i, key_j]
       │  │       │       └─ 第几个可查询的 Key/资料
       │  │       └──────── 第几个 Query/问题
       │  └──────────────── 第几个 Attention head
       └─────────────────── 第几个 batch/序列
```

当前练习固定单头，所以实际是：

```text
scores.shape = [B, 1, T_q, T_kv]
```

因果规则只取决于“Query位置”和“Key位置”，与 batch/head 的编号无关。因此：

```text
scores ：[B, 1, T_q, T_kv]
blocked：[   T_q, T_kv]
```

PyTorch 广播时从右向左对齐：

```text
scores ：[B, 1, T_q, T_kv]
blocked：[   1,    T_q, T_kv]  # 缺少的前导轴按1理解
```

于是同一张 `[T_q,T_kv]` 的“未来位置地图”会套到每个 batch/head。
这通常是逻辑上的扩展，不需要你手动复制成 `[B,1,T_q,T_kv]`。
如果想显式写出来，等价于：

```python
blocked_4d = blocked[None, None, :, :]  # [1,1,T_q,T_kv]
scores = scores.masked_fill(blocked_4d, -torch.inf)
```

例如 `B=2、H=2、T_q=3、T_kv=7`：

```text
scores.shape = [2,2,3,7]
```

这表示有 `2×2=4` 张“3行问题 × 7列资料”的分数表。
`scores[0,1,2,5]` 表示第0条序列、第1个head、第2个Query、
第5个Key的分数；它对应的因果规则仍然只查 `blocked[2,5]`。

只有当规则因 batch 不同（例如不同 padding）时，mask 才需要写成
`[B,1,T_q,T_kv]`；如果每个 head 也不同，才需要 `[B,H,T_q,T_kv]`。
记忆口诀：**最后两轴是“问题×资料”的核心表，前面的轴是这张表的批次/head层。**

<a id="attention-modalities"></a>

### 2.3 用文本、图片、音频、视频理解同一套 Attention

Attention 不关心输入最初是文字还是媒体；它看到的是一串特征向量。
变化的只是“一个位置/一个 token”具体代表什么。

| 模态 | 一个序列单位 | `T`/`N`代表什么 | `Q/K/V`直觉 | 因果 Mask |
| --- | --- | --- | --- | --- |
| 文本 | token | token 数 | 当前 token 查找上下文 token | 自回归语言模型通常遮挡未来 token |
| 图片 | patch 小方块 | patch 数，例如 `14×14=196` | 当前 patch 查找相关区域 patch | 普通图像理解常双向；自回归图像生成可遮挡未来 patch |
| 音频 | 时间帧 | 帧数，不是采样点总数 | 当前帧查找相关历史声学帧 | 流式语音通常只读过去帧 |
| 视频 | 时空 patch/tubelet | 帧数×每帧 patch 数 | 当前时空块查找其他时间/空间块 | 可遮挡未来帧；空间是否遮挡由模型设计决定 |

#### 文本：token 排成一列

```text
“猫  坐  在  垫子  上”
 0   1   2   3    4
```

位置 2 的 Query 可以看位置 0、1、2，不能看 3、4。
这里 `T=5`，如果有 8 条文本就是 `B=8`；不同 head 可以分别关注语法、指代或位置关系。

#### 图片：把图像切成 patch

```text
[patch0][patch1][patch2]...
```

一张图切成 `14×14` 个 patch 后，`N=196`，单层 Attention 的核心 map 可以是：

```text
[B,H,196,196]
```

最后两轴仍然是“当前 patch × 可查询 patch”。普通图像理解通常允许 patch 互相查看，
因此不一定使用文本式的上三角因果 Mask；不能看到“patch”这个词就自动套文本规则。

#### 音频：声音切成时间帧

```text
| frame0 | frame1 | frame2 | frame3 | ...
```

`T`是时间帧数，不是采样点总数。流式语音中，frame2 不能读取还没到达的 frame3，
所以 `start`、因果 Mask 和 KV Cache 都很像文本 Decode：新帧查询历史帧。

#### 视频：时间帧里再切空间 patch

```text
第0帧: [patch0][patch1]...
第1帧: [patch0][patch1]...
```

若视频有 `F` 帧，每帧有 `P_h×P_w` 个 patch：

```text
N = F × P_h × P_w
Attention map = [B,H,N_q,N_kv]
```

这里用 `F` 表示视频帧数、`N`表示时空 token 数，避免把公式中的 `T`
误解成 Tensor 或只理解成“视频帧数”。时间因果 Mask 可以遮挡未来帧，
但空间方向是否遮挡取决于模型设计。

四种模态的统一记忆法：

```text
文本 token、图片 patch、音频 frame、视频时空 patch
                         ↓
                 都变成特征向量
                         ↓
最后两轴永远是：Query单位 × Key/Value单位
```

所以 `Value` 不是原始文字、像素、声音或视频文件，而是模型内部的内容特征；
“查资料”只是帮助理解 Q/K/V 和 Mask 的比喻。

## 3. start为什么不能忽略

完整3位置Prefill：

```text
start=0，T_q=T_kv=3
允许：[1,0,0]
      [1,1,0]
      [1,1,1]
```

历史已有4个位置，本次给3个新Query时：

```text
start=4，T_q=3，T_kv=7
Query绝对位置=[4,5,6]
Key位置=[0,1,2,3,4,5,6]
每行分别允许前5/6/7个Key
```

这里只模拟“已经给定历史K/V”的矩形匹配，不实现Cache保存/追加。
未来分数不能填0，因为exp(0)=1；应该在softmax前让禁止位置不贡献权重。
本模板bool True表示禁止，与SDPA的bool True表示允许的语义不要混用。

## 4. 本地验收命令

现有Jupyter Terminal中，从自己的文件运行：

```bash
cd /workspace
python -m learning.local_practice --task attention --device cpu --candidate learning/my_ops.py
python -m learning.local_practice --task attention --device cuda --candidate learning/my_ops.py
```

两个关卡使用同一实现文件：`--task norm-rope`只要求RMSNorm与rope，
`--task attention`只要求causal_attention。Attention尚未实现不会影响上一关验收；
若尚未定义该函数，Attention任务会明确报告缺少接口，不回退到参考答案。
未传candidate时，仅运行 `reference_baseline`；用户candidate未完成时不能记成个人通过。
每轮使用新报告名，默认自动生成到 `logs/practice/`，旧结果不覆盖。
从宿主机运行要先把新文件有选择地同步进 `/workspace/learning/`；
Jupyter卷不等于宿主机checkout，上一关首次报“candidate不存在”就是两处文件未同步。

14项检查：

```text
17.5手算、完整Prefill、T=1、单Query历史匹配、多Query历史匹配
未来K/V扰动不影响前缀、零Q在有效Key上均匀汇集、输入不原地修改
Q/K/V反向梯度有限并对齐SDPA
负start、Query越界、K/V长度不符、Q/K维度不符、空Key拒绝
```

独立数值oracle用CPU FP64逐Query截取合法Key，不复用你的Mask代码。
这些是本关支持范围的验收，不是全部Attention变体、低精度、训练质量或性能证明。

## 5. 反复练习与故障注入

1. 先手写17.5与三角可见表，再填A1–A5。
2. 先让CPU通过，失败从第一个case定位，不先放宽容差。
3. 故意去掉sqrt(D_k)，检查手算/随机数值测试能否发现。
4. 故意softmax沿Query轴、未来填0或忽略start，观察哪些项失败，再修回。
5. 新增自己的B/T_q/T_kv/D_k/D_v组合，再明确请求CUDA。
6. 口述“哪一轴在点积后消失、哪一轴在汇集后消失”，填写实战日志。

上一关文件中的已完成TODO和return后的占位异常可在你确认后清理，
但不需要为了进入本关重写已经通过的计算。练习与参考实现分开，不自动提交你的填写结果。

## 6. 文件职责：一个模板，一份个人实现

延续P1-01的练习方式：模板集中提示，你在同一份个人文件中逐步积累实现。
不用每学一个算子就增加一组模板/实现文件，也不会误把模板当成已完成代码。
这里的“都放一起”指学习模板和个人实现各自集中，独立验收器仍保留，避免自测自证。

| 文件 | 职责 | 是否是你的实现 |
| --- | --- | --- |
| `learning/my_ops_template.py` | 统一提示；R1–R4、P1–P7、A1–A5及接口/shape/手算/命令 | 否，保持未完成的教学模板 |
| `learning/my_ops_template_zh.py` | 同一模板的中文图解版；比喻、逐步手算、广播与常见误区 | 否，接口/校验/TODO与英文版一致 |
| `learning/my_ops.py` | 个人实现；保留RMSNorm/RoPE，新增causal_attention | 是，所有已支持任务共用该candidate |
| `docs/PRACTICE_P1_02_ATTENTION.md` | 解释公式、shape、Mask、`start`、失败模式和验收边界 | 否，学习与复盘文档 |
| `learning/practice02_attention.ipynb` | 把手算、Mask可视化和验收命令放进Jupyter，便于逐格观察 | 否，可选观察工具 |
| `learning/practice_attention.py` | 独立oracle、负例和梯度验收；不应被改成“放宽测试”的版本 | 否，验收器/参考基线 |
| `learning/local_practice.py` | 根据 `--task attention` 加载candidate并生成报告 | 否，统一入口 |

因此，本关的实现动作是：**看统一模板的A1–A5，在
`learning/my_ops.py` 新增causal_attention，先CPU后CUDA**。
参考baseline通过只说明验收器和环境可用；candidate通过才记录为你这一关的实测结果。
模板中的 `NotImplementedError` 故意保留；验收指向自己的文件，
不能通过删除测试或直接调用参考实现来绕过练习。

## 7. 阅读与边界

- [单头Attention详细课](DAY09_SINGLE_ATTENTION.md)
- [Day 5汇集到输出投影的手算](ATTENTION_HAND_CALC.md)
- [PyTorch 2.10 SDPA：参考公式、shape与Mask语义](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [本关验收器与参考基线](../learning/practice_attention.py)

P1-02完成后再进P1-03多头拆合，不提前加入GQA、FlashAttention性能或服务调度。
