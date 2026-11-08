# Day 8：RMSNorm 与 RoPE：先把一行信号调稳，再给特征加位置

Day 8 是第二周的第一天：开始从“读懂参考实现”转向“自己实现并验证”。
配套：[学习记录](DAY08_STUDY_LOG.md)、[Notebook](../learning/day08_rmsnorm_rope.ipynb)。

## 1. 先记两句人话

```text
RMSNorm：把每个位置这一行的整体音量调到可控范围，再按通道乘可学习的增益。
RoPE：把每个位置的特征对按位置旋转不同角度，让Q/K的匹配感知顺序。
```

它们处理的是不同问题：

| 模块 | 处理对象 | 影响什么 | 不做什么 |
| --- | --- | --- | --- |
| RMSNorm | 一个位置的特征向量 | 数值尺度/稳定性 | 不读取别的token，不改变shape |
| RoPE | Q/K每个head的特征对 | 位置信息/相对位置关系 | 不负责屏蔽未来，不改变shape |

RMSNorm全称 **Root Mean Square Layer Normalization，均方根层归一化**；
RoPE全称 **Rotary Position Embedding，旋转位置编码**。
RMS读“均方根”；gamma是希腊字母γ，eps对应epsilon/ε。
二者不需要接在一起调用：本仓库Block先做Norm，Attention投影出Q/K后才对Q/K用RoPE，
注意力之后再有另一处Norm和FFN。

## 2. RMSNorm：给每一行信号调音量

<a id="rmsnorm-derivation"></a>

### 2.1 推导目标：调节整体尺度，不是把每个数都变成1

下面推导前向计算。只看一个位置的隐藏向量：

$$
x=[x_1,x_2,\ldots,x_E]
$$

E是这一行的特征数。对 `[B,T,E]`，本课只沿最后的E轴计算，
不把其他token或Batch行一起求平均。
如果在每个Attention head上做Norm，归一化宽度可为D；最终以该模块的轴定义为准。

先忽略epsilon和可学习增益，对非零向量选择一个正的共同缩放系数c，
让输出的均方根等于1。这是本节选择的归一化目标，不是所有归一化方法的唯一目标。

### 2.2 从“平方、平均、开根号”得到RMS

先平方，避免正负数直接求和时互相抵消；再平均，最后开根号：

$$
q=\frac{1}{E}\sum_{j=1}^{E}x_j^2,\qquad
\operatorname{RMS}(x)=\sqrt{q}
$$

可以按下面的顺序读：

```text
每个分量平方 → 沿特征轴求平均 → 开平方 → 这一行的共同尺度
```

平方均值q不是方差：这里没有先减去均值。

### 2.3 为什么要除以RMS：求出缩放系数c

设归一化后的分量为 `x_hat_i=c*x_i`，要求：

$$
\frac{1}{E}\sum_{i=1}^{E}(c x_i)^2=1
$$

把共同的c平方提到求和外面：

$$
c^2\frac{1}{E}\sum_{i=1}^{E}x_i^2=1
$$

取保持方向的正解：

$$
c=\frac{1}{\sqrt{\frac{1}{E}\sum_{i=1}^{E}x_i^2}}
=\frac{1}{\operatorname{RMS}(x)}
$$

因此：

$$
\hat{x}_i=\frac{x_i}{\operatorname{RMS}(x)}
$$

直觉就是“把整行音量按同一个旋钮调整”，不是每个分量分别除以自己。

### 2.4 加上epsilon和gamma，得到实际公式

零向量的RMS为0，不能直接做上面的除法；加上正的epsilon稳定分母。
之后再给每个特征乘一个可学习增益gamma：

$$
\boxed{
y_i=\gamma_i\,
\frac{x_i}
{\sqrt{\frac{1}{E}\sum_{j=1}^{E}x_j^2+\epsilon}}
}
$$

对应向量形式：

```text
RMSNorm(x) = gamma ⊙ x / sqrt(mean(x²) + eps)
```

| 符号 | 这里的职责 | 对应代码 |
| --- | --- | --- |
| E | 当前归一化行的特征数 | `x.shape[-1]` |
| q | 沿特征轴的平方均值 | `x.square().mean(dim=-1, keepdim=True)` |
| epsilon / ε | 分母稳定项 | 本课显式`eps=1e-6` |
| gamma / γ | 每个特征的可学习增益 | 本课接口为`weight`，shape `[E]` |
| ⊙ | 逐元素相乘 | `normalized * weight`，按特征广播 |

加入eps后，即使gamma全为1，也不要求每行RMS严格等于1：

$$
\operatorname{RMS}(\hat{x})
=\sqrt{\frac{q}{q+\epsilon}}
$$

q远大于eps时才接近1；全零输入仍输出0。乘不同gamma后更不保证单位RMS，
gamma也不必全为正。这不是softmax，不要求每行和为1，更不保证零均值。

### 2.5 用 `[3,4]` 手算，区分RMSNorm与L2归一化

忽略eps，令gamma为 `[1,1]`：

```text
x = [3,4]
平方均值 = (3²+4²)/2 = 12.5
RMS = sqrt(12.5) ≈ 3.5355
y ≈ [3/3.5355,4/3.5355] = [0.8485,1.1314]
```

为什么不是熟悉的 `[0.6,0.8]`？因为后者除的是L2长度5：

$$
\|x\|_2=\sqrt{\sum_i x_i^2},\qquad
\operatorname{RMS}(x)=\frac{\|x\|_2}{\sqrt{E}}
$$

| 操作 | `[3,4]`的输出 | 被归一到1的量（忽略eps） |
| --- | --- | --- |
| L2归一化 | `[0.6,0.8]` | 向量二范数 |
| RMS归一化，gamma=1 | `[0.8485,1.1314]` | 分量的均方根 |

后者的二范数为sqrt(2)，不是1；归一化目标不同，不是哪一个算错了。

### 2.6 公式怎样落到PyTorch代码

以下对应本课程的FP32内部计算路径，不是声明所有实现都必须这样写：

```python
input_dtype = x.dtype
x = x.float()
mean_square = x.square().mean(dim=-1, keepdim=True)
normalized = x * torch.rsqrt(mean_square + eps)
y = (normalized * weight).to(input_dtype)
```

`rsqrt(a)`是 `1/sqrt(a)`，所以乘rsqrt等价于除以平方根。
`keepdim=True`将 `[B,T,E]` 的平方均值保留为 `[B,T,1]`，方便给每个特征广播。
weight应是注册的可学习参数，不是每次forward临时新建一份。

本仓库Mini RMSNorm明确调用 `x.float()`；即使对模块执行 `.double()`，
内部归约也仍是FP32。Notebook因此分别保留FP64手算oracle与FP32实现对齐，
不会把1e-12的容差强加给FP32算法。FP64输入也不代表该实现提供FP64内部计算。
`torch.nn.RMSNorm`的默认eps可能不同，比较时要显式统一 `eps=1e-6` 和gamma。

### 2.7 对照源码与一手阅读链接

建议按“本页推导 → 小例子 → API定义 → 原论文”的顺序阅读：

- [PyTorch 2.10 RMSNorm：公式、normalized_shape、eps与可学习参数](https://docs.pytorch.org/docs/2.10/generated/torch.nn.RMSNorm.html)
- [RMSNorm原论文：Zhang与Sennrich，2019](https://arxiv.org/abs/1910.07467)
- [PyTorch 2.10 rsqrt：平方根的倒数](https://docs.pytorch.org/docs/2.10/generated/torch.rsqrt.html)
- [本仓库Mini RMSNorm，`class RMSNorm`](../learning/mini_transformer/model.py)
- [手算与边界Notebook](../learning/day08_rmsnorm_rope.ipynb)
- [本地实战接口与CPU/CUDA验收](LOCAL_PRACTICE.md)

本页是前向推导和教学实现说明，不等于完整反向公式推导或GPU性能证明。

### 2.8 RMSNorm 与 LayerNorm 的区别

```text
RMSNorm：  只看平方均值，不减均值
LayerNorm：先减均值，再按方差归一化
```

所以 RMSNorm 不是 LayerNorm 的简单改名。一个快速负例：

```text
x = [10,11,12]
```

RMSNorm 后仍然保留“整体偏正”的方向；LayerNorm 会先把均值移到0附近。

### 2.9 RMSNorm 的边界测试

至少测试：

```text
全零向量
常数向量
很大值：1e4
很小值：1e-4
不同最后维度
FP32 与 BF16/FP16 的误差
```

不要只用随机正态输入；随机输入可能掩盖 `eps`、dtype 和维度错误。

## 3. RoPE：把每两维看成一根小指针

以二维向量为例：

```text
[x0,x1]
```

位置 `p` 给它一个角度 `theta_p`，旋转后：

```text
x0' = x0*cos(theta_p) - x1*sin(theta_p)
x1' = x0*sin(theta_p) + x1*cos(theta_p)
```

二维平面上，这只是“转动指针”：

```text
位置0：本页标准RoPE的角度为0，不旋转
位置1：转一个角度
位置2：转更大的角度
```

head dimension 为 `D` 时，有 `D/2` 个特征对，但要先看配对布局。
本仓库Mini与Qwen3实现采用 **split-halves**：

```text
[x0,x1 | x2,x3] → (x0,x2)、(x1,x3)
                    第0对     第1对
```

相邻配对的interleaved布局则是 `(x0,x1)、(x2,x3)`。
两种布局可以通过对应置换表达同样的旋转思想，但不能不转换相应权重/特征顺序就混用。
本页Notebook和测试都固定split-halves，不仅“看起来在旋转”就算对齐。

不同的特征对使用不同频率，因此一部分“指针”转得快，一部分转得慢。
这组频率由 `theta/rope_base` 和特征索引决定。
更明确地写，频率 `omega_i=theta_base^(-2i/D)`，位置p的旋转角为
`phi_{p,i}=p*omega_i`，i从0到D/2-1。**theta_base是频率基数，不是直接旋转的角度。**
例如D=4、theta_base=10000，两对频率分别为1和0.01弧度/位置。

<a id="rope-angle-layout"></a>

### 3.1 “按theta旋转”不是与split-halves并列的另一种RoPE

这里有三个不同问题，分开理解就不会混：

| 问题 | 决定什么 | 示例 |
| --- | --- | --- |
| 配对布局 | 哪两个特征组成一个二维向量 | split-halves或interleaved |
| 位置与频率 | 这一对转多少角度phi | `phi = position * frequency` |
| 计算写法 | 怎样算出旋转后的值 | sin/cos公式、旋转矩阵或复数乘法 |

**两种配对布局都做角度旋转**，不是split-halves不旋转、另一种才旋转。
对选定的一对特征 `(a,b)`，统一计算为：

$$
\begin{bmatrix}a'\\b'\end{bmatrix}
=
\begin{bmatrix}
\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi
\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}
$$

展开就是 `a'=a*cos(phi)-b*sin(phi)`、
`b'=a*sin(phi)+b*cos(phi)`。
如果把同一对写成复数 `z=a+ib`，也可写 `z'=z*exp(i*phi)`；
Llama官方实现用复数乘法完成这一计算。
矩阵、三角函数和复数是计算表达方式，不是可以不核对布局就混用的三套权重格式。

### 3.2 代码参数theta是基数，真正的旋转角是phi

本模板接口为 `rope(x, start=0, theta=10000.0)`。
这里保留theta这个常见参数名，但它表示频率基数theta_base，不是直接给每一对转10000弧度。
为避免同名字母混用，文中用omega表示频率，用phi表示角度：

$$
\omega_i=\mathrm{theta\_base}^{-2i/D},\qquad
\phi_{p,i}=p\,\omega_i,\qquad i=0,\ldots,D/2-1
$$

这是本页标准RoPE配置，位置角按弧度计。举例：

```text
D=4，theta_base=10000，位置p=2
第0对：omega_0=10000^0=1    → phi=2*1=2弧度
第1对：omega_1=10000^-0.5=0.01 → phi=2*0.01=0.02弧度
```

split-halves时，两角分别作用于 `(x0,x2)`、`(x1,x3)`；
interleaved时则分别作用于 `(x0,x1)`、`(x2,x3)`。
直接对同一原始排列使用不同配对，数值一般不同；
若要表达同一个映射，必须同时保持特征排列与对应投影权重的转换一致。

记忆：**配对方式决定谁和谁一起转；位置与频率决定转多少；矩阵/复数决定怎样写计算。**
长上下文缩放或其他模型变体可能进一步改频率/位置规则，仍以目标实现为准。

一手链接：
[RoFormer原论文](https://arxiv.org/abs/2104.09864)、
[Llama 3官方代码中的precompute_freqs_cis与apply_rotary_emb](https://github.com/meta-llama/llama3/blob/main/llama/model.py)、
[本仓库split-halves与interleaved示意](../reasoning_from_scratch/qwen3.py)。

**在本课程架构里，RoPE只作用于Q/K，不作用于V。**

```text
Q → 旋转后的Q
K → 旋转后的K
V → 保持内容表示
```

这样 `Q·K` 的结果既包含内容匹配，也包含位置信息；V仍是最终被汇集的内容。

## 4. 为什么能表达相对位置

用p、r表示位置，q、k表示内容向量，避免位置q与Query向量q同名。
每个频率下的二维旋转矩阵满足：

```text
R(p)ᵀ R(r) = R(r-p)
```

因此：

```text
(R(p)q) · (R(r)k) = qᵀ R(r-p) k
```

会包含与位置差 `r-p` 相关的结构。
固定相同内容q/k，如果两个位置同时平移相同距离，理想旋转点积不变；
数值实验按浮点容差检查。不是任意不同token只要距离相同就应有相同分数。

这不是说模型只看距离：

- q/k 内容向量仍然不同；
- 多个频率叠加；
- 长距离可能出现周期性混淆；
- RoPE不负责因果Mask。

因此不要说“距离相同，最终分数一定相同”。

## 5. offset：为什么Cache Decode必须传位置偏移

Prefill 处理位置0、1、2、3时：

```text
RoPE positions = [0,1,2,3]
```

如果Cache已有4个位置，Decode新token必须使用：

```text
RoPE positions = [4]
```

不是重新使用位置0。

```text
prefill：start=0，T=4
decode： start=4，T=1
```

shape不变：

```text
[B,H,1,D] → [B,H,1,D]
```

但数值会变，因为旋转角度变了。
offset错误通常不会立刻报shape错误，却会让full/cached logits对不齐。

新增两个有用的正确性检查：

```text
旋转不改每头向量的二范数（浮点容差内）。
同一序列整段旋转，与按正确start切块后拼接的旋转结果一致。
```

例如一次处理位置0..6，应与 `前3位置start=0 + 后4位置start=3` 对齐。
若第二段错误重置成start=0，shape仍正确，但数值不再对齐。

## 6. Day 8的实现顺序

### RMSNorm

1. 写 `[B,T,E]` 输入。
2. 只沿最后一维计算 `mean(x²)`。
3. 用 `eps` 做除法。
4. 乘 `gamma`。
5. 用 FP32 oracle 对齐。

Notebook给出可读参考与待做实验；学习者应另建自己的实现，
先写预测再与参考比较，不能把Run All通过写成“已独立实现”。

### RoPE

1. 写 `[B,H,T,D]` 输入。
2. 检查 `D` 为正偶数。
3. 生成位置 `[start, start+T)`。
4. 生成每个特征对的频率。
5. 逐对应用旋转。
6. 检查 shape不变、offset改变数值、`start=0`与参考实现对齐。

本页不修改已有模型。FP32/BF16/FP16的CPU数值比较属于参考验证，
不证明目标GPU的低精度kernel兼容或性能；本实现也未宣称支持所有无效输入策略。

## 7. 先正确，再优化

| 阶段 | 目标 | 暂时不做 |
| --- | --- | --- |
| 1 | FP32、CPU、随机/边界输入对齐 | GPU性能 |
| 2 | BF16/FP16误差与有限性 | 直接放宽阈值 |
| 3 | 多batch、不同T、offset、偶数D | 只测一个shape |
| 4 | 计时和内存 | 没有正确性证据就优化 |
| 5 | Triton/CUDA | 把PyTorch逐行实现叫生产kernel |

## 8. Day 8自测与参考答案

**问：RMSNorm读取其他token吗？**

答：不读。它沿每个位置的最后特征维度独立归一化。

**问：RMSNorm减均值吗？**

答：不减；减均值是LayerNorm的步骤。

**问：RoPE改变shape吗？**

答：不改变。它在每个head内对特征对做旋转。

**问：RoPE为什么不旋转V？**

答：本架构把位置关系注入Q/K匹配，V保留待汇集内容；具体模型仍需看实现。

**问：offset从0改成5，为什么shape不变但数值改变？**

答：shape只由B/H/T/D决定，offset改变位置角度，旋转后的数值随之改变。

**问：能用一个随机输入通过就宣布Norm/RoPE正确吗？**

答：不能。至少要有独立公式、边界输入、offset、dtype和误差检查。

**问：4维RoPE在本仓库配哪两对？**

答：split-halves是 `(0,2)、(1,3)`，不是相邻的 `(0,1)、(2,3)`。

**问：为什么`.double()`后不应要求Mini RMSNorm达到1e-12误差？**

答：模块forward内部仍转FP32；FP64只是输入/输出或参数类型，不改变内部算法。

**问：旋转会改变向量长度吗？**

答：理想旋转保持二范数；浮点误差允许小偏差，不把范数保持误解为点积不含位置信息。

## 推荐深入阅读

- [本仓库Mini Transformer实现](../learning/mini_transformer/model.py)
- [PyTorch RMSNorm文档](https://docs.pytorch.org/docs/2.10/generated/torch.nn.RMSNorm.html)
- [RMSNorm原论文](https://arxiv.org/abs/1910.07467)
- [RoFormer论文：Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Transformer原论文：Layer Normalization与位置编码背景](https://arxiv.org/abs/1706.03762)
