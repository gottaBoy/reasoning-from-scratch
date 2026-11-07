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

假设一个位置的隐藏向量为：

```text
x = [3,4]
```

平方均值：

```text
mean(x²) = (3²+4²)/2 = 12.5
RMS(x) = sqrt(12.5) ≈ 3.5355
```

忽略可学习缩放 `gamma` 和极小的 `eps`：

```text
RMSNorm(x) = x / RMS(x)
            ≈ [0.8485, 1.1314]
```

形象理解：

```text
原始声音：同一个人突然一会儿很大声、一会儿很小声
RMSNorm：先把整体音量拉到稳定范围
gamma：模型再决定每个频段应该放大或压低多少
```

“音量/频段”只是帮助理解尺度与通道，不代表隐藏特征真是音频频率。
gamma可以不全为正，也不保证乘完后每行的均方根仍然等于1。

公式：

```text
RMSNorm(x) = gamma ⊙ x / sqrt(mean(x²) + eps)
```

- `mean(x²)` 沿最后一个特征维度计算。
- `gamma` shape 是 `[E]`，按特征广播到 `[B,T,E]`。
- `eps` 防止输入全零或极小值时除零。
- 常用 FP32 计算平方、求和和平方根，再转回原 dtype。

本仓库Mini RMSNorm明确调用 `x.float()`；即使对模块执行 `.double()`，
内部归约也仍是FP32。Notebook因此分别保留FP64手算oracle与FP32实现对齐，
不会把1e-12的容差强加给FP32算法。FP64输入也不代表该实现提供FP64内部计算。
`torch.nn.RMSNorm`的默认eps可能不同，比较时要显式统一 `eps=1e-6` 和gamma。

### RMSNorm 与 LayerNorm 的区别

```text
RMSNorm：  只看平方均值，不减均值
LayerNorm：先减均值，再按方差归一化
```

所以 RMSNorm 不是 LayerNorm 的简单改名。一个快速负例：

```text
x = [10,11,12]
```

RMSNorm 后仍然保留“整体偏正”的方向；LayerNorm 会先把均值移到0附近。

### RMSNorm 的边界测试

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
