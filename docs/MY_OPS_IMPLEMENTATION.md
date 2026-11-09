# P1-01：my_ops.py的实现思路与注释模板

本页是实现指导，不直接把你的练习替换成参考答案。
配套文件：[可安全导入的模板](../learning/my_ops_template.py)、
[实战Notebook](../learning/practice01_start.ipynb)。

## 1. 先分清“定义接口”和“调用接口”

下面是调用，不是实现：

```python
RMSNorm(width, eps=1e-6)
rope(x, start=0, theta=10000.0)
```

如果width或x还未定义，而且这些调用放在文件最外层，
验收器导入my_ops.py时就会执行并报错，根本到不了数值测试。

你要写的是：

```python
class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        ...

    def forward(self, x):
        ...


def rope(x, start=0, theta=10000.0):
    ...
```

这里的省略号只是说明结构；实际练习按模板TODO填写。
不需要在模块顶层调用它们，也不必直接导入参考RMSNorm。
测试会创建实例、准备输入并执行forward/rope。

## 2. RMSNorm：四行计算对应四个步骤

目标公式见 [完整推导](DAY08_RMSNORM_ROPE.md#rmsnorm-derivation)：

```text
y = weight * x / sqrt(mean(x²,最后一维) + eps)
```

### 初始化：只创建一次gamma

```python
super().__init__()
self.eps = eps
self.weight = nn.Parameter(torch.ones(width))
```

`weight`就是gamma，shape为 `[width]`。注册为Parameter后才是模块可学习参数，
会出现在parameters里，也会随模块的`.to(device)`移动。
验收器按接口读写`.weight`，所以不能只复用名字为`.scale`的参考类而不适配接口。
gamma不应在每次forward里新建，否则它不再是同一套持续学习的参数。

### forward：对应模板R1–R4

| TODO | 写什么计算 | 输入shape → 输出shape |
| --- | --- | --- |
| R1 | 对x_fp32平方后沿dim=-1求mean，keepdim=True，结果命名mean_square | `[B,T,E] → [B,T,1]` |
| R2 | 计算inv_rms = torch.rsqrt(mean_square + self.eps) | `[B,T,1] → [B,T,1]` |
| R3 | normalized = x_fp32 * inv_rms，共同缩放系数按特征广播 | `[B,T,E]` |
| R4 | output = normalized * self.weight，转回input_dtype并return | `[B,T,E]` |

`x.float()`不会改变device，但会选择FP32计算；保存原dtype用于最后恢复。
模板用 `input_dtype` 保存输入类型，用 `x_fp32` 表达计算输入，
不再用含义较泛的 `work`；`inv_rms`表示加入eps后的均方根倒数，
`normalized`表示尚未乘gamma的结果，`output`表示乘gamma后的最终结果。
这些名称符合小写下划线习惯，也直接对应公式含义，不是强制所有项目都采用相同名字。
最后一轴是特征，不是Batch或时间；不能不加dim而对整个张量求mean。
keepdim保留大小为1的特征轴，让广播方向明确。
不要先减均值，不要detach或在forward里包no_grad，否则梯度验收会失败。
禁止项分开读：不要在此forward中截断梯度；不要在每次forward中新建Parameter。
这不禁止测试时在no_grad上下文初始化/复制weight；不同作用范围应分清。

## 3. RoPE：先算位置与频率，再旋转两半特征

输入 `[B,H,T,D]`，当前实战固定split-halves：

```text
D=4：[x0,x1 | x2,x3]
配对：   (x0,x2)、(x1,x3)
```

注意不是相邻配对；与验收oracle布局必须一致。
两种配对都按角度旋转，不是“split-halves之外还有一种才是theta旋转”。
接口中的theta是频率基数，真正角度为 `position * theta**(-2*i/D)`。
例如D=4、theta=10000、位置2，两对角度为2和0.02弧度。
旋转矩阵、sin/cos和复数乘法是不同表达方式，完整对照和官方源码链接见
[配对、频率与旋转写法](DAY08_RMSNORM_ROPE.md#rope-angle-layout)。

| TODO | 写什么计算 | shape |
| --- | --- | --- |
| P1 | 特征对索引i，从0到D/2-1；torch.arange显式指定device和浮点dtype | `[D/2]` |
| P2 | 每对频率 `theta**(-2*i/D)` | `[D/2]` |
| P3 | 位置从start到start+T-1，不总从0开始 | `[T]` |
| P4 | 位置列向量乘频率行向量，得到旋转角 | `[T,D/2]` |
| P5 | 输入前半left、后半right | 各`[B,H,T,D/2]` |
| P6 | left/right按同一angles的cos/sin旋转 | 各`[B,H,T,D/2]` |
| P7 | 两个新半部按特征轴cat，恢复dtype后return | `[B,H,T,D]` |

旋转公式：

```text
new_left  = left*cos(angle) - right*sin(angle)
new_right = left*sin(angle) + right*cos(angle)
```

角度 `[T,D/2]`可广播到各Batch和head，不要把位置向量放错轴。
`start`改变角度，不改变T或D；奇数D、负start先抛ValueError。
所有新建位置/频率张量应在x.device，避免CPU/CUDA混用报错。

## 4. 建议填写顺序

1. 先只完成RMSNorm的R1–R4，运行CPU验收，观察7项Norm检查。
2. 此时RoPE尚未填，报告整体失败是正常的，不修改验收器来隐藏它。
3. 再填写RoPE的P1–P7，先核对D=4、T=3、start=5各中间shape。
4. 让CPU 14项通过后再明确请求CUDA，不静默回退。
5. 故意忽略start，确认offset检查报错；修回后保留错误修复记录。

```bash
cd /workspace
python -m learning.local_practice --device cpu --candidate learning/my_ops.py
python -m learning.local_practice --device cuda --candidate learning/my_ops.py
```

模板中的NotImplementedError是有意的：它尚未实现，不应该通过数值验收。
只有把你自己的文件补完，再传入对应candidate，才是在检验该文件。
目录存在于宿主机还是容器也要分清：宿主机改文件后，只同步自己的my_ops.py，
不要用全量同步覆盖Jupyter里其他尚未保存的笔记。

## 5. API、公式与本地入口

- [PyTorch 2.10 Parameter：注册可学习参数](https://docs.pytorch.org/docs/2.10/generated/torch.nn.parameter.Parameter.html)
- [PyTorch 2.10 mean：dim与keepdim](https://docs.pytorch.org/docs/2.10/generated/torch.mean.html)
- [PyTorch 2.10 rsqrt](https://docs.pytorch.org/docs/2.10/generated/torch.rsqrt.html)
- [RMSNorm公式与论文链接](DAY08_RMSNORM_ROPE.md#rmsnorm-derivation)
- [RMSNorm/RoPE验收器](../learning/local_practice.py)
- [每关反复练习路线](LOCAL_PRACTICE.md)

本模板说明接口与实现顺序，不是完整生产算子，也不自动证明独立实现或GPU性能。
自己的RMSNorm/RoPE文件通过定义范围的CPU/CUDA验收后，继续
[P1-02单头因果Attention](PRACTICE_P1_02_ATTENTION.md)，不要覆盖已完成的my_ops.py。
