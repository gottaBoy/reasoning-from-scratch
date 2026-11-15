# P1-03：多头 Attention 的拆头与合头

P1-02 已完成单头 Attention。现在只练布局，不计算投影、分数、RoPE 或 KV Cache。
统一模板：[英文](../learning/my_ops_template.py) / [中文图解](../learning/my_ops_template_zh.py)。
个人实现仍只写在 `learning/my_ops.py`。

## 1. 两个接口

```python
def split_heads(x, heads, head_dim):
    # [B,T,H*D] -> [B,H,T,D]

def merge_heads(heads):
    # [B,H,T,D] -> [B,T,H*D]
```

以 `B=1、T=2、H=2、D=3` 为例：

```text
x [B,T,H*D] = [1,2,6]
       ↓ view
[B,T,H,D]   = [1,2,2,3]
       ↓ transpose(1,2)
heads       = [1,2,2,3]
```

拆头只是在每个位置把 `H*D` 个特征分成 `H` 组；不是把时间位置分给不同 head。
每个 head 仍然看到全部 `T` 个位置。

## 2. 你要实现什么

在 `my_ops.py` 中新增：

```text
S1：校验三维、FP32、连续输入以及最后一维 H*D
S2：x.view(B,T,H,D).transpose(1,2)
M1：heads.transpose(1,2)
M2：reshape(B,T,H*D)
```

`split_heads` 要求输入 contiguous，是为了把 `view` 的布局条件暴露出来；
`merge_heads` 的输入通常来自 transpose，因此不能要求它 contiguous。

## 3. 标签例子

```python
x = torch.tensor([[
    [0., 1., 2., 3., 4., 5.],
    [10., 11., 12., 13., 14., 15.],
]])
heads = split_heads(x, heads=2, head_dim=3)
```

正确结果：

```text
head 0:
  position 0 -> [0, 1, 2]
  position 1 -> [10, 11, 12]

head 1:
  position 0 -> [3, 4, 5]
  position 1 -> [13, 14, 15]
```

合头必须先把 `[B,H,T,D]` 换回 `[B,T,H,D]`：

```python
merged = heads.transpose(1, 2).reshape(B, T, H * D)
```

故意写成 `heads.reshape(B,T,H*D)`，`T>1` 时会把不同时间位置和 head 拼错。
先 `.contiguous()` 再直接 reshape 也不能修复轴语义，只会把错误布局复制得更规整。

## 4. 验收

先运行参考基线确认链路：

```bash
cd /workspace
python -m learning.local_practice --task layout --device cpu
```

完成 `my_ops.py` 后：

```bash
python -m learning.local_practice \
  --task layout \
  --device cpu \
  --candidate learning/my_ops.py

python -m learning.local_practice \
  --task layout \
  --device cuda \
  --candidate learning/my_ops.py
```

10 项检查覆盖：

```text
拆头 shape/stride、标签顺序、合头往返、合头标签顺序、
非方形 B/T/H/D、storage/contiguous、往返梯度、
宽度错误、非连续输入、空输入拒绝
```

个人通过条件是自己的 candidate CPU/CUDA 报告，不是 reference baseline 通过。

## 5. 练习顺序

1. 手算标签矩阵，不看实现。
2. 在 `my_ops.py` 写 `split_heads`，先验收。
3. 写 `merge_heads`，再验收。
4. 故意去掉合并前 transpose，保存失败报告。
5. 改成 `B=3,T=5,H=4,D=7`，确认不是只对小例子有效。
6. 口述：Linear 改数值，view 改解释，transpose 改轴顺序，reshape 可能复制。

通过后再进入 P1-04：MHA/MQA/GQA 的 KV head 共享。
