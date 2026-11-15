# P1-06：Transformer Block、SwiGLU 与 LM 输出

P1-05 已完成请求级 Cache。现在先回到无 Cache 的训练路径，确认 Block 的残差顺序、
SwiGLU 门控和完整梯度链。

## 目标

```text
Y = X + Attention(norm1(X))
Z = Y + down(SiLU(gate(norm2(Y))) * up(norm2(Y)))
```

第二次 residual 加的是 `Y`，不是最初的 `X`。

## 参考验收

```bash
python -m learning.local_practice --task block --device cpu
python -m learning.local_practice --task block --device cuda
```

覆盖：

1. `FFN = down(SiLU(gate(x)) * up(x))`；
2. 两次 residual 的 `10 → 12 → 15` 连线；
3. FFN 输入和参数梯度有限；
4. LM logits shape 为 `[B,T,vocab]`；
5. cross-entropy backward 后全部参数有有限梯度；
6. `eval()` 不会自动关闭 autograd。

## 先做的个人练习

在 `learning/my_block.py` 中导出 `Config`、`RMSNorm`、`FFN`、`Block`。
可以暂时复用参考 `Attention`，本关只实现/核对 FFN 与 Block：

```python
class FFN(nn.Module):
    # gate: E -> F
    # up:   E -> F
    # down: F -> E
    ...

class Block(nn.Module):
    # x1 = x + attention(norm1(x))
    # x2 = x1 + ffn(norm2(x1))
    ...
```

运行个人验收：

```bash
python -m learning.local_practice \
  --task block --device cpu --candidate learning/my_block.py
```

不要在训练验收中使用 `inference_mode()` 或 `no_grad()`；`eval()` 只改变模块行为，
不会关闭计算图。

## 四种模态类比

文本中的位置可以是 token，图像中可以是 patch，音频中可以是时间帧，视频中可以是
时空 token。Block 的残差规则不因载体改变：每个位置先经过归一化和 Attention，
再经过 SwiGLU，主路径始终保留两次更新。
