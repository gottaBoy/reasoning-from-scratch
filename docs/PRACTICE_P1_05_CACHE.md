# P1-05：请求级 KV Cache

P1-04 已完成 KV head 映射。现在练习的是**请求状态的生命周期**：
Cache 属于请求，不属于模型全局。

## 目标

使用 `learning/mini_transformer/model.py` 中的小型 FP32 Transformer，验证：

```text
full forward ≈ 分块 forward + 请求级 Cache
```

输入可以用四种模态来类比：

| 模态 | 序列单位 | Cache 保存 |
| --- | --- | --- |
| 文本 | token | 历史 token 的 K/V |
| 图像 | patch/token | 历史 patch 的 K/V |
| 音频 | 时间帧 | 历史帧的 K/V |
| 视频 | 时空 patch/token | 历史帧的 K/V |

Cache 保存的是投影后的 K/V 特征，不是原始文本、像素、音频或视频文件。

## 第一组实验

在容器或 Jupyter Terminal 中运行参考验收：

```bash
python -m learning.local_practice --task cache --device cpu
python -m learning.local_practice --task cache --device cuda
```

个人实现暂时使用一个导出 `Cache`、`Config`、`Model` 的独立 `.py` 文件：

```bash
python -m learning.local_practice \
  --task cache --device cpu --candidate learning/my_cache.py
```

不要把 `learning/mini_transformer/model.py` 直接当作个人实现证据；它是课程参考基线。

## 必须观察的结果

1. `3+2+3` 分块时 Cache 长度为 `3 → 5 → 8`。
2. 分块 logits 与一次性 full logits 全部位置对齐，不能只比较最后一个位置。
3. `5+1+1+1` 这种不同切分也要对齐。
4. 两个请求使用两个 Cache；模型权重可以共享，历史 K/V 不能串线。
5. 上下文超限必须在修改 Cache 前拒绝，拒绝后旧长度和旧 K/V 不变。
6. 所有层的有效 Cache 长度一致。

## 位置与 Mask

第一个请求块从位置 0 开始。若已有长度为 3，下一块的 Query 绝对位置是
`3, 4, ...`，不是重新从 0 开始。对应的 score 形状（省略 batch/head）为：

```text
第一块：3×3
第二块：2×5
第三块：3×8
```

新 Query 可以读取历史和当前已到达位置，不能读取未来位置。

## 失败实验

依次只改一个变量并记录结果：

- 每个块都把 `start` 重置为 0；
- 两个请求复用同一个可变 Cache；
- 只比较最后一行 logits；
- 先追加 K/V，再检查上下文长度；
- 清空 K/V 但保留旧 position。

这些错误有些会直接报错，有些 shape 合法但数值错误。

P1-05 只验证教学小模型的请求级 Cache，不代表分页 Cache、连续批处理、
生产 Attention kernel 或多模态编码器的全部位置编码实现。
