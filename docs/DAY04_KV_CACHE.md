# Day 4：在 Jupyter 中理解 KV Cache

打开 [Day 4 Notebook](../learning/day04_kv_cache.ipynb)，从上到下运行。
它默认使用 Tiny 模型和固定输入，先证明机制，再切换真实 Qwen3。

## 你要观察什么

固定输入：

```text
[token1, token2, token3, token4, token5, token6, token7, token8]
```

两种计算：

```text
完整前向：全部 8 个 token 一次计算

Cached Decode：
Prefill：token1...token7
Decode： token8
```

如果机制正确，两种方式最后位置的 logits 应该一致。Notebook 会自动比较最大
绝对误差，并检查每层 Cache 的 K/V 形状。Tiny FP32 应逐元素通过；
Qwen3 BF16 还要单独看误差和 top-1 是否一致，不能套用 FP32 的严格阈值。

## 五个问题的直接答案

1. **Prefill 处理 7 个 prompt token，Decode 处理 1 个最新 token。**
   真实生成时 Prefill 长度取决于 prompt，Decode 通常每轮只输入一个新 token。
2. **Decode 只输入最新 token**，因为历史 token 的 K/V 已经保存在 Cache；
   重复输入历史 token 会重复计算它们的 K/V。
3. **Cache 保存 K 和 V，不保存历史 Q。** 历史 K/V 会被后续 token 读取；
   历史 Q 只在产生对应位置输出时使用一次。
4. **Qwen3 使用 8 个 KV groups 而不是 16 个 Query heads**，因为 GQA 让每
   2 个 Query head 共享一组 K/V，减少 KV Cache 存储。
5. **完整前向和 Cached Decode 的最后位置 logits 应一致**，允许的误差取决于
   dtype；Tiny FP32 实验使用严格容差。Qwen3 0.6B 在本机 CUDA/BF16 实测
   top-1 一致，最大绝对误差约 `0.25`；换成 FP32 后最大绝对误差约
   `1.35e-5`。这说明 Cache 路径逻辑一致，同时低精度矩阵运算会产生可见数值差异。

## 本机实测结果

2026-10-31 在 DGX Spark 上用固定 ID `[0,1,2,3,4,5,6,7]`：

| 配置 | Cache 形状（第 0 层） | 最后位置最大误差 | Top-1 |
| --- | --- | ---: | --- |
| Tiny / FP32 | Prefill `[1,2,7,16]`，Decode 后 `[1,2,8,16]` | `2.38e-7` | 一致 |
| Qwen3 0.6B / BF16 CUDA | Prefill `[1,8,7,128]`，Decode 后 `[1,8,8,128]` | `0.25` | 一致 |
| Qwen3 0.6B / FP32 CUDA | 同上 head/sequence 形状 | `1.35e-5` | 一致 |

不要因为 BF16 不是逐元素严格相等就判定 Cache 错误；先用 FP32 对照，
再确认 top-1、误差分布和 Cache 位置是否正确。也不要把一次 8 token 的
top-1 一致性当成长期生成质量保证。

## 调试位置

如果要观察 Cache 追加，给下面代码行设置断点：

```text
reasoning_from_scratch/qwen3.py:206
reasoning_from_scratch/qwen3.py:208
reasoning_from_scratch/qwen3.py:213
```

第二次调用 Decode 时查看：

```python
print(tuple(prev_k.shape))
print(tuple(keys_new.shape))
print(tuple(keys.shape))
```

预期是：

```text
prev_k   [B, KV, 7, D]
keys_new [B, KV, 1, D]
keys    [B, KV, 8, D]
```

这里的 `keys` 是 Cache 拼接后的紧凑 K；到第 216 行之后才扩展成 Query head
数量参与矩阵乘法。

`Qwen3Model.current_pos` 同步记录 Cache 的逻辑长度：Prefill 后为 7，
Decode 后为 8。每次独立实验都要 `reset_kv_cache()` 并创建新的 `KVCache`，
否则上一轮的位置会污染下一轮结果。
