# 70B 模型 KV Cache 内存估算

## 1. 公式

每层每个请求的 K/V Cache：

```text
bytes_per_layer =
  2 × batch_size × seq_len × kv_heads × head_dim × dtype_bytes
```

前面的 `2` 表示两份张量：

```text
K 一份 + V 一份
```

全部层：

```text
total_bytes =
  2 × num_layers × batch_size × seq_len
    × kv_heads × head_dim × dtype_bytes
```

这里的 `2` 与 dtype 的 `2 bytes` 是不同的两个 2，不能合并后忘记含义。
公式适用于各层维度相同、K/V head_dim 相同的普通全上下文 MHA/GQA/MQA。
`seq_len` 是当前实际保留的历史长度，通常包含 prompt 和已处理的生成 token，
不是只计算 `max_new_tokens`，也不一定等于配置允许的最大上下文。
MLA、滑动窗口、跨请求 prefix sharing 等需要根据具体缓存表示另算。

### 重要：70B 不是计算公式中的变量

“70B”是参数规模，不能直接代入 KV Cache 公式。
需要从具体 checkpoint/config 得到：

```text
num_layers
num_key_value_heads
head_dim
```

以 [Meta 官方配置中的 Llama 3.1 70B](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)
为例：80 层、主路径宽8192、64个 Query heads、8个 KV heads，
其每头维度为128。**公式使用的是8个 KV heads，不是64个 Query heads。**

```text
L       = 80
Hkv     = 8
D       = 128
```

具体模型版本仍应以实际 config 为准；不要仅凭“70B”猜参数。

## 2. Llama 3.1 70B 算例

假设：

```text
batch_size = 1
dtype      = FP16/BF16 = 2 bytes
```

每层：

```text
2 × 1 × seq_len × 8 × 128 × 2 bytes
= 4096 × seq_len bytes
```

换算后：

| 上下文长度 | 每层 K/V | 80 层总 K/V |
|---:|---:|---:|
| 4,096 | 16 MiB | 1.25 GiB |
| 8,192 | 32 MiB | 2.5 GiB |
| 32,768 | 128 MiB | 10 GiB |
| 65,536 | 256 MiB | 20 GiB |
| 131,072 | 512 MiB | 40 GiB |

所以：

```text
FP16/BF16, batch=1, 32K context  → 约 10 GiB KV Cache
FP16/BF16, batch=1, 128K context → 约 40 GiB KV Cache
```

这些是理论紧凑 K/V 数组大小，不包含 block/page metadata、allocator 碎片、
padding、workspace、logits 或其他运行时张量。
也不是已在本机加载70B实测的峰值；本例是按官方结构做的算术估算。

### 一行可套用的公式

这个模型每增加一个历史 token，所有层的 FP16/BF16 K/V 共增加：

```text
2 × 80 × 8 × 128 × 2 = 327,680 bytes = 320 KiB
```

所以：

```text
Llama 3.1 70B：
KV GiB = 10 × batch_size × (seq_len / 32768) × (dtype_bytes / 2)
```

例如32K、batch=1：

```text
2 × 80 × 1 × 32768 × 8 × 128 × 2
= 10,737,418,240 bytes
= 10 GiB（约10.74 GB）
```

### Batch 线性放大

```text
batch=2, 32K → 约 20 GiB
batch=8, 32K → 约 80 GiB
```

服务系统常按活跃 token 总数管理 Cache：

```text
KV bytes ∝ active_sequences × average_context_tokens
```

异长请求更准确地用 `sum(T_i)` 代替 `batch_size × seq_len`，
前提是没有额外 padding/共享状态；实际预留的池容量也可能大于已用 Cache。

## 3. dtype 对照

仍以 batch=1、32K、上述 70B 结构为例：

| KV dtype | 理论 K/V |
|---|---:|
| FP32，4 bytes | 20 GiB |
| FP16/BF16，2 bytes | 10 GiB |
| INT8/FP8，约 1 byte | 5 GiB |
| INT4，理想 0.5 byte | 2.5 GiB |

INT8/FP8/INT4 的实际占用还要加 scale、分组 metadata 和对齐空间，
并重新验证 Attention 数值误差和速度。目标推理框架和硬件还需实际支持该 KV dtype。
**权重INT4不代表KV也是INT4**，二者量化配置不能混用。

## 4. 与权重内存区分

70B FP16 权重的粗略量级：

```text
70B × 2 bytes ≈ 140 GB
```

但这只是权重本身：

```text
权重：
  主要随模型结构变化，通常多个请求共享

KV Cache：
  随层数、KV heads、head_dim、上下文和活跃 batch 变化
  每个请求有自己的历史状态
```

因此量化权重能让模型装入内存，不代表长上下文/高并发不会因为 KV Cache OOM。
GQA 减少 `kv_heads`，对 Cache 有直接线性收益；FlashAttention 减少 score
中间张量 IO/物化，不等于删除 KV Cache。

## 5. 与 MiniMind 例子对照

你的 MiniMind 假设：

```text
L=8, Hkv=2, D=64, T=32768, B=1, dtype=2 bytes
```

全部层：

```text
2 × 8 × 1 × 32768 × 2 × 64 × 2
= 134,217,728 bytes
= 128 MiB
```

每层是 `16,777,216 bytes = 16 MiB`。原例写16MB/128MB时，严格来说应写
16MiB/128MiB；MB和GB用1000进制，MiB和GiB用1024进制。

两组假设的结构因子：

```text
MiniMind：L×Hkv×D = 8×2×64 = 1024
70B：     L×Hkv×D = 80×8×128 = 81920
```

所以在相同 `B/T/dtype` 下，70B 这个例子的 Cache 约是 80 倍。
这不是说 70B 参数量是 MiniMind 的 80 倍，而是 Cache 结构因子不同。

## 6. 面试回答

**问：70B 模型 32K 上下文需要多少 KV Cache？**

> 不能只根据 70B 参数量回答。以 80 层、8 个 KV heads、head_dim 128 为例，
> FP16/BF16、batch=1、32K context 的理论紧凑 K/V Cache 约为 10 GiB；
> 128K 约为 40 GiB。实际还要加分页 metadata、workspace、碎片和其他张量，
> 并按活跃 batch 线性放大。

**问：为什么 GQA 有帮助？**

> KV Cache 对 `kv_heads` 线性增长。Query heads 可以保持不变，但减少 KV heads
> 会减少紧凑历史状态；计算前按组给 Query 使用。是否真正减少临时复制要看 kernel。

**问：FlashAttention 能解决 KV Cache OOM 吗？**

> 不能直接解决。FlashAttention 主要避免完整 score/probability 的中间物化并优化 IO；
> KV Cache 是历史 K/V 状态，仍需要分页、量化、offload 或容量管理。

## 7. 自己算一遍

```text
L=80, Hkv=8, D=128, B=4, T=65536, dtype=2 bytes
```

先手算，再运行：

```python
L, Hkv, D, B, T, bytes_per = 80, 8, 128, 4, 65536, 2
total = 2 * L * B * T * Hkv * D * bytes_per
print(total / 1024**3, "GiB")
```

参考答案为80GiB。验收自己能解释每个乘数，而不是只得到一个数字。

## 参考

- [Meta 官方模型结构配置](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)
