# P1-04：MHA、GQA、MQA 的 KV head 共享

P1-03 已完成拆头与合头。现在只练紧凑 K/V 如何映射到 Query heads，
暂不加入完整 Attention、RoPE、Cache 分配或模型投影。

## 1. 一个函数看懂三种结构

```python
def expand_kv_heads(kv, query_heads):
    # [B,H_kv,T,D] -> [B,H_q,T,D]
```

固定 `H_q=4`：

```text
MHA：H_kv=4  Q0→KV0，Q1→KV1，Q2→KV2，Q3→KV3
GQA：H_kv=2  Q0/Q1→KV0，Q2/Q3→KV1
MQA：H_kv=1  Q0/Q1/Q2/Q3→KV0
```

共享的是 K/V head，不是 Q。四个 Query 仍然可以不同，因此输出不必相同。

## 2. 你要实现什么

在 `my_ops.py` 中新增：

```text
G1：group_size = query_heads // kv.shape[1]
G2：kv.repeat_interleave(group_size, dim=1)
```

输入和输出：

```text
紧凑 kv：[B,H_kv,T,D]
扩展结果：[B,H_q,T,D]
```

要求 `H_q % H_kv == 0`。例如 `H_q=4,H_kv=2` 时：

```text
KV0 -> Query0、Query1
KV1 -> Query2、Query3
```

`repeat_interleave` 是教学阶段的显式扩展，可能产生临时副本。
真正的 KV Cache 应保持紧凑 `[B,H_kv,T,D]`，不能因为计算时扩展就把四头副本永久缓存。

## 3. 本地验收

参考基线：

```bash
cd /workspace
python -m learning.local_practice --task gqa --device cpu
```

完成个人实现后：

```bash
python -m learning.local_practice \
  --task gqa \
  --device cpu \
  --candidate learning/my_ops.py

python -m learning.local_practice \
  --task gqa \
  --device cuda \
  --candidate learning/my_ops.py
```

10 项检查覆盖 MHA/GQA/MQA 映射、非方形 shape、输入不修改、
扩展副本独立、梯度和非法 head 数。

## 4. 先算内存，再看代码

普通 KV Cache 近似：

```text
2 × layers × batch × sequence × H_kv × D × dtype_bytes
```

固定其他条件时，`H_kv` 从 4 降到 2，紧凑 KV Cache 减半；
模型其他权重、Q 投影、临时扩展和 allocator 开销不会自动减半。

## 5. 练习顺序

1. 画出 `H_q=4、H_kv=4/2/1` 的映射。
2. 手算 `group_size` 和输出 shape。
3. 实现 `expand_kv_heads`，先 CPU 后 CUDA。
4. 故意用 `kv.repeat(1, query_heads, 1, 1)`，观察为什么映射错误。
5. 解释“紧凑持久 Cache”和“计算临时扩展”的区别。

通过后进入 P1-05：请求级 KV Cache 的分配、追加、回收和 full/cached 对齐。
