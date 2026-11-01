# 剩余 25 天执行包

当前学习位置为 **Day 5**，个人验收由学习者填写；后面还有
**Day 6–30，共 25 个学习日**。
本文件把后续任务转换成可执行的入口、产出和验收标准。
配套的 25 份每日 Notebook 与参考实现已经准备，实际入口以
[课程总表](COURSE_DAY06_30.md) 为准；以下旧目录树是练习拆分建议，不表示每个
建议文件名都对应一个独立实现。各模块已按现有代码组织，未创建空文件凑数量。
知识讲义与面试训练见 [每日知识索引](KNOWLEDGE_INDEX.md)。

## 已完成基线

- DGX Spark CUDA、Triton、Qwen3 推理链路已验证。
- Day 3 Q/K/V、Mask、RoPE 形状实验已验证。
- Day 4 Tiny FP32 Cache 严格对齐已验证。
- Qwen3 0.6B FP32/BF16 Cache 对照已记录。
- Day 5 架构、MoE、MLA 教学 Notebook 已准备。

## Day 6–7：从模型到系统

### Day 6：FlashAttention 前置知识

入口：

- `docker/check_triton.py`
- `docs/DAY03_SHAPES_DEBUGGING.md`
- `reasoning_from_scratch/qwen3.py:219`

任务：

1. 画出完整 attention score 的形状和显存规模。
2. 用小矩阵手算 Online Softmax 的 `m`、`l` 和累积输出。
3. 解释为什么分块后不需要保存完整 `[B,H,T,T]`。
4. 比较朴素 Attention、`scaled_dot_product_attention` 和教学版分块实现。

产出：

```text
learning/day06_flashattention_notes.md
learning/day06_online_softmax.ipynb
```

验收：能解释“省的是中间 score 的读写和存储”，而不是简单说“用了 GPU 所以更快”。

### Day 7：vLLM 机制地图

本仓库没有生产 vLLM 实现，今天只建立机制映射：

```text
请求 → Scheduler → Prefill/Decode → KV Block → Attention → 输出 token
```

任务：

- 对比连续 KV Cache 和 Paged KV Cache。
- 画逻辑 token position 到物理 block 的映射。
- 说明 Static Batch 和 Continuous Batching。
- 标出仓库 `KVCache` 与生产服务之间缺少的模块。

产出：

```text
docs/DAY07_VLLM_MAP.md
learning/day07_vllm_map.ipynb
```

## Day 8–14：Mini Transformer

统一项目目录：

```text
learning/mini_transformer/
  __init__.py
  config.py
  norm.py
  rope.py
  attention.py
  kv_cache.py
  model.py
  benchmark.py
  tests/
```

### Day 8：RMSNorm + RoPE

- 先用 FP32。
- 固定随机种子。
- 与仓库函数对齐。
- 测试偶数 head dimension、offset 和边界序列长度。

### Day 9：单头 Attention

- Q/K/V 投影。
- Scaled Dot-Product。
- Causal Mask。
- 验证未来 token 不影响前缀 logits。

### Day 10：MHA

- `B,T,E → B,H,T,D`。
- 检查 transpose 后的 stride。
- 与 PyTorch 参考实现对齐。

### Day 11：GQA

- 支持 `n_kv_groups=1/2/n_heads`。
- 统一测试 MQA、GQA、MHA。
- 记录 Cache head 数量和存储比例。

### Day 12：KV Cache

- Prefill。
- 单 token Decode。
- Cache reset。
- RoPE position offset。
- 完整前向与 Cache 输出对齐。

### Day 13：Transformer Block

- RMSNorm。
- Attention residual。
- SwiGLU residual。
- 多层堆叠。

### Day 14：统一验收

必须同时通过：

- 单元测试。
- 固定输入数值对齐。
- Cache 对齐。
- 小 Shape Benchmark。

## Day 15–21：CUDA/Triton

目录：

```text
learning/kernels/
  reference.py
  triton_vector_add.py
  triton_rmsnorm.py
  triton_softmax.py
  triton_swiglu.py
  triton_attention.py
  tests/
  benchmark.py
```

### Day 15：CUDA 执行模型

产出 Thread/Block/Grid 图和 Vector Add 边界测试。

### Day 16：Reduction

实现 Sum/Max，比较朴素、Shared Memory 和 Warp Reduction。

### Day 17：Triton Vector Add + RMSNorm

必须包含：

- Mask tail。
- 随机 shape。
- FP32 数值对齐。
- CUDA 同步后的计时。

### Day 18：Softmax + LayerNorm

重点观察 row-wise reduction、数值稳定性和长行边界。

### Day 19：MatMul/Tiling

只改变一个 tile 参数，记录：

```text
shape、tile、耗时、吞吐、误差、显存
```

### Day 20：教学版 Causal Attention

先正确，再讨论融合；不能把调用 SDPA 当作手写 kernel。

### Day 21：Online Softmax/FlashAttention

对比：

- 完整 score 矩阵。
- 分块 score。
- Online Softmax。
- PyTorch SDPA。

## Day 22–30：Mini vLLM

目录：

```text
learning/mini_vllm/
  paged_kv_cache.py
  scheduler.py
  batch_engine.py
  requests.py
  benchmark.py
  tests/
  README.md
```

### Day 22：真实推理框架边界

核对 ARM64/GB10 兼容性；如果安装失败，记录失败原因，不覆盖已验证 Spark 镜像。

### Day 23：Paged KV Cache 设计

先画数据结构：

```text
free_blocks
block_table[request_id]
logical_position → (block_id, offset)
```

### Day 24：Continuous Batching 状态机

```text
WAITING → PREFILL → RUNNING → FINISHED
                 ↘ ABORTED
```

### Day 25：实现 Paged Cache 模拟器

测试：

- 分配。
- 释放。
- 不同序列长度。
- Block 不连续。
- Cache 满和回收。

### Day 26：实现 Scheduler

支持等待队列、运行队列、完成回收和 token budget。

### Day 27：接入小模型或假执行器

明确区分：

- 调度模拟指标。
- 真实模型 GPU 指标。

### Day 28：选一个优化

只选一个：Prefix Cache、量化、Speculative Decoding、CUDA Graph 或 Tensor Parallel。

### Day 29：整理 README 和报告

报告必须包含问题、设计、测试、Benchmark、失败原因和限制。

### Day 30：最终答辩

回答：

1. GQA 为什么减少 Cache？
2. Paged Cache 解决什么碎片问题？
3. Continuous Batching 为什么提升利用率？
4. 哪个模块是瓶颈，证据是什么？
5. 哪些是实测，哪些是模拟？

## 统一运行约定

每个阶段都执行：

```bash
git status --short
git diff --check
```

Spark 上的 Notebook：

```bash
docker compose -f docker-compose.spark.yml up -d
```

只提交源码、测试、Notebook 和报告；模型、日志、checkpoint 保持在 Docker volume。
