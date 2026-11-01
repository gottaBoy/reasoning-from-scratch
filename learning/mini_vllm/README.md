# Mini 推理引擎：教学实现，不是外部 vLLM

核心代码：

- `paged_kv_cache.py`：固定块、请求 block table、所有权、OOM 和回收。
- `scheduler.py`：sequential/static/continuous 的离散轮次模拟。
- `engine.py`：真正调用小 Transformer 的同长度分桶合批、Cache 拆回请求。
- `PrefixStore`：完整 token 前缀 snapshot，clone 隔离续写。

```bash
python -m learning.curriculum --day 30 --device cuda --output logs/day30.json
python -m pytest -q tests/test_curriculum.py
```

## 实际支持

每请求先预留最大预算的 KV blocks；每轮准入新请求；
同 Cache position、本轮 input length 的请求合并 batch。
分页数据显式 gather 成连续 K/V，模型按普通 Attention 计算，再写回新 token。
相同 prompt/budget 的输出与独立 greedy 完整对齐，异常时 finally 回收。

## 不支持或不承诺

- 不含 paged GPU Attention kernel、生产 vLLM scheduler、异长 ragged batch。
- 不含 HTTP server、用户取消、抢占/offload、多卡或 quantization。
- PrefixStore 不是 COW；权重更新后必须 clear，不支持跨模型共享。
- 只展示 exact 完整前缀，不支持自动最长前缀搜索/驱逐。
- 输入为小随机模型 ID，不是语言质量评测。
- 模拟 round 不是 GPU 时间；真实 engine 的同步 wall-clock 单独记录。
- 单线程教学所有权不等于线程安全/多租户隔离方案。

## 报告模板（Day 29–30）

```text
目标与假设：
代码基线/本地改动：
硬件、驱动、torch/triton：
随机 seed：
模型来源（本项目默认随机权重）：
输入 token IDs / 输出预算 / 到达轮：
Batch 和 Cache 配置：
dtype、计算精度：
正确性 oracle / atol / rtol：
测试通过/跳过/失败：
warm-up / 重复次数 / 同步方式：
prefill / decode / 总耗时：
峰值 allocated / 系统内存：
与基线的差异：
失败实验和限制：
下一个单变量实验：
```

参考第三周与第四周知识讲义中的口试，不把 API 冒烟成功、教学分页或一个
3 请求短测称作“实现了生产 vLLM 并完成性能验证”。
