# Mini Transformer

Day 8–14 的独立参考实现，入口 `model.py`；接口保持简洁，便于调试，不修改书籍主线模型。

| 组件 | 实现/职责 |
| --- | --- |
| `Config` | 明确 E/Hq/Hkv/D/layers/context，验证可分组与偶数 head_dim |
| `RMSNorm` | FP32 reduction，恢复输入 dtype |
| `rope` | split-halves 位置旋转、显式 offset |
| `Attention` | MHA/MQA/GQA、紧凑缓存、绝对位置矩形 Mask |
| `FFN` | gate/up/down + SiLU |
| `Cache` | 请求级状态，length 从第一层 K 推导 |
| `Model` | Embedding/Blocks/Norm/Head，超长先拒绝，greedy generation |

在 Jupyter Terminal：

```bash
cd /workspace
python -m learning.curriculum --day 12 --device cpu --output logs/day12.json
python -m pytest -q tests/test_curriculum.py
```

这里无 QK Norm（与 Qwen3 的一个结构差异），参数不对应预训练 Qwen3 checkpoint。
验证单个算子与仓库对齐，Attention 与 SDPA 对齐，同模型 Cache 与 full forward 对齐；
没有宣称整个独立随机模型与不同结构的 Qwen3 logits 相同。

作业顺序：先自己写 RMSNorm/RoPE，再单头/MHA/GQA、Cache、Block、测试和计时。
参考实现不等于作业完成。训练可做小 loss backward，不提供成熟训练器。
