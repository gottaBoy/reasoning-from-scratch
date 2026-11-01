# 每日关键知识与面试索引

个人进度：[Day 5](LEARNING_PROGRESS.md)。以下是学习材料，不是已通过的能力证书。
跨天模拟面试、场景题与自评分见 [面试训练](INTERVIEW_PRACTICE.md)。
按五个工程方向继续实践见 [推理优化与工程路线](INFERENCE_ENGINEERING_PATH.md)。
Triton编译术语见 [IR讲义与真实产物实验](TRITON_IR.md)。

| 学习日 | 主线 | 知识讲义 | 实验 |
| --- | --- | --- | --- |
| 1–7 | 环境、模型、GQA、Cache、MoE/MLA、FlashAttention、调度 | [第一周](KNOWLEDGE_DAY01_07.md) | `learning/day03*`、`day04*`、`day05*`、`day06_course.ipynb`、`day07_course.ipynb` |
| 8–14 | 独立实现并验证 Transformer | [第二周](KNOWLEDGE_DAY08_14.md) | `learning/day08_course.ipynb` 至 `day14_course.ipynb` |
| 15–21 | CUDA/Triton、访存、融合、计时 | [第三周](KNOWLEDGE_DAY15_21.md) | `learning/day15_course.ipynb` 至 `day21_course.ipynb` |
| 22–30 | 真实框架边界、分页、调度、Prefix Cache、交付 | [第四周](KNOWLEDGE_DAY22_30.md) | `learning/day22_course.ipynb` 至 `day30_course.ipynb` |

## 每天如何使用

Day 5 配置概念还没稳固时，先做
[字段、矩阵和内存实验](../learning/day05_config_memory.ipynb)，再从
[Day 6 两块小数字](DAY06_ONLINE_SOFTMAX.md) 进入后续课程，
不要一开始就背 Online Softmax 递推式。

1. 读“概念”：用自己的话复述，不能只念英文全称。
2. 看“公式/直觉”：明确每个符号、张量 shape、设备和 dtype。
3. 做“实验”：先预测结果，再运行，记录失败与修复。
4. 遮住答案回答“面试主问”，再回答追问。
5. 看“误区”：区分数学等价、浮点近似和实现限制。
6. 完成验收后再标记自己的进度，不因 Notebook 的参考输出自动打勾。

## 四级掌握标准

| 级别 | 你需要做到什么 | 不能拿什么替代 |
| --- | --- | --- |
| L1 解释 | 不看代码说明问题、方案、代价 | 背缩写 |
| L2 推导 | 给定配置推导 shape、内存、关键公式 | 只记某模型的固定数字 |
| L3 实现验证 | 写最小实现、数值对齐、边界和负例 | 复制参考答案后打印 passed |
| L4 权衡 | 用 profiling/实验说明适用条件和失败边界 | 泛泛声称“更快、更省” |

“精通”意味着面对新模型、新输入和异常仍能分析与验证，不是 30 天看完所有文件。

## 面试回答的固定结构

```text
问题是什么 → 核心原理 → shape/公式 → 实现关键点 → 成本与限制 → 我的实验数据
```

例如回答 KV Cache：

> 因果 decoder 中历史位置不依赖未来，所以每层历史 K/V 可以复用。
> 新 token 的 Q 仍需与全部可见 K 比较，不能说 decode 不再计算 Attention。
> 普通 GQA 的缓存约为 2×L×B×T×Hkv×D×dtype_bytes。
> 代价是随上下文线性增长的状态及读取带宽；分页解决分配问题，FlashAttention
> 解决中间 score 的 IO 问题，两者不是相同优化。
> 我用同一权重验证 full forward 和 prefill+decode，并单独报告 FP32/BF16 误差。

## 常混概念

| 不应混为一谈 | 区别 |
| --- | --- |
| Token / 单词 / 字符 | tokenizer 的编码单位，不保证一一对应 |
| Head dimension / residual width | 可通过投影不同；不能总假设 `D=E/H` |
| FFN gate / MoE router | 前者逐通道连续调节，后者选择专家及混合权重 |
| KV Cache / 参数权重 | 前者随请求和序列增长，后者主要随模型结构变化 |
| RoPE / causal Mask | 前者改变 Q/K 的位置相位，后者禁止读取未来 |
| GQA / MLA | 前者共享 KV heads，后者改变 KV 表示及缓存计算路径 |
| FlashAttention / PagedAttention | 前者 IO-aware 分块融合，后者使用分页 KV 的读取/管理 |
| Continuous batching / 普通动态组批 | 前者可在迭代边界更新运行请求集合，后者常在请求开始前拼批 |
| TTFT / TPOT / tokens/s | 首 token 延迟、后续 token 间隔、吞吐，不可互相替代 |
| CUDA stream / Python 线程 | GPU 排队执行与 CPU 并发不是同一层 |
| GPU 可见 / kernel 可运行 | `nvidia-smi` 成功不保证 wheel/扩展支持目标架构 |
| API 冒烟 / 推理系统压测 | 简单调用成功不验证调度公平、内存边界或性能 |
| 教学模拟 / 实测 | round 数不等于秒，假执行器不等于 GPU 合批 |

## 每周口试

- 第 1 周：画出输入到 logits，再指出 GQA/Cache/RoPE/MoE/MLA 所在位置。
- 第 2 周：给出 bug：Mask 方向反、Cache offset 错或混用请求，设计负例定位。
- 第 3 周：算子慢于 PyTorch 时，解释 launch、带宽、计算、occupancy 各如何检查。
- 第 4 周：请求突然增多或取消时，说明准入、预算、公平性、回收和错误处理。

记录证据的模板见 [实战报告](../learning/mini_vllm/README.md)，自动测试与运行边界
见 [课程总入口](COURSE_DAY06_30.md)。
