# 推理优化、系统、部署与工程实践：五方向衔接

当前个人学习位置统一见 [进度记录](LEARNING_PROGRESS.md)。已有参考实现与通过记录，不能自动换成个人
项目经历或生产部署能力。每天知识和面试题见 [索引](KNOWLEDGE_INDEX.md)。

## 1. 五个方向分别解决什么

```text
模型数学：该计算什么，怎样复用历史状态？
    ↓
GPU算子：同样的计算怎样减少IO/launch等开销？
    ↓
推理系统：多个请求怎样分配Cache、调度和回收？
    ↓
服务部署：模型怎样变成可调用、可观测、可控的服务？
    ↓
工程验证：怎样证明正确、性能有效、异常可恢复？
```

这是学习依赖图，不意味着生产所有请求必须按五道串行步骤运行。

| 方向 | 当前具体材料 | 已验证的范围 | 不能据此声称 |
| --- | --- | --- | --- |
| 模型推理优化 | GQA、Cache、Prefill/Decode、Online Softmax、PrefixStore | 固定输入对齐、状态形状、前缀隔离 | 所有模型/精度/长文本质量已验证 |
| 算子优化 | CUDA/Triton、Norm、Softmax、MatMul、融合Attention、IR | 实际GPU forward、边界与数值、部分同步计时 | 生产库全功能、所有Shape加速 |
| 推理系统 | PagedPool、scheduler、真实小模型分桶Engine | OOM/回收/所有权、独立greedy对齐、真实batch trace | 生产paged GPU kernel、抢占/取消/多卡齐备 |
| 部署基础 | Spark Docker、GPU自检、Jupyter、代理、volume | 本机教学环境、checkpoint、Notebook运行 | 真实vLLM部署与公网服务已完成 |
| 工程实践 | pytest、oracle、负例、Benchmark、日志、报告、备份同步 | 当前套件和明确记录的实验 | 压测、安全、可用性、回滚等生产验收全覆盖 |

## 2. 接下来按什么顺序实践

### A. 先把数学和状态对齐（Day 6–14）

入口：

- [Online Softmax 手算](DAY06_ONLINE_SOFTMAX.md) 与
  `learning/day06_online_softmax.ipynb`。
- [Mini Transformer](../learning/mini_transformer/README.md)。
- Cache、GQA 的具体公式与内存，见 [Day 5](DAY05_ARCHITECTURES.md)。

验收：

- 同权重 full/cached 对齐，未来输入不影响前缀。
- 多步Decode、不同split、不同batch、Cache重置/拒绝异常。
- 不用采样文本代替 logits；不混 FP32严格对齐与 BF16观察结论。

### B. 再做 GPU 算子和编译观察（Day 15–21）

入口：

- [算子契约和编译](../learning/kernels/README.md)。
- [Triton IR](TRITON_IR.md)，`learning/day17_triton_ir.ipynb`。
- 每日 `day17_course.ipynb` 至 `day21_course.ipynb`。

实验闭环：

```text
先写reference → 数值/边界测试 → GPU执行 → IR核查 → 同步计时 → profiler归因
```

只改变一个关键变量，例如 tile 或 BLOCK。IR检查不是profiling；
kernel更短、编译成功都不等于性能提升。

### C. 把请求与模型接起来（Day 23–28）

入口：[教学引擎](../learning/mini_vllm/README.md)。

验收：

- Paged pool 分配原子性、释放后句柄失效、无双重释放/泄漏。
- 每个请求独立Cache，分桶后真正调用batch模型。
- 结果与独立greedy对齐，异常finally清理。
- Prefix snapshot clone隔离，权重变化后失效。
- 模拟round和真实同步wall-clock分开。

### D. 真实服务单独部署并验收（Day 22，外部依赖）

当前只准备了 [vLLM接口探针](../learning/vllm_probe.py)，服务端尚未部署验证。
先保持以下顺序，不覆盖Spark已验证的torch环境：

1. 确认目标框架版本对ARM64/GB10/驱动的兼容性，独立容器固定版本。
2. 用框架支持的模型配置/tokenizer/权重格式启动，不能直接传任意自定义`.pth`。
3. 做单请求和错误请求校验，再流式响应，再并发。
4. 验证最大prompt/输出预算、队列容量、超时、取消、资源回收。
5. 最后做持续压测、指标监控、鉴权/限流、进程退出和重启。

本月已准备的探针仅覆盖 `/v1/models` 与 non-streaming `/v1/completions`。
没有鉴权、SSE/取消、限流、健康恢复、监控告警和并发压测的完整实现，
这些保持为**待补的生产工程实践**，不因学习材料齐全而标记完成。

### E. 对“优化有效”给出证据（Day 14/27/29/30）

至少记录：

| 指标 | 口径必须说明 |
| --- | --- |
| TTFT | 首token，是否含排队/网络/prefill |
| TPOT/ITL | 后续token间隔，采样与聚合口径 |
| 吞吐 | output tokens/s还是total tokens/s，是否包含输入 |
| 延迟分位 | p50/p95/p99，样本数和失败请求 |
| 内存 | allocated/reserved/系统内存、权重/Cache/workspace |
| 环境 | 模型版本、dtype、shape、并发、seed、框架/驱动 |
| 成本 | 功耗/时间/资源，数据是否可比 |

non-streaming总时间不是TTFT；小随机模型三请求实验不是正式容量评估。
固定工作负载、先证明正确再比较性能，慢于基线也如实写原因。

## 3. 当前能做与不能做的清单

**现在可以直接做：**

- 在现有Jupyter按 [统一进度](LEARNING_PROGRESS.md) 继续学习，不因翻页重建环境。
- 用已准备的参考实现做数值、Cache、分页与调度实验。
- 在Spark跑CUDA/Triton教学kernel并查看真实IR。
- 回答面试主问/追问，再用实验验证答案。

**需要额外准备或开发：**

- 真实vLLM服务及支持的预训练模型格式/依赖。
- SSE、取消、超时、排队/限流、鉴权、监控、服务恢复。
- 量化/Speculative/CUDA Graph/多卡等独立优化方向。
- 正式质量评测、持续并发压测与生产SLO。

个人进度见 [学习记录](LEARNING_PROGRESS.md)；参考实现验证见
[课程验证](COURSE_VALIDATION.md)，不要把“代码存在”替代“已部署与验证”。
