# 本地实战：从看懂概念到自己交付代码

用户已确认概念课程全部阅读，现进入本地实战。
“已读完”是用户学习记录，不自动等于所有模块已独立实现或性能验收。
当前第一关为 **P1-01：RMSNorm/RoPE，自己的代码通过CPU与CUDA验收**。
交互入口：[P1-01 Notebook](../learning/practice01_start.ipynb)。

## 1. 先做一个最小闭环

```text
你编写my_ops.py
    ↓ 验收器加载你的文件，不是只测参考实现
CPU FP32：数值、边界、梯度、offset
    ↓ 通过后
CUDA FP32：真实GPU执行，不静默改用CPU
    ↓
保存报告、源码hash与一次错误修复记录
```

不需要重建Docker、不下载大模型、不重新读30天概念。
现有 `learning/mini_transformer`、`kernels`、`mini_vllm` 保留为参考，
你自己的实现放在单独文件，不覆盖课程参考。
第一关不追求更快，先正确；P1-01报告不包含性能结果。

## 2. 现在在本机怎么开始

在现有Jupyter的Terminal里：

```bash
cd /workspace
python -m learning.local_practice --device cpu
python -m learning.local_practice --device cuda
```

未传candidate时，这是 **reference_baseline**，只确认本机参考和验收链可跑。
报告自动生成新UTC文件名，保存在 `logs/practice/`，不会覆盖旧结果。
不要把参考baseline通过写成你已经独立实现。

在宿主机终端执行等价命令：

```bash
docker compose -f docker-compose.spark.yml exec -T reasoning \
  /opt/venv/bin/python -m learning.local_practice --device cpu
```

GPU验证把最后的cpu改为cuda即可。CUDA不可用会明确失败，保留错误报告，不改用CPU掩盖问题。
无需停止现有Jupyter、删除volume或升级torch。

## 3. P1-01：今天你真正要写的代码

在当前工作区编写 `learning/my_ops.py`，导出以下两个接口：
实现思路和每个TODO的中文说明见 [my_ops实现指导](MY_OPS_IMPLEMENTATION.md)，
另有 [注释模板](../learning/my_ops_template.py)，不会覆盖你已有的my_ops.py。

| 接口 | 输入/输出契约 |
| --- | --- |
| `RMSNorm(width, eps=1e-6)` | `nn.Module`，可学习参数名为`weight`、shape `[width]`，输入输出同shape/dtype/device |
| `rope(x, start=0, theta=10000.0)` | `[B,H,T,D]`，split-halves配对、保持shape/dtype/device；奇数D、负start抛ValueError |

第一版支持FP32，内部归约先按FP32；不要直接调用参考实现冒充独立实现。
RMSNorm不会推导时先看 [目标→公式→代码的逐步推导](DAY08_RMSNORM_ROPE.md#rmsnorm-derivation)，
其中附原论文、PyTorch官方API与rsqrt链接，以及RMS归一化和L2归一化的区别。
验收器能确认文件被加载和本次数值通过，不能证明作者是否独立编写。
只加载你信任的本地文件，candidate导入会执行该文件里的Python代码。

你的测试命令：

```bash
python -m learning.local_practice --device cpu --candidate learning/my_ops.py
python -m learning.local_practice --device cuda --candidate learning/my_ops.py
```

如果Notebook里改文件后担心旧import缓存，使用上述新进程命令重跑。
candidate_file代表加载了该路径，不是“已经精通”或“有独立创作证明”。

### 14个验收项

```text
RMSNorm：
  随机、全零、常数、大值、小值（使用非全1的gamma）
  不混其他token、输入与gamma梯度有限

RoPE：
  offset=0/5/17与独立FP64逐对公式对齐
  正确offset分块与整段一致
  二范数保持
  奇数D拒绝、负offset拒绝
```

遇到失败先读具体case.name/error，不要第一反应放宽阈值。
本关不覆盖任意布局、所有非法输入、低精度GPU性能或生产backward。
低精度扩展练习完成后可对照 [Day 8 Notebook](../learning/day08_rmsnorm_rope.ipynb)。

## 4. 不断练习：每一关都跑这六步

| 步骤 | 你要做什么 | 必须留下什么 |
| --- | --- | --- |
| 预测 | 不运行代码，写shape、公式和输出 | 一段自己的推导 |
| 实现 | 先遮住参考，在自己的文件写最小版本 | 源码路径与Git diff |
| 对齐 | 独立oracle、边界、CPU先后GPU | 原始报告及误差 |
| 改错 | 故意改错一个轴、offset、Mask或释放位置 | 一个可复现失败及修复 |
| 泛化 | 只改一个维度/幅值/dtype/请求组合 | 新输入上的验证 |
| 复述 | 解释正确性、代价、当前不支持什么 | 3分钟口述提纲 |

每完成一个小模块就做这个循环，不是一次Run All把21天打勾。
建议每轮只选一个能独立验证的问题，下一轮从上次失败继续。
如果某关已经能独立实现，可直接交对应证据进入下一关，不要求重复阅读概念课。

## 5. 后续实战关卡

下面是实践顺序，不是自动完成清单。除P1-01自写模块入口外，
现有课程测试主要验证参考代码；要验收你自己的后续实现，先把对应测试明确接到你的模块，
不能只运行参考测试就宣称个人实现通过。

| 关卡 | 亲手任务 | 第一组验收/故障注入 | 对照资料 |
| --- | --- | --- | --- |
| P1-01 | RMSNorm/RoPE | 14项CPU→CUDA；错offset | [Day 8](DAY08_RMSNORM_ROPE.md) |
| P1-02 | 单头Attention | 未来token扰动前缀；反转Mask应失败 | [Day 9](DAY09_SINGLE_ATTENTION.md) |
| P1-03 | MHA拆头/合头 | T>1标签追踪；漏transpose | [Day 10](DAY10_MULTIHEAD.md) |
| P1-04 | MQA/GQA参数化 | Hkv=1/2/4各自full/cached对齐 | [Day 11](DAY11_GQA_MQA.md) |
| P1-05 | 请求级KV Cache | 3+2+3、多步增长、上下文拒绝不污染 | [Day 12](DAY12_REQUEST_CACHE.md) |
| P1-06 | Block/SwiGLU/LM输出 | 两次残差、loss/backward、全部梯度有限 | [Day 13](DAY13_BLOCK_BACKWARD.md) |
| P1-07 | 模型基线 | 相同工作量、预热/同步、完整logits误差 | [Day 14](DAY14_MODEL_BASELINE.md) |
| P2-01 | CUDA Vector Add | N=255/256/257；尾部越界检查 | [Day 15](DAY15_CUDA_INDEXING.md) |
| P2-02 | Shared Reduction | 无效线程仍同步；partial独立参考 | [Day 16](DAY16_SHARED_REDUCTION.md) |
| P2-03 | Triton Add/RMSNorm | 1003尾部、非全1gamma、拒绝错误layout | [Day 17](DAY17_TRITON_ROWS.md) |
| P2-04 | Softmax/LayerNorm | 大分数、常数行、错误padding负例 | [Day 18](DAY18_STABLE_ROW_OPS.md) |
| P2-05 | Tiled MatMul | M/N/K非整齐、tile16/32/64同oracle | [Day 19](DAY19_TILED_MATMUL.md) |
| P2-06 | 分块/融合Attention | 支持shape误差、因果性、真实同步计时 | [Day 20](DAY20_ATTENTION_ORACLE.md)、[21](DAY21_FUSED_KERNELS.md) |
| P3-01 | KV页池 | 非连续页读回、逻辑/物理映射 | [Day 23](DAY23_PAGED_ADDRESSING.md) |
| P3-02 | 调度器 | arrival/budget/公平与工作守恒 | [Day 24](DAY24_SCHEDULER_FAIRNESS.md) |
| P3-03 | OOM/异常回收 | 旧handle拒绝、分配失败不破坏状态 | [Day 25](DAY25_OWNERSHIP_OOM.md) |
| P3-04 | 真实小模型合批 | 独立greedy对齐、真实Batch trace、回收 | [Day 26](DAY26_BATCHED_ENGINE.md) |
| P3-05 | 性能对照 | 同prompt/输出预算、尾延迟与吞吐分开 | [Day 27](DAY27_METRICS_BENCHMARK.md) |
| P3-06 | Prefix隔离 | miss/hit、续写不改原稿、权重变更失效 | [Day 28](DAY28_PREFIX_CACHE.md) |
| P4-01 | 隔离真实服务 | 模型格式、单请求、授权、日志/回滚 | [Day 22](DAY22_SERVING_BOUNDARY.md) |
| P4-02 | 复现报告/答辩 | 一个真实瓶颈、单变量优化、失败与限制 | [Day 29](DAY29_REPRODUCIBLE_REPORT.md)、[30](DAY30_CAPSTONE.md) |

P4不自动安装vLLM、不覆盖现有torch；此前真实服务未配置，继续保留该状态。
GPU kernel更快、模型质量更好、服务更稳定是不同结论，各自拿证据。

## 6. 优化循环

```text
正确性全过
  → 固定输入/权重/dtype/设备
  → 预热、同步计时、收集内存/调用轨迹
  → 找最大耗时项
  → 只改一个原因
  → 重新跑正确性与同负载计时
  → 保留更慢/失败结果
```

P1-01不要急着测ns级小算子的“加速比”；先交出CPU与CUDA正确性证据。
进入P2/P3再复用 `learning/bench.py`，明确同步wall-clock与kernel-only不同。

## 7. 个人记录

本次实战开始和待填记录见 [实战日志](LOCAL_PRACTICE_LOG.md)。
示例报告只是环境/参考/候选实现证据，不自动更新个人掌握程度。
下一步最具体的动作：**完成自己的 `learning/my_ops.py`，先跑CPU验收，带着第一个失败项修改。**
