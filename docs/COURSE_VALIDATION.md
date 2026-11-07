# Day 6–30 材料与参考实现验证

验证日期：2026-10-31（America/Los_Angeles）；日志 UTC 日期为 2026-11-01。
代码基线 `2c6e047` 加当时尚未提交的课程修改。**首轮验证时个人进度为Day 5。**
下文保留各次验证时的历史状态；当前学习位置见 [进度记录](LEARNING_PROGRESS.md)。

## 结果

| 范围 | 结果 | 说明 |
| --- | --- | --- |
| Day 6–30 共 25 个新 Notebook | 全部执行到结束 | 不等于所有外部依赖项通过 |
| 其中参考实验 | 24 个 `passed` | 包含概念模拟、模型数值对齐、真实 GPU 算子与引擎 |
| Day 22 | `external_service_not_configured` | 没有部署/调用真实 vLLM 服务，不计通过 |
| 首轮课程全仓库普通测试 | **261 passed / 27 skipped / 1 warning** | 非 expensive/真实全量下载套件 |
| 补充配置/内存实验后的普通测试 | **269 passed / 27 skipped / 1 warning** | 新增 8 项配置算术/参数核对测试 |
| 补充Triton IR后的普通测试 | **274 passed / 27 skipped / 1 warning** | 新增 5 项导出/CPU拒绝/真实GPU编译产物测试 |
| 补充MLA代数实验后的普通测试 | **282 passed / 27 skipped / 1 warning** | 新增 8 项展开/吸收对齐、状态计数与非法维度测试 |
| 补充Day 7调度与真实页池后的普通测试 | **301 passed / 27 skipped / 1 warning** | 现有Spark工作区使用缓存MATH-500数据重跑；19项Day 7测试 |
| CUDA C++ 独立程序 | Vector Add / Shared Memory Reduction 通过 | 6 组输入大小，宿主机 nvcc 编译 |
| Python/Ruff | 通过 | Notebook 路径初始化的 E402 单独排除，其他检查保留 |
| Notebook schema/代码 | 生成器和执行检查通过 | 25 份新文件，不含未执行的个人练习答案 |

环境为现有 DGX Spark / GB10，PyTorch `2.10.0+cu130`，Triton `3.6.0`，
CUDA Toolkit 13.0。保留 PyTorch 对 `sm_121` capability 的提示；
这里的实际 kernel 都已运行，但不推断所有第三方扩展兼容。

后续已将提交 `87428f3` 的课程构建进独立镜像，并在全新 volume 中重跑完整 smoke、
普通测试和课程 Notebook；见 [课程镜像验收记录](IMAGE_VALIDATION.md)。
以下保留原学习卷上的历史验证，不混用两次运行的日志。

## 实测项目

- Online Softmax 的大幅值数值稳定和分块 Attention 与 SDPA 对齐。
- 独立 RMSNorm/RoPE 与书籍实现对齐。
- MHA/MQA/GQA、因果前缀不变、完整/缓存多步对齐。
- Block backward、所有参数有限梯度、超长输入不破坏旧 Cache。
- Triton Add/RMSNorm/Softmax/LayerNorm/SwiGLU/MatMul/Online Attention。
- FP32/FP16/BF16 kernel 多种大小与尾部 Mask，对照独立参考。
- 分页池非连续映射、OOM 原子性、回收、旧 handle 失效。
- 真实小模型的分桶 batch、greedy 对齐、异常清理。
- 精确前缀 snapshot 命中、clone 隔离与 continuation 对齐。
- 模拟 scheduler 的到达、预算、顺序和完成轮次。
- vLLM 探针的 HTTP payload 只用 mock 测试，不冒充服务端实测。

MatMul 全仓库回归曾发现旧测试启用 TF32 的全局设置影响 reference：
现改为独立 CPU FP64 oracle 后按 FP32 误差阈值检查，未用放宽阈值掩盖差异。

## Day 5–6 的补充验证

以下两份 Notebook 已另行执行全部代码单元格，不改变原 25 份课程的计数：

- `learning/day05_config_memory.ipynb`：投影参数形状、Hkv/F 单变量实验、
  T 翻倍时权重/Cache/score 的 1/2/4 倍关系。
- `learning/day06_online_softmax.ipynb`：错误的块内归一化平均、正确 m/l/o 合并、
  块宽变化、大幅值数值稳定，以及 PyTorch 分块 Attention 对齐 SDPA。

固定 Qwen3 配置且 B=1、T=4096、每元素2字节时，计算出的全部层 KV 为448 MiB，
单层朴素 score 为512 MiB；只是公式估算，不是完整进程实测峰值。
手算示例的正确输出约34.9265，错误的块内输出平均约27.3106。
配置工具不加载权重、不分配长序列张量，并以真实小模块核对 projection 参数。

## Triton IR补充验证

2026-11-01 UTC，在现有Spark Jupyter容器的Python GPU内核完整执行
`learning/day17_triton_ir.ipynb`，没有修改运行依赖：

- Triton3.6.0；数值对齐后导出真实TTIR/TTGIR/LLVM IR/PTX/cubin。
- TTGIR目标`cuda:121`，本次PTX为`.version 8.8`、`.target sm_121a`。
- Vector Add的N=1003/BLOCK=256，num_warps=4；报告16寄存器/线程、0 shared bytes。
- 5项专项测试通过，覆盖1/1003/1025元素、不同BLOCK、尾部Mask、拒绝覆盖旧目录。
- 这些产物不等于性能加速证据；本实验未给出加速比。

已执行Notebook：

```text
/workspace/logs/day17-triton-ir-executed-20261101T094814000699Z.ipynb
```

实际编译产物和报告：

```text
/workspace/logs/triton-ir-20261101T094815307170Z/
```

新增 [五方向实践路线](INFERENCE_ENGINEERING_PATH.md)，标明模型优化、算子、系统、
部署与工程验证分别的入口与限制。真实vLLM服务仍未配置，不因IR实验完成而改变其状态。

## Day 5 MLA补充验证

2026-11-01，在现有Spark Jupyter容器完整执行
`learning/day05_mla_intuition.ipynb`，不下载模型，不修改运行依赖：

- 同一组随机低秩参数，两条路径分别展开历史K/V、直接在latent上计算。
- 含独立RoPE分支；比较最终位置的分数、softmax权重与Attention输出。
- 默认CPU FP32配置：分数最大误差约`2.61e-8`，输出最大误差约`8.73e-9`。
- 7个历史位置：物化完整各头K/V为1008字节，latent加位置Key为140字节。
  前者重复拼接各头位置Key；仅为教学数组计数，不是实际DeepSeek压缩率。
- 8项专项测试通过；最终全仓库普通回归为282 passed / 27 skipped / 1 warning。
  保留GB10 capability提示；Ruff、Notebook schema和配套本地链接检查通过。

已执行Notebook：

```text
/workspace/logs/day05-mla-executed-20261101T104038254204Z.ipynb
```

本实验没有实现完整DeepSeek模型、训练质量评估、生产KV更新或性能加速对比。
“等价”仅指同一套低秩参数的两种计算顺序，不保证任意普通MHA无损改为MLA。
教学数据流与参考答案见 [MLA直观讲义](MLA_INTUITION.md)；个人进度仍为Day 5。

## 统一符号与Day 6形状补充验证

2026-11-07 UTC，新增 [课程符号规范](NOTATION.md)，并在现有Spark Jupyter容器
完整执行更新后的 `learning/day06_online_softmax.ipynb`，6个代码单元格通过：

- 新增CPU形状实验，区分 `T_q/T_kv` 与 `D_k/D_v`，使用末两轴转置。
- Prefill分数为 `[1,2,3,3]`，单token Decode形状示例为 `[1,2,1,9]`。
- 检查softmax沿Key轴的行和，并与PyTorch SDPA输出数值对齐。
- 原有两块手算、不同分块宽度、大分数稳定性及因果Attention对齐继续通过。
- Notebook schema、Ruff与修改文档的本地链接检查通过。

形状实验使用随机MHA张量，不加载模型、不验证生产Cache更新，也不测试GPU速度。
本次修改仅为说明与Notebook，不重跑全仓库测试；上文282项通过为此前回归记录。
Notebook内核仍提示本机TCP传输未加密，未将Jupyter暴露到公网。

已执行Notebook保存在：

```text
/workspace/logs/day06-notation-executed-20261107T090015681879Z.ipynb
```

## Day 6收益、累积量与发音补充验证

2026-11-07 UTC，扩充同一份Notebook并在现有Spark Jupyter完整执行，
8个代码单元格通过，旧文件及先前执行记录保留：

- 新增二维Value，流式输出为 `[34.9265273458577,134.9265273458577]`，
  与完整softmax加权结果对齐。
- 检查自然指数定义下 `m+ln(l)` 与 `torch.logsumexp` 一致。
- 只用整数算术核对单层FP16分数数组512/2048/8192 MiB；
  单行FP32累积状态520 bytes，示意tile为32 KiB分数与32.5 KiB状态。
- 现有6项Online Softmax/分块Attention专项测试通过，10项其他课程测试未选择。
- Notebook schema、Ruff、修改文档本地链接及 `git diff --check` 通过。
- 统一规范补充专业变量角色名、希腊字母及数学运算符的常用读法、完整公式朗读；
  英语读音为文本提示，不是生成或验证过的语音文件。

已执行Notebook：

```text
/workspace/logs/day06-flashattention-benefits-executed-20261107T093649300005Z.ipynb
```

本次不修改算子实现或依赖，没有重跑全仓库测试、生产FlashAttention backward、
GPU性能或峰值显存测试。上述内存数字是逻辑数组计数，不是实测资源占用。

## Day 5投影、拆头与两次残差补充验证

2026-11-07 UTC，更新符号规范的分叉图，区分本仓库Block中先后执行的Attention/FFN
与各自的残差支路，并在现有Spark Jupyter完整执行
`learning/day05_config_memory.ipynb`，6个代码单元格通过：

- 手工 `[6,4]` 权重将 `[1,2,3,4]` 投影为 `[3,5,7,-2,-2,10]`，
  再拆成2个3维头；直接把4个输入元素view为6个元素的负例按预期报错。
- 仓库 `GroupedQueryAttention` 固定E=8/Hq=2：
  默认D=4时Q权重 `[8,8]`，显式D=6时Q权重 `[12,8]`，
  各自输出投影输入宽度同步匹配，投影及拆头shape均通过断言。
- 配置算术和Day 5相关测试14项通过，Notebook schema、Ruff、
  修改文件本地链接及空白检查通过。

已执行Notebook：

```text
/workspace/logs/day05-projection-heads-executed-20261107T113217147380Z.ipynb
```

本次不修改模型实现、不下载checkpoint、不作质量/性能比较，也不重跑全仓库测试。
手工例与随机小模块用于验证“先投影、后拆头”，不代表任意改动维度后仍可加载旧权重。

## Day 5 Attention汇集与输出投影手算

2026-11-07 UTC，在现有Spark Jupyter中完整执行新增
`learning/day05_attention_handcalc.ipynb`，6个代码单元格通过：

- CPU FP64，两Query头共享一组K/V，概率分别为 `[0.75,0.25]`、`[0.25,0.75]`。
- 手算每头内容 `[7.5,5]`、`[2.5,15]` 与PyTorch SDPA forward对齐。
- 检查同位置合并为4维，给定 `[3,4]` 输出权重投影为 `[10,20,5]`，
  加示例残差X后为 `[11,22,8]`。
- 负例覆盖不匹配宽度的残差相加、多Query合并头时漏transpose造成的数据布局错误。
- 单变量实验覆盖两个头使用相同Q、所有V翻倍；断言更新与残差结果的区别。
- Notebook schema、Ruff、本地链接与空白检查通过。

已执行Notebook：

```text
/workspace/logs/day05-attention-handcalc-executed-20261107T120111709091Z.ipynb
```

Q/K/V直接给定，不由示例残差X生成；没有Norm/RoPE/FFN、真实Cache更新、
训练权重、GPU性能或完整Block验证。本次未重跑全仓库测试。

## Day 6启动与进度提示检查

2026-11-07（Asia/Singapore），用户明确确认从Day 5进入Day 6；
更新 [统一进度](LEARNING_PROGRESS.md) 与 [Day 6学习记录](DAY06_STUDY_LOG.md)。
各课程副本改为链接统一进度，保留历史报告当时的状态。

- 28项课程工具测试通过，新增断言保证生成Notebook不再写死Day 5。
- 34份学习Notebook的schema与代码语法检查通过。
- 对照Git基线检查25份分日课程：代码单元格内容、数量和cell ID不变，只修改进度提示。
- Ruff、本地文档链接与空白检查通过；同步到现有Jupyter前备份差异文件，不删除旧文件。

这是进度及文档元数据更新，不等于用户已完成Day 6；未重跑全部课程、
全仓库测试或GPU性能实验，个人结果保留待填写。

## Day 6收尾与Day 7调度补充验证

2026-11-07（America/Los_Angeles），用户要求完善教程并继续下一天。
已将当前学习位置更新为Day 7；Day 6收尾和Day 7个人自测仍不预填通过。
新增 [Day 7讲义](DAY07_SCHEDULER_MAP.md)、[学习记录](DAY07_STUDY_LOG.md)
与 [逐轮Notebook](../learning/day07_vllm_map.ipynb)。

验证结果：

- 在现有Spark Jupyter容器重新完整执行Day 6 Notebook的8个代码单元格，
  以及Day 7新Notebook的4个代码单元格，均通过。
- Day 7的A/B/C固定工作负载，sequential/static/continuous分别用6/5/3个模拟轮次。
  C在continuous的trace round=1首次执行，对应first_round=2；
  static中为trace round=3、first_round=4。两者编号不同，不是秒数。
- 单轮预算改为1后，三种模式均为6轮，未将连续批处理描述成必然减少轮数或固定加速。
- 映射示例检查逻辑位置6、block_size=4，对应逻辑块1、偏移2、
  表 `[3,7,1]` 中物理块7；仅做地址与容量算术，不分配真实KV。
- 新增13项Day 7专项测试，覆盖精确轨迹、总工作守恒、到达前不执行、
  单轮不重复、预算上限、计数约定与空闲等待轮。
  连同现有课程工具/课程测试共 **57 passed**。
- 全部35份学习Notebook的schema与代码语法检查通过；
  新文件Ruff、修改文档本地链接和空白检查通过。

执行产物保存在：

```text
/workspace/logs/day06-day07-transition-20261107T151339572090Z/summary.json
/workspace/logs/day06-day07-transition-20261107T151339572090Z/day06_online_softmax.ipynb
/workspace/logs/day06-day07-transition-20261107T151339572090Z/day07_vllm_map.ipynb
```

本次没有修改模型或调度器实现，没有部署vLLM、测GPU吞吐或重跑全仓库测试。
Notebook内核仍提示本机TCP传输未加密；执行记录不是个人能力验收记录。

## Day 7四角色、存储与内存生命周期验证

2026-11-07（America/Los_Angeles），用户要求形象解释Request、Scheduler、
KV Cache Manager、Attention Kernel及存储、调用、优化步骤。
新增 [四角色与抽屉图解](DAY07_KV_LIFECYCLE.md)、
[CPU页池/真实小模型Notebook](../learning/day07_kv_lifecycle.ipynb)
与 `learning/day07_memory.py`；复用现有PagedPool和Engine，未修改它们的实现。

- 实际CPU页池shape `[6,2,2,1,2,2]`、FP32总计384 bytes、每物理块64 bytes。
- 验证A/B预留容量与逐层有效长度不同；B回收后，C以目录 `[4,5,2]`
  复用非连续页，逻辑位置4正确映射物理块2/offset 0，读取数据与连续参考对齐。
- 验证旧B句柄失效、释放不自动清零，新C有效长度从0开始；
  OOM时free/请求表/storage均保持不变；全部释放后pool地址和大小不变。
- 真实CPU随机小模型A/B/C与独立greedy输出一致，首轮两条3-token prompt
  实际组成Batch=2；后续不同位置/长度分桶，最终全部页回收。
- 注入一次同步forward异常后finally回收全部请求页。
  不把这个结果称为生产取消、多stream或跨进程恢复验证。
- Day 7测试累计19项；连同课程工具和课程测试共 **63 passed**。
- 新Notebook在现有Spark Jupyter完整执行4个代码单元格，全部通过；
  全部36份学习Notebook的schema/语法检查、Ruff、本地链接及空白检查通过。

已执行Notebook：

```text
/workspace/logs/day07-kv-lifecycle-executed-20261107T154520011653Z.ipynb
```

额外全仓库普通回归：临时容器首轮300 passed、27 skipped、1 failed；
失败为既有 `test_math500_parser_self_answers` 下载数据时TLS EOF错误，
不是本课断言失败。随后在已同步的现有Spark工作区使用缓存的500条MATH-500数据，
同样启用 `SKIP_EXPENSIVE=1/RUN_REAL_DOWNLOAD_TESTS=0` 完整重跑，
结果 **301 passed / 27 skipped / 1 warning**，26.87秒。
未关闭TLS验证、修改该测试或跳过失败项；保留PyTorch的GB10 capability提示。

本课没有真实vLLM部署、GPU性能压测、KV量化或生产多租户安全验收。
数据大小是CPU数组计数；教学句柄防越界不等于安全擦除或原始storage访问隔离。
官方深入阅读链接按Cache、vLLM架构、分页、Triton kernel、调优排序收录在图解中。

## 原始记录

已执行 Notebook 和分日实验报告保存在现有 volume 中，不加入 Git：

```text
/workspace/logs/course-notebooks-20261101/summary.json
/workspace/logs/course-notebooks-20261101/day06_course.ipynb
...
/workspace/logs/course-notebooks-20261101/day30_course.ipynb
/workspace/logs/day06-course.json
...
/workspace/logs/day30-course.json
/workspace/logs/cuda-vector-reduce
```

这些 JSON 包含 `status`，不能仅因 nbclient 没抛异常就把 Day 22 写成通过。
同步前的旧文件保存在 `/workspace/logs/course-sync-backup/<UTC时间>/`，
不会自动覆盖浏览器中尚未保存的编辑。

复现：

```bash
# 宿主机：先编译，再复制；本机架构为 sm_121
nvcc -arch=sm_121 -O2 learning/kernels/vector_reduce.cu -o /tmp/reasoning-vector-reduce
docker compose -f docker-compose.spark.yml cp \
  /tmp/reasoning-vector-reduce reasoning:/workspace/logs/cuda-vector-reduce

# 每次使用新的报告目录，不覆盖上次验证证据
docker compose -f docker-compose.spark.yml exec -T \
  -e CUDA_LESSON_BINARY=/workspace/logs/cuda-vector-reduce reasoning \
  /opt/venv/bin/python -m learning.verify_course --output-dir logs/course-notebooks-rerun
```

全套普通测试在新课程代码可见的环境中运行：

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 python -m pytest -q tests
```

此套件仍有少量下载 fixture，网络受限时按 Spark 教程传入可用代理。
27 个 skipped 没有被改为 passed，也没有执行全部重训练。

## 不作出的结论

- 没有称用户已完成 Day 6–30 或已“精通”。
- 没有称部署了真实 vLLM 或验证其真实调度/分页 kernel。
- 没有称教学 Triton kernel 优于生产 FlashAttention/cuBLAS。
- 没有把模拟 round 写成 GPU tokens/s。
- 没有把小随机 Transformer 当成加载了 Qwen3 权重的语言模型。
- 没有执行本月全部长训练、量化、多卡、多节点或正式大规模服务压测。

材料已具备可复现的学习与面试训练入口；当前行动见
[学习进度](LEARNING_PROGRESS.md) 与 [Day 7学习记录](DAY07_STUDY_LOG.md)，
历史参考实验通过不自动替代个人验收。
