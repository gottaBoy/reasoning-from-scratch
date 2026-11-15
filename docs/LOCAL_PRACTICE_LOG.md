# 本地实战日志

开始日期：2026-11-08（America/New_York）。
用户确认：概念课程全部阅读，转入本地实战。
当前关卡：**P1-02，单头因果Attention计算核心**。
教程：[本地反复练习路线](LOCAL_PRACTICE.md)。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| 概念阅读 | 用户自述已全部看完，不等于逐模块独立实现 |
| 已有环境 | 复用Spark Docker/Jupyter，不重建、不升级依赖 |
| 验收入口 | `python -m learning.local_practice --task attention`；默认任务仍为norm-rope，可传自己的candidate文件 |
| 参考CPU/CUDA基线 | 本次均14/14通过，implementation_role为reference_baseline |
| P1-01用户candidate | `/workspace/learning/my_ops.py`，支持范围内CPU/CUDA均14/14通过 |
| P1-02用户candidate | 已约定使用 `/workspace/learning/my_ops.py`；causal_attention仍待实现/验收 |
| 后续模型/算子/引擎/服务 | 按关卡推进，未自动通过；真实vLLM仍未配置 |

## 已执行的参考证据

2026-11-08在现有DGX Spark容器执行，未升级或重建环境：

- NVIDIA GB10、PyTorch 2.10.0+cu130；实际矩阵乘法和Attention前后向自检通过。
- P1-01参考实现CPU 14/14、CUDA 14/14通过；未传用户candidate文件。
- 新Notebook三个代码单元格完整执行；原始证据保存在以下容器路径。
- 全仓库普通回归 `355 passed / 27 skipped / 1 warning`；
  保留GB10 capability提示，不把它隐藏或当作所有扩展兼容证明。

```text
/workspace/logs/practice/environment-20261108T075408094557Z.json
/workspace/logs/practice/p1-01-cpu-20261108T075728451905Z.json
/workspace/logs/practice/p1-01-cuda-20261108T075728827959Z.json
/workspace/logs/practice/practice01-executed-20261108T075729344831Z.ipynb
```

以上是助手运行的环境与参考代码证据。你的独立实现、调错记录和性能实验仍待填写。

## P1-01用户文件验收：2026-11-08（America/Los_Angeles）

继续练习时读取到用户已经填写RMSNorm与RoPE。首先尝试容器内验收，
发现candidate文件不存在；这是宿主机与named volume源码未同步，不是数值计算失败。
确认容器目标不存在后，仅复制my_ops.py，未改算法或占位注释。

```text
implementation_role = candidate_file
source = /workspace/learning/my_ops.py
source_sha256 = 70b81c835cd769c7799595b9db44e051bae5b9df741fe4ea7174c2ec2af6bff9
CPU：14 passed / 0 failed
CUDA：14 passed / 0 failed
```

实际报告：

```text
/workspace/logs/practice/p1-01-20261108T161905605802Z.json
/workspace/logs/practice/p1-01-20261108T161922365171Z.json
```

两份报告源码hash一致。代表该文件通过已定义FP32数值、边界、位置及有限梯度检查，
不代表所有dtype/layout、训练质量、性能或独立创作已验证。
源码中的TODO/return后不可达占位仍可由学习者清理，本次不替换或自动提交该文件。

下一关：[P1-02步骤与模板](PRACTICE_P1_02_ATTENTION.md)。
当前任务是自己的 `causal_attention(q,k,v,start=0)`；不把参考baseline记成用户已完成。

## P1-02参考验收：2026-11-08（America/Los_Angeles）

新增A1–A5待填模板、逐步Notebook与 `--task attention` 验收入口。
参考CPU/CUDA各14/14通过，implementation_role为reference_baseline；
用户尚未指定my_attention.py，不登记为个人实现已通过。
本次只验证单头FP32核心，Q/K/V已给定，不含投影、RoPE或Cache管理。

```text
/workspace/logs/practice/p1-02-cpu-20261108T162801132622Z.json
/workspace/logs/practice/p1-02-cuda-20261108T162801698385Z.json
/workspace/logs/practice/practice02-attention-executed-20261108T162802215225Z.ipynb
```

全仓库普通回归364 passed / 27 skipped / 1 warning；保留GB10提示。
源码、文件路径和报告的变化要按各自的hash与时间记录，不覆盖P1-01证据。

## 此前文档归档决定（已由下节更新）

本轮确认：P1-02不另建`my_attention.py`，直接在
`learning/my_attention_template.py`完成A1–A5。模板集中放置变量说明、shape推导、
17.5手算例、Mask规则、验收命令和待填写代码；解释、文件职责、失败模式和验收边界
统一记录在 [P1-02教程](PRACTICE_P1_02_ATTENTION.md)。

`practice02_attention.ipynb`只作为逐格观察工具，`practice_attention.py`保持独立oracle
与负例验收，不能为让测试通过而修改。`learning/my_ops.py`属于已完成的P1-01个人文件，
本轮不覆盖、不重写。后续每个实战关卡沿用同一归档方式：

```text
一个可填写的实现模板
→ 一份解释与边界文档
→ 一个独立验收器
→ 一份个人运行日志
```

## 2026-11-14：统一模板与个人实现

用户决定沿用P1-01方式，不再直接填写独立Attention模板。
当前约定取代上一节的文件组织方式：

```text
my_ops_template.py：集中RMSNorm、RoPE、Attention的接口、TODO、shape与手算提示
my_ops.py：用户唯一的个人实现文件，保留已完成部分，逐关新增函数
local_practice.py：按--task选择接口，加载同一个candidate文件，记录独立报告
```

已将Attention的A1–A5合并到统一模板并移除独立模板；
同步更新教程、Notebook和命令。`--task norm-rope`只检查RMSNorm/rope，
`--task attention`只检查causal_attention，后续未完成的函数不阻塞已完成关卡。
模板保持未完成；本轮不修改用户 `my_ops.py`，也不登记Attention个人验收通过。
后续新增关卡继续集中维护模板与个人实现，独立oracle/验收器仍与二者分离。

本轮容器内两份实战验收器测试19项通过，Ruff与Notebook结构/语法检查通过；
这不是用户Attention实现通过。已选择性同步教学文件至现有Jupyter卷，
旧文件（含移出的独立Attention模板）保存在
`/workspace/logs/course-sync-backup/unified-ops-20261114T112334019835Z/`。
用户文件hash保持
`70b81c835cd769c7799595b9db44e051bae5b9df741fe4ea7174c2ec2af6bff9`。

## 2026-11-14：增加中文图解模板

保留统一英文模板，增加 [my_ops_template_zh.py](../learning/my_ops_template_zh.py)，
同一文件涵盖RMSNorm、RoPE、Attention，不重新按算子拆文件。
两种语言版本使用相同接口、参数校验、TODO编号和未完成占位，
任选一版阅读，实现仍只写在个人 `my_ops.py` 中。

中文说明采用“直觉比喻 -> 小数字手算 -> shape变化 -> TODO”的顺序：
RMSNorm是总强度缩放与通道权重，RoPE是前后半部成对指针的不同频率旋转，
Attention是问题/资料索引匹配后按权重混合内容。
额外解释eps、dtype恢复、theta基数/弧度、绝对位置、mask方向及两个矩阵乘法消去的轴，
并标明比喻不是真实通道语义，模板也不是完成的实现或性能证明。

本轮不修改或同步个人实现文件；宿主机 `my_ops.py` 的起始hash为
`e26236d5fd6cc2541c3a155ed88fb562159ef1e82d32f32b20bd97dcd59ecda0`，
它是用户继续编辑后的状态，与此前验收hash分开记录，不沿用旧验收结论。

中文模板与两份已有实战验收器测试合计 **24 passed**，Ruff通过。
新增测试去除文档字符串后比较两版AST，确认接口与可执行代码一致；
同时确认中文模板仍抛未完成异常，并保留参数拒绝校验。
本轮只验收教学模板/验收链，不重新验收个人文件，不产生GPU性能结论。
教学文件已选择性同步至Jupyter，旧文件备份于
`/workspace/logs/course-sync-backup/ops-template-zh-20261114T124418464084Z/`。

### 补充A1：Key转置与逐列打分

用户追问 `k.transpose(-2,-1)`，已补入中文模板的Attention讲解及A1注释，
并在 [P1-02教程第2.1节](PRACTICE_P1_02_ATTENTION.md#attention-key-transpose)
集中说明：负轴编号、`[B,1,T_kv,D_k] -> [B,1,D_k,T_kv]`、
3份资料/2个特征的行列图，以及`[1,2] @ K转置 = [5,11,17]`手算与Jupyter示例。
区分未缩放点积分数、后续Mask/softmax和最终输出，不把转置误解为元素交换、
reshape或原地修改；个人实现文件不改动。

### 补充A3：多维分数与Mask广播

用户继续追问多维张量的理解，已补入中文模板和
[P1-02教程第2.2节](PRACTICE_P1_02_ATTENTION.md#attention-mask-broadcasting)：
`scores[b,h,query_i,key_j]` 的四个轴、当前单头的
`[B,1,T_q,T_kv]`、Mask的`[T_q,T_kv]`，以及从右向左对齐的广播规则。
补充`B=2、H=2`时四张“问题×资料”表共用一张因果地图，
并区分与batch/padding相关的`[B,1,T_q,T_kv]` mask及每个head不同的完整mask。
本轮只更新教学说明，不修改个人 `my_ops.py`。

### 补充A4：四种模态的直觉映射

用户希望用文本、图片、音频、视频分别理解多维 Attention，已补入中文模板和
[P1-02教程第2.3节](PRACTICE_P1_02_ATTENTION.md#attention-modalities)：
文本用token、图片用patch、音频用时间帧、视频用时空patch/tubelet。
文档同时说明`T`/`N`的含义、`[B,H,T_q,T_kv]`的统一结构，
以及视觉双向Mask、流式音频因果Mask、视频时空Mask不能互相默认替换。
这些是理解特征向量的类比，不把Value误说成原始文字/像素/音频/视频文件。

## 每轮填写

```text
关卡：
源码路径 / source_sha256：
实现来源（自己编写 / 参考改写 / 直接调用参考）：
预测：
CPU报告：
CUDA报告：
第一个失败case：
错误原因：
单变量修改：
新输入验证：
性能是否实测、口径：
当前不支持：
下一轮：
```

每轮使用新报告文件；未通过项保留失败记录，不以删除case或放宽阈值制造通过。
