# Day 3：调试模型、验证模块职责与 Q/K/V 形状

前置阅读：[Day 1–2 学习笔记](DAY01_DAY02_NOTES.md)。
对应 [30 天计划](../LLM_30_DAY_STUDY_PLAN.md) 的 Day 3；今天用仓库实现验证理解，
不开始第二周的独立重写，也不要求立刻理解全部 RoPE 公式。
符号与轴顺序见 [统一规范](NOTATION.md)；本页旧实验的 `H/KV/D` 分别对应
`Hq/Hkv/D`，其中Q/K/V每头维度相同。

对 RoPE、`gate_proj/up_proj/down_proj` 或 SiLU 还不直观时，先看
[形象解释](ROPE_SWIGLU_INTUITION.md) 和
[旋转与门控 Notebook](../learning/day03_rope_swiglu.ipynb)，再回到形状和断点实验。

## 1. 今天要拿到哪些证据

不要只看文字解释或打印一遍 shape。完成三个实验：

| 实验 | 验证什么 | 对应 Day 2 的疑问 |
| --- | --- | --- |
| 模块 hooks | 捕获 Embedding、Norm、Attention、FFN、Block 的实际输入输出 | 各模块具体改变了什么？ |
| 复算并对齐 | 同一组权重复算 Attention/FFN，与模型真实模块输出比较；验证两次残差加法 | 流程图是否真的符合代码？ |
| 修改未来 token | 固定前半段输入，只改变后半段，比较前半段 logits | 因果 Mask 是否真的阻止读取未来？ |

已经提供 [可运行实验](../learning/day03_shapes.py)。它调用仓库的 `Qwen3Model`，
不修改模型源码、不下载模型、不更新权重。脚本会断言形状和数值结果，不一致则
报错退出，而不是只打印“成功”。

先自己预测结果，再运行，对照错误修正理解。

## 2. 先用小模型走通

在宿主机仓库根目录执行：

```bash
docker compose -f docker-compose.spark.yml run --rm reasoning \
  python learning/day03_shapes.py --config tiny --device cpu \
  --output logs/day03-tiny-shapes.json
```

小模型使用 FP32、2 层、随机权重，不需要 tokenizer：

```text
B=1  T=8  E=32  H=4  KV=2  D=16
```

- `B`：batch size。
- `T`：本次输入长度。
- `E`：残差主路径的特征宽度 `emb_dim`。
- `H`：Query head 数。
- `KV`：Key/Value head 数。
- `D`：每个 head 的特征维度。

故意令 `H×D=64` 不等于 `E=32`，帮助你观察 Q 投影的扩展和输出投影的还原。
输入是固定的整数 ID；随机模型的 logits 不能用于判断文本质量。

本机使用 named volume 保存源码副本，重建镜像不会自动覆盖旧副本。
如果提示找不到脚本，先看第 8 节的同步方法，不要删除 volume。

## 3. 在 Spark 上看真实 Qwen3 配置

小模型所有检查通过后，再运行本地已有的 Qwen3 0.6B 权重：

```bash
docker compose -f docker-compose.spark.yml run --rm reasoning \
  python learning/day03_shapes.py --config qwen3 --device cuda \
  --weights qwen3/qwen3-0.6B-base.pth \
  --output logs/day03-qwen3-shapes.json
```

这次使用 28 层 Qwen3、BF16 和 CUDA。脚本只细看第一个 Block 内部的形状，
但最终 logits 来自完整模型；限制 `B<=2,T<=32`，避免调试时误用长上下文。
省略 `--weights` 则使用随机权重，不会自动下载 checkpoint。

默认输入还是 8 个固定 ID，不对应前一节的中文示意句。这是数值与形状实验，
不是一次文本生成或吞吐 Benchmark。

本次已实际验证：小模型 CPU/FP32 和已有权重的 Qwen3 CUDA/BF16 均通过 13 项检查，
本节断点命令也已在 pdb 中执行。加入本实验的 9 项测试后，普通回归测试结果为
`186 passed, 27 skipped`，保留 GB10 capability 警告。旧 Spark 报告中的
`177 passed` 是加入这 9 项测试之前的基线，不是相互矛盾的结果。

## 4. 对照这张形状表

使用 `B=1,T=8` 时：

| 阶段 | 小模型 | Qwen3 0.6B |
| --- | --- | --- |
| 输入 IDs | `[1,8]` | `[1,8]` |
| Embedding | `[1,8,32]` | `[1,8,1024]` |
| Q 投影 | `[1,8,64]` | `[1,8,2048]` |
| K/V 投影，各自 | `[1,8,32]` | `[1,8,1024]` |
| Q，拆 head 并交换轴 | `[1,4,8,16]` | `[1,16,8,128]` |
| K/V，扩展前 | `[1,2,8,16]` | `[1,8,8,128]` |
| K/V，扩展后 | `[1,4,8,16]` | `[1,16,8,128]` |
| QK 分数 | `[1,4,8,8]` | `[1,16,8,8]` |
| Attention weights | `[1,4,8,8]` | `[1,16,8,8]` |
| weights × V | `[1,4,8,16]` | `[1,16,8,128]` |
| 合并 head | `[1,8,64]` | `[1,8,2048]` |
| Attention 输出投影 | `[1,8,32]` | `[1,8,1024]` |
| FFN，两条中间分支各自 | `[1,8,96]` | `[1,8,3072]` |
| FFN 输出、Block 输出 | `[1,8,32]` | `[1,8,1024]` |
| 全部 Block 后的 logits | `[1,8,64]` | `[1,8,151936]` |

同样出现 `1024`，含义也可能不同：这里 K/V 投影的 `1024=8×128`，
Embedding 的 `1024=E`，两者只是碰巧数值相同。

### `view` 与 `transpose` 在做什么

以 Qwen3 的 Q 为例：

```text
[B,T,2048]
    ↓ view：将2048个特征拆为16组，每组128个
[B,T,16,128]
    ↓ transpose(1,2)：交换时间轴和head轴
[B,16,T,128]
```

这两个动作都没有学习新的权重，也没有增加元素数量。
`transpose` 改变轴顺序和访问步长；JSON 中的 `stride` 可用于观察，不要以为
换轴后内存一定重新排列成连续布局。

### GQA 如何共享 K/V

Qwen3 中 `H/KV=16/8=2`。仓库的教学实现通过 `repeat_interleave` 扩展 K/V：

```text
KV head 0 → Query head 0、1
KV head 1 → Query head 2、3
...
KV head 7 → Query head 14、15
```

是每两个 Query head 共用一组 K/V，不是每个 Query head 有独立的 K/V 投影。
扩展后参与矩阵乘法，返回的 K/V 仍是扩展前的 8 个 head。
今天观察的是 Attention 返回的紧凑 K/V；没有启用跨次 forward 的 Cache 追加，
那是 Day 4 的任务。生产 kernel 也不一定需要真正复制 K/V。

## 5. 把 Day 2 的模块职责变成断言

在 JSON 的 `checks` / `max_absolute_errors` 中查看：

| 检查项 | 在验证什么 |
| --- | --- |
| `norm1_feeds_attention` | Attention 接收第一次 Norm 的输出 |
| `attention_replay_matches_hook` | 用同一权重复算 Q/K/V、RoPE、Mask、加权求和，与真实 Attention 模块输出一致 |
| `attention_residual` | 第一次相加等于原输入加 Attention 分支输出 |
| `norm2_feeds_ffn` | FFN 接收第二次 Norm 的输出 |
| `ffn_replay_matches_hook` | 两条 FFN 投影、SiLU、门控、输出投影对应真实 FFN |
| `ffn_residual` | 第二次相加等于更新后的状态加 FFN 分支输出 |

hooks 捕获的是实际模块边界。Q/K/V 的 reshape、分数矩阵等是用捕获的输入和
同一权重复算的，因为普通模块 hooks 不能直接看到函数内部的每一个局部变量。
复算与真实输出对齐是重要证据；如需直接观察原始局部变量，使用下一节的断点。

数值容差：FP32 使用 `atol=rtol=1e-5`；BF16 使用 `atol=rtol=2e-2`，
并保留实际最大绝对误差。低精度不要求逐位一致；通过容差也不代表所有形状、
设备和 kernel 都已被验证。

## 6. 真正打断点：停在原始模型代码中

### 已经启动 Jupyter，可以直接调试

**不需要再创建一个容器，也不需要关闭 Jupyter。**
前面的 `docker compose run` 是宿主机终端的一条可选入口，不是调试的必要步骤。
当前 Spark Jupyter 的 `Python 3 (ipykernel)` 内核已声明 `debugger: true`。

可以按自己使用的界面选择：

| 入口 | 怎样启动 | 暂停时在哪里操作 |
| --- | --- | --- |
| Notebook 可视化调试 | 打开 `learning/day03_debug.ipynb`，启用调试开关并设断点 | Variables、Call Stack 和单步按钮 |
| Notebook 文本调试 | 在代码单元格使用 `%run -d ...` | 单元格下方的 `ipdb>` 输入框 |
| Jupyter Terminal | Launcher → Terminal，运行 `python -m pdb ...` | Terminal 的 `(Pdb)` 提示符 |

这些入口都可以使用现有容器。Notebook 调试器控制的是当前内核中的代码，
不会自动接管 Terminal 或另一个容器里单独运行的 Python 进程。
可直接打开 [Day 3 交互式 Notebook](../learning/day03_debug.ipynb)。

#### 推荐：先试 Notebook 的可视化断点

1. 在 Jupyter 左侧文件浏览器打开 `learning/day03_debug.ipynb`。
2. 选择 `Python 3 (ipykernel)`，先运行初始化单元格。
3. 启用 Notebook 右上角的调试开关（图标以当前界面为准）。
4. 在 `split = projected.view(...)` 所在行左侧的断点栏单击，确认出现红点。
5. 运行函数定义单元格，再运行调用 `inspect_query` 的单元格。
6. 暂停后查看 Variables 中的 `projected`，再点 Step Over 观察 `split`、
   `heads`，最后点 Continue 完成当前调用。

你应观察到：

```text
projected  [1,8,64]
    ↓ view
split      [1,8,4,16]
    ↓ transpose
heads      [1,4,8,16]
```

如 Variables 默认只显示张量内容，展开张量的 `shape` 属性看维度，不要把
张量数值列表误当作形状。

Notebook 将原代码同一行的 `view` 和 `transpose` 拆开，便于观察中间状态；
之后的验证单元格仍调用仓库真实模型做 13 项检查。
暂停时不要继续运行其他计算单元格，先 Continue、结束调试或重启内核。

#### 直接在 Notebook 调试原始脚本

要停在 `qwen3.py` 中，先结束可视化调试，再在一个代码单元格运行：

```python
%cd /workspace
%run -d -b reasoning_from_scratch/qwen3.py:192 learning/day03_shapes.py --config tiny --device cpu
```

出现 `ipdb>` 后，逐条输入以下命令。先用 `c` 进入已设置的断点：

```text
c
p tuple(queries.shape)
n
p tuple(queries.shape)
q
```

`%run` 是 IPython magic，不要加 shell 前缀 `!`。`%run -d` 才是在当前内核
使用文本调试器运行脚本，不要改成 `!python -m pdb ...`。
Notebook 可视化调试和文本 `ipdb` 调试选一种，不要同时让两个调试器暂停。

本次已执行 Notebook 的全部代码单元格，并通过内核调试协议验证断点暂停和
Step Over：`projected` 为 `(1,8,64)`，执行 `view` 后 `split` 为 `(1,8,4,16)`。
Notebook `%run -d` 路径也已验证；这不是浏览器界面截图测试。

#### 在 Jupyter Terminal 使用 pdb

Launcher（左上角 `+`）→ Terminal，输入：

```bash
cd /workspace
python -m pdb learning/day03_shapes.py --config tiny --device cpu
```

此处已经在容器内，不输入 `docker compose`。后面的断点命令与宿主机启动时一样。
关闭一次文本调试不会关闭 Jupyter 服务。

如果可视化调试开关不可用，检查是否选中支持调试的 Python 内核，而非仅打开了
Markdown 或文件编辑器。重启内核会清空已有变量；重新运行初始化单元格即可。
Terminal 的 pdb 路径不依赖 Notebook 可视化调试开关。

官方说明：

- [JupyterLab Debugger](https://jupyterlab.readthedocs.io/en/stable/user/debugger.html)
- [IPython %run 的 -d / -b 参数](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-run)

### pdb 的原理：暂停真实执行，检查当前状态

普通运行时，Python 连续执行程序；在 pdb 控制下运行时，解释器仍在真实执行
同一份程序，只是调试器可以在特定位置接管控制，等待你的下一条命令。
不是复制一份模型，也不是根据源码猜测变量值。

本机使用 Python 3.12，pdb 通过 `bdb` 调试框架和 `sys.settrace` 的执行追踪机制，
接收行执行、函数调用、返回和异常等事件。概念上：

```text
Python 即将执行一行代码
    |
    v
追踪机制通知调试器：当前文件、行号、函数执行现场
    |
    v
是否命中断点，或者满足单步停止条件？
    | 否                         | 是
    v                            v
继续执行                     出现 (Pdb)，等待命令
                                 |
                      查看变量 / 下一步 / 继续
```

断点保存的是“文件 + 行号”等条件，并不需要把模型源码里的这一行改掉。
函数执行现场叫栈帧（frame），包含当前局部变量和源码位置；
例如在 Attention 的 `forward` 栈帧里，pdb 可以找到 `x`、`queries`、`keys`。

在当前例子中：

```text
> .../qwen3.py(192)forward()
-> queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
(Pdb)
```

这表示已经进入 Attention 的 `forward`，但第 192 行还没执行。对小模型：

```text
(Pdb) p tuple(queries.shape)
(1, 8, 64)
(Pdb) n
> .../qwen3.py(193)forward()
-> keys_new = ...
(Pdb) p tuple(queries.shape)
(1, 4, 8, 16)
```

第一次 `p` 在当前栈帧中求值并打印形状；它没有推进模型的执行位置。
`n` 执行第 192 行并停到本函数的下一行，因此第二次 `p` 才看到重排后的形状。

`n` 与 `s` 的区别不是“是否执行函数”，而是“要不要停进被调用的 Python 函数”：

- `n`：调用仍然执行完，只是不逐行停进其内部；适合先观察模块边界。
- `s`：单步进入被调用的 Python 函数，可能先进入 PyTorch 的调用包装代码。
- `c`：恢复执行，直到下一次命中断点或其他停止条件。
- `q`：退出调试，停止本次被调试程序，不是让它继续在后台跑完。

调试时程序会多次停下，不能把这时的总耗时当作正常推理性能。
先查正确性，结束调试后再单独 Benchmark。

### 在哪里输入命令

注意区分三个提示符：

| 看到的位置 | 应输入什么 |
| --- | --- |
| 宿主机普通终端，如 `$` 或 `%` | `docker compose ... python -m pdb ...` |
| Jupyter 的 Terminal 中，已经位于 `/workspace` | `python -m pdb learning/day03_shapes.py ...` |
| 调试器的 `(Pdb)` | `tbreak`、`c`、`p`、`n` 等调试命令 |

不要把 `(Pdb)`、输出结果或 `->` 复制成要执行的命令，也不要在 `(Pdb)` 后再输入
`docker compose`。每条命令输入后按回车；空行可能重复上条调试命令，初学时不要
随意连续按回车。

### 第一次操作：从启动到查看残差

先使用 CPU 小模型，避免一开始就打印 GPU 大张量：

```bash
docker compose -f docker-compose.spark.yml run --rm reasoning \
  python -m pdb learning/day03_shapes.py --config tiny --device cpu
```

出现 `(Pdb)` 后，先观察 Day 2 的残差分支：

```text
tbreak reasoning_from_scratch/qwen3.py:125
c
p tuple(x.shape)
n
n
p tuple(shortcut.shape)
p tuple(x.shape)
```

此时 `shortcut` 是原输入，`x` 已经过 `norm1`。可以比较两者的数值，
形状相同不代表数值相同。随后：

```text
n
p tuple(x.shape)
n
p tuple(x.shape)
```

依次执行 Attention 分支和第一次残差相加。注意 pdb 的箭头指向
**尚未执行**的那一行，`n` 才执行当前行。

输入 `q` 退出，再执行同一条启动命令。回到初始 `(Pdb)` 后，
观察 Day 3 的 Attention：

```text
tbreak reasoning_from_scratch/qwen3.py:192
c
p tuple(x.shape)
p tuple(queries.shape), tuple(keys.shape), tuple(values.shape)
n
p tuple(queries.shape)
n
n
p tuple(keys_new.shape), tuple(values_new.shape)
until 220
p tuple(keys.shape), tuple(values.shape)
p tuple(next_cache[0].shape)
until 224
p tuple(attn_scores.shape), tuple(attn_weights.shape)
p attn_weights[0,0]
n
p tuple(context.shape)
```

默认小模型中，你应看见 Q 从 `[1,8,64]` 变成 `[1,4,8,16]`；
计算前 K/V 已扩展成 4 个 head，但 `next_cache[0]` 仍只有 2 个 head。

常用命令：`l` 看附近代码，`p` 打印，`n` 执行下一行但不进入函数，
`s` 进入函数，`c` 继续，`q` 退出。查看形状时用 `p tuple(x.shape)`，
不要直接打印整个模型或整张 logits。

这些行号对应当前仓库版本；若代码行号变化，先查找：

```bash
rg -n 'shortcut = x|queries = queries.view|attn_scores =|context =' \
  reasoning_from_scratch/qwen3.py
```

重新计算 Attention 的实验代码也会再次调用一些子模块；使用 `tbreak`
这种一次性断点，避免多次停在同一位置而误以为模型多了一层。

## 7. 验证 Mask：不能只看三角形

脚本会打印：

```text
0 1 1 1 1 1 1 1
0 0 1 1 1 1 1 1
0 0 0 1 1 1 1 1
...
0 0 0 0 0 0 0 0
```

在本仓库 `masked_fill` 的布尔 Mask 中，`1/True` 表示禁止访问。
这不是所有 Attention API 的通用语义，不能不加区分地直接复用 Mask。
对角线为 0，表示可以读取当前位置本身。

还会检查：

1. Mask 是严格上三角。
2. softmax 后被禁止的未来位置权重为 0，每行权重之和近似为 1。
3. 两次输入的前半段 ID 相同，后半段 ID 全部替换，前半段 logits 仍数值一致。

例如：

```text
原输入：0 1 2 3 | 4 5 6 7
新输入：0 1 2 3 | 5 6 7 8
               ↑
比较前4个位置的logits，而不是只比较argmax出的文字
```

`future_tokens_do_not_change_prefix_logits` 通过，才是今天的因果性验证证据。
不要要求后半段 logits 也相同，它们读取到的输入已经变了。

## 8. 不重建镜像，如何同步本次实验

此方法针对已经创建、启动过的 Spark Jupyter 服务。
先检查容器的 `/workspace/learning` 是否有自己的同名脚本；有的话先备份并比较，
不要覆盖。当前新增的实验脚本和文档可以同步：

```bash
docker compose -f docker-compose.spark.yml exec -T reasoning \
  mkdir -p /workspace/learning /workspace/docs

docker compose -f docker-compose.spark.yml cp \
  learning/day03_shapes.py reasoning:/workspace/learning/day03_shapes.py

docker compose -f docker-compose.spark.yml cp \
  learning/day03_debug.ipynb reasoning:/workspace/learning/day03_debug.ipynb

docker compose -f docker-compose.spark.yml cp \
  docs/DAY03_SHAPES_DEBUGGING.md reasoning:/workspace/docs/DAY03_SHAPES_DEBUGGING.md
```

用新 volume 从头开始的学习者，先按 [Spark 教程](../DGX_SPARK.md) 重建包含
最新文件的镜像即可。只同步教学脚本，不删除旧模型、笔记和 checkpoint。

也可以在 Jupyter 的 Terminal 中执行：

```bash
python learning/day03_shapes.py --config tiny --device cpu \
  --output logs/day03-tiny-shapes.json
```

终端不要输入宿主机的 `docker compose` 命令；已经在容器内时直接用 `python`。

## 9. 今日验收与 Day 4 入口

- [ ] 能用断点找到第一次和第二次残差相加。
- [ ] 在运行前写对 Q/K/V 投影、拆 head、扩展和合并后的形状。
- [ ] 能解释 Qwen3 的 Q 为何是 `1024 → 2048`，而非始终 1024。
- [ ] 能解释 `view`、`transpose` 和线性投影的区别。
- [ ] 能解释 GQA 的 head 对应关系及紧凑 K/V 的返回位置。
- [ ] 能说明因果 Mask 的方向，并展示修改未来 token 不影响前缀的检查。
- [ ] 保存 JSON 报告和自己的预测/错误记录，而不是只记录“所有检查通过”。

完成后进入 Day 4：固定相同权重与 8 token 输入，对比完整前向和
`7 token prefill + 1 token decode`。届时再学习 Cache 追加、RoPE 位置偏移和重置；
今天不必同时修改这些机制。
