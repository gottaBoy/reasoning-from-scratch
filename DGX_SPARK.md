# DGX Spark 实战教程

与 RTX 3090 的训练/部署选择见
[DGX Spark vs RTX 3090 对比](DGX_SPARK_VS_RTX3090.md)。

本教程对应当前仓库的 Qwen3 教学实现，不等于已完成 vLLM、所有 notebook 或
所有训练任务。先验证运行链路，再按 30 天计划学习；短 smoke 的准确率不代表
模型的正式评测成绩。

## 1. 本机环境与启动方式

2026-10-30 检查到的本机环境：

| 项目 | 实际值 |
| --- | --- |
| CPU 架构 | aarch64 / ARM64 |
| GPU | NVIDIA GB10 |
| 驱动 | 580.173.02 |
| 驱动报告的 CUDA 上限 | 13.0 |
| 宿主机 CUDA Toolkit | 13.0.88 |
| 系统可见内存 | 约 119 GiB |
| Docker GPU 注入 | CDI，`nvidia.com/gpu=all` 已通过 `nvidia-smi` 验证 |

`nvidia-smi` 的 CUDA Version 不是容器里安装的 PyTorch CUDA 版本。即使容器能
执行 `nvidia-smi`，CPU PyTorch 仍然不会使用 GPU，必须继续执行第 3 节自检。

在仓库根目录执行：

```bash
cd /home/my/workspace/llm/reasoning-from-scratch
uname -m
nvidia-smi
docker info
docker compose -f docker-compose.spark.yml config --quiet
```

本机建议使用独立的 `docker-compose.spark.yml`：

- 镜像为 `reasoning-from-scratch:spark`，不覆盖 CPU 镜像 `:local`。
- ARM64 PyTorch 来自仓库锁定的 CUDA 13.0 源，不使用 CPU wheel。
- 通过 CDI 注入 GPU，不要求 Docker 配置名为 `nvidia` 的 runtime。
- Host 网络避免本机已有的 Docker bridge 地址池耗尽问题。
- 共享内存设为 8 GiB；Jupyter 只监听本机 `127.0.0.1`。
- 包含 Triton 首次 JIT 需要的 C/C++ 编译工具。
- 默认 volume 为 `reasoning_spark_workspace`，与此前 CPU volume 分开。

这是单机开发环境；Host 网络和固定 Jupyter token 不适合直接暴露到公网。

## 2. 构建 GPU 镜像

无需代理、能直接访问包源时：

```bash
docker compose -f docker-compose.spark.yml build
```

本机此前的下载超时涉及两条链路：pip 安装 uv，uv 从锁文件下载 PyTorch/CUDA
wheel。只更换 pip 镜像不能解决第二条链路。宿主机已设置可用的 `HTTP_PROXY`、
`HTTPS_PROXY` 时，向构建过程传递代理：

```bash
docker compose -f docker-compose.spark.yml build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg HTTP_PROXY \
  --build-arg HTTPS_PROXY \
  --build-arg NO_PROXY
```

这里从环境读取代理，不把密码写入 Dockerfile。构建使用 Host 网络，因此本机
监听在 `127.0.0.1` 的代理可达。没有代理就去掉后三个参数。

GPU 模式保留 `uv sync --frozen`：下载地址由 `uv.lock` 决定，
`UV_INDEX_URL` 不会把所有已锁定的 CUDA wheel 重定向到镜像。首次需要下载数 GiB
GPU 依赖，不只是 505 MiB 的 torch；不要仅通过“没有实时百分比”判断下载卡死。
BuildKit 缓存会保留 uv 已下载的包，失败后重试同一命令，不要加 `--no-cache`。

## 3. 验证真实 CUDA 运算

```bash
docker compose -f docker-compose.spark.yml run --rm reasoning \
  python docker/check_environment.py --device cuda \
  --output logs/spark-environment.json
```

自检不仅检查 CUDA 可见性，还执行矩阵乘法、BF16 causal attention 和
反向传播，并在结束前同步 GPU。应确认：

- `device` 是 `cuda`，`cuda_build` 不为 `null`。
- `gpu` 是 `NVIDIA GB10`。
- `matmul` 和 `attention_forward_backward` 都为 `passed`。

自检失败时停止后续训练，不要自动切换到 CPU 掩盖错误。

第三周使用 Triton 前，先运行包含非整齐长度边界的 Vector Add：

```bash
docker compose -f docker-compose.spark.yml run --rm reasoning \
  python docker/check_triton.py
```

预期为 `vector_add: passed`。这一步验证 JIT 编译和 Kernel Launch，不是
算子加速比测试。`Failed to find C compiler` 表示缺少编译工具，Spark 镜像
已启用 `INSTALL_BUILD_TOOLS=1`，应重建镜像而不是只安装 Python 包。

本机这版 wheel 的架构列表包含 `sm_120` / `compute_120`，GB10 报告的是
`sm_121`，因此 PyTorch 会显示 capability 警告。本次矩阵乘法和 Attention
前后向已通过，但不能由此推断所有第三方 CUDA 扩展都兼容。如果出现
`no kernel image is available`，应检查对应扩展是否包含适用的 cubin/PTX，
而不是关闭警告。

## 4. 模型缓存与第一次生成

首次运行会下载 base 模型和 tokenizer。已有 CPU 模型缓存时，可先复用它，
而不复制旧源码。下面的源 volume 为只读，不会改动旧的学习进度：

```bash
docker run --rm --network none \
  -v reasoning_workspace:/seed:ro \
  -v reasoning_spark_workspace:/workspace \
  --entrypoint sh reasoning-from-scratch:spark \
  -c 'if [ ! -e /workspace/qwen3 ]; then cp -a /seed/qwen3 /workspace/qwen3; fi'
```

没有旧缓存就跳过。此前 CPU Compose 实际 volume 也可能带项目名前缀，可用
`docker volume ls` 确认，不要凭目录名判断。这里使用 `docker run` 避免 Compose
把名为 `reasoning_workspace` 的逻辑引用解析成 Spark 的新 volume。

GPU 生成：

```bash
docker compose -f docker-compose.spark.yml run --rm \
  -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY reasoning \
  python ch02/05_use_model/generate_simple.py \
  --device cuda --max_new_tokens 32 \
  --prompt "The capital of France is"
```

模型下载和数据集下载也需要网络；构建代理不会自动成为容器运行代理，所以此处
单独使用 `-e` 显式传入。Spark Compose 也会读取宿主机的这三个代理变量，
用于 Jupyter 中的下载；若无需代理，请清空宿主机的对应变量。

base 模型不是聊天助手，输出可能是网页续写；应先检查 `device: cuda`、
`cache: True` 和是否生成 token，不用一次生成的内容质量判断部署是否成功。

## 5. 小规模端到端验证

一次运行覆盖生成、评测、推理扩展、蒸馏和普通测试：

```bash
docker compose -f docker-compose.spark.yml run --rm \
  -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
  -e MAX_NEW_TOKENS=16 -e DATASET_SIZE=1 \
  reasoning sh docker/smoke_examples.sh
```

进一步验证第 6、7 章各一次训练步骤：

```bash
docker compose -f docker-compose.spark.yml run --rm \
  -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
  -e MAX_NEW_TOKENS=16 -e DATASET_SIZE=1 \
  -e RUN_TRAINING=1 -e RUN_ADVANCED_TRAINING=1 \
  reasoning sh docker/smoke_examples.sh
```

第 7 章格式奖励脚本使用 reasoning checkpoint，会额外下载一份模型。
蒸馏用仓库自带的 5 条样例，1 条作为验证，其余用于训练。训练脚本自动选择
可用 GPU；不是 `--device` 参数控制。

每一步保存到 `logs/smoke-<UTC 时间>-<PID>/`：

| 文件 | 内容 |
| --- | --- |
| `environment.json` | 本次运行的设备和 PyTorch 信息 |
| `summary.tsv` | 已执行步骤、退出码和日志路径 |
| `ch02_generation.log` 等 | 每个步骤的标准输出和错误 |
| `tests.log` | pytest 的通过、跳过和失败数量 |

脚本遇错即停，失败阶段退出码不为 0；最后一行
`Smoke examples completed` 才表示本次选中的阶段全部成功。
`RUN_TESTS=0` 可在单独调试推理链路时跳过测试，但不能据此声称测试通过。

普通测试也可单独执行：

```bash
docker compose -f docker-compose.spark.yml run --rm \
  -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
  -e SKIP_EXPENSIVE=1 -e RUN_REAL_DOWNLOAD_TESTS=0 reasoning \
  python -m pytest -q tests
```

`RUN_REAL_DOWNLOAD_TESTS=0` 不是绝对离线模式，部分普通测试的 tokenizer
fixture 仍可能访问网络。GPU 可见也不意味着每个测试都在 GPU 上计算。

## 6. Jupyter 和源码更新

```bash
docker compose -f docker-compose.spark.yml up -d
docker compose -f docker-compose.spark.yml logs --tail 30
```

浏览器访问 `http://127.0.0.1:8888/lab?token=reasoning`。端口已占用时先检查
`ss -ltn`，不要停止其他项目的服务。

源码在第一次启动时复制进 named volume。重建镜像不会覆盖 volume 内的 notebook
或源码修改。需要验证新的仓库版本时，使用新 volume，保留旧进度：

```bash
SPARK_WORKSPACE_VOLUME=reasoning_spark_next \
  docker compose -f docker-compose.spark.yml run --rm reasoning \
  python docker/check_environment.py --device cuda
```

GPU 自检通过不代表新课程已进入镜像。发布课程改动后，先重建镜像，再使用独立
Compose 项目和全新 volume 验证课程文件及普通测试，不挂载宿主机源码：

```bash
SPARK_WORKSPACE_VOLUME="reasoning_spark_check_$(date -u +%Y%m%dT%H%M%SZ)" \
  docker compose -p reasoning-course-check -f docker-compose.spark.yml \
  run --rm --no-deps reasoning sh -lc \
  'test -f learning/curriculum.py &&
   test -f learning/day30_course.ipynb &&
   SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 python -m pytest -q tests'
```

普通测试仍可能下载 tokenizer 和数据，需配置本教程前面说明的代理。
只复用模型/数据缓存不影响源码独立性；不要把旧工作区的源码一起复制过来。
`learning.sync_workspace` 只同步现有学习卷中的课程，不能代替重建镜像。
实际新镜像、新卷的完整 smoke 与 Notebook 验收见
[课程镜像验收记录](docs/IMAGE_VALIDATION.md)。

不要为更新代码直接执行 `down -v`；它会删除容器中的模型、训练结果和笔记。
停止容器但保留数据只需要：

```bash
docker compose -f docker-compose.spark.yml down
```

导出日志到宿主机，可先创建不启动的容器，再复制文件：

```bash
docker compose -f docker-compose.spark.yml create reasoning
docker compose -f docker-compose.spark.yml cp reasoning:/workspace/logs ./logs
```

日志导出不包含模型 checkpoint。要迁移学习进度，还需单独备份对应的 checkpoint
和 notebook，不能只复制 `logs`。

## 7. 对应 30 天学习计划

| 阶段 | 在 Spark 上的实践 | 完成标准 |
| --- | --- | --- |
| 第 1 周 | CUDA 自检、Qwen3 推理、阅读 Q/K/V shape 和 Cache | 能解释运行设备与张量流 |
| 第 2 周 | 小 shape 的 FP32 对齐，再试 BF16 和 KV Cache | 固定权重与输入、测试误差和缓存一致性 |
| 第 3 周 | CUDA/Triton 算子、Profiler、SDPA 对照 | 正确性和同步后的耗时均有记录 |
| 第 4 周 | 调度/分页模拟器，再单独评估推理框架 | 不把模拟器吞吐当作真实 GPU 性能 |

Spark 的 CPU/GPU 共享系统内存，不能把约 128 GB 当成全部可分配的独立显存。
结合 `free -h` 和 `torch.cuda.max_memory_allocated()` 记录内存，给桌面和其他
服务留出空间；`nvidia-smi` 的 `Memory-Usage: Not Supported` 不代表没有 GPU。

当前教学镜像是 Python slim 加 CUDA PyTorch，不包含 nvcc 编译工具链。
宿主机虽然已安装 nvcc，容器不自动继承它；第 3 周编译 CUDA C++ 前需要另行准备
ARM64 CUDA devel 环境。不要把安装 GPU PyTorch 等同于完整 CUDA 开发环境。

本仓库没有生产级 vLLM 服务或手写 FlashAttention 实现。vLLM/FlashAttention 的
外部安装应使用独立容器并核对 ARM64、GB10 和 CUDA 的兼容性，不要直接覆盖此处
已经验证的 torch/triton 环境。

## 8. 已验证范围

实测结果与未覆盖项目见 [Spark 验证记录](docs/DGX_SPARK_VALIDATION.md)。
每次修改依赖、驱动、checkpoint 或输入长度后应重新验证，不能只复用旧报告。

## 参考

- NVIDIA DGX Spark Porting Guide：
  <https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/index.html>
- NVIDIA CUDA / ARM64 移植说明：
  <https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/porting/cuda.html>
- NVIDIA Container Toolkit CDI：
  <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html>
