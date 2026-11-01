# CUDA / Triton 学习算子

`triton_ops.py` 包含 Add、RMSNorm、Softmax、非 affine LayerNorm、
SiLU×up、tiled MatMul、online causal Attention。全部为教学 forward-only；
wrapper 显式拒绝 CPU、非连续 tensor、不匹配 dtype/device 和 requires_grad 输入。
要看真实编译产物，运行 `python -m learning.triton_ir --output-dir logs/ir-new-run`，
或在Jupyter打开 `learning/day17_triton_ir.ipynb`。
概念解释见 [Triton IR](../../docs/TRITON_IR.md)；不能用IR检查替代Benchmark。

Attention 限制：相同 shape `[B,H,T,D]`，D=16/32/64，causal prefill，
无 padding/GQA/Cache/backward。Row kernel 限制宽度 1..8192。

在 Spark Jupyter Terminal：

```bash
python -m learning.curriculum --day 21 --device cuda --output logs/day21.json
python -m pytest -q tests/test_learning_kernels.py
```

`vector_reduce.cu` 独立于 torch C++ ABI，展示 Vector Add、shared memory reduction。
6 组大小含 255/256/257、1003/65537；最后一步在 CPU 合并 block sums。

当前 Spark slim 镜像有 C++ 编译器用于 Triton JIT，但没有 nvcc。
在宿主机仓库根目录，使用已安装的 CUDA Toolkit：

```bash
nvcc -arch=sm_121 -O2 learning/kernels/vector_reduce.cu -o /tmp/reasoning-vector-reduce
/tmp/reasoning-vector-reduce
```

`sm_121` 对应本机 GB10，换硬件必须核对目标架构。不要把宿主机路径假定为容器路径。
如需在现有容器课程里复用这个独立可执行文件：

```bash
docker compose -f docker-compose.spark.yml cp \
  /tmp/reasoning-vector-reduce reasoning:/workspace/logs/cuda-vector-reduce
docker compose -f docker-compose.spark.yml exec -T reasoning \
  env CUDA_LESSON_BINARY=/workspace/logs/cuda-vector-reduce \
  /opt/venv/bin/python -m learning.curriculum --day 16 --device cuda
```

实际文件是否能在另一镜像运行还取决于动态库/ABI；本机这条路径会实际验证。
没有提供编译结果时，Day 15/16 明确返回 `requires_cuda_compiler` 而非通过。

性能数据：`learning.bench.measure` 排除 warm-up，用同步 wall-clock 多轮采样，
不是 kernel-only profiler。小 shape 教学 kernel 可以慢于 PyTorch；没有通用加速承诺。
