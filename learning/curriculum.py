"""Offline-first Day 6-30 labs. Reports distinguish GPU, simulation and external work."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from learning.bench import measure
from learning.mini_transformer import Cache, Config, Model, RMSNorm, rope
from learning.mini_vllm.engine import Engine, PrefixStore, Request
from learning.mini_vllm.paged_kv_cache import PagedPool
from learning.mini_vllm.scheduler import Job, simulate
from learning.online_attention import blocked_attention, online_softmax_chunks


# Each notebook embeds these explanations, a concrete experiment and self-check questions.
LESSONS = {
    6: ("Online Softmax 与显存",
        "完整 score 为 B×H×T×T。分块时维护每行最大值 m、指数和 l、加权分子 o；"
        "新块最大值变成 m' 后，旧 l/o 要乘 exp(m-m')，最后输出 o/l。",
        "online_attention.py", "为何不能直接相加不同最大值下的 softmax？把块宽改为 1/3/8，输出应变化吗？"),
    7: ("请求生命周期与调度地图",
        "请求先等待，再 prefill，再多轮 decode，完成后释放 Cache。"
        "连续调度在每轮边界补入新请求；Static Batch 等整批结束才补入。轮数不是秒。",
        "mini_vllm/scheduler.py", "短请求完成后，静态/连续策略分别何时接纳下一条？这证明 GPU 加速了吗？"),
    8: ("RMSNorm 与 RoPE 独立实现",
        "RMSNorm 沿特征维度归一化。RoPE 用位置旋转成对特征，保持 shape。"
        "先固定 FP32、offset、theta，与仓库参考对齐，不用语言输出判断算子正确性。",
        "mini_transformer/model.py", "改 offset 为 5 后数值为何变而形状不变？RMSNorm 是否减均值？"),
    9: ("单头因果 Attention",
        "QK^T/sqrt(D) 给出位置间分数，Mask 禁止未来，softmax 后权重沿 key 维求和为 1。"
        "再与 V 相乘。教学 Mask 的 True=禁止，SDPA bool Mask 的 True=允许，要显式转换。",
        "mini_transformer/model.py", "去掉 Mask，改变未来输入会影响前缀吗？最后一维 softmax 为什么是 key 维？"),
    10: ("MHA：拆头与合头",
         "线性投影改变内容；view 拆分 H×D；transpose 交换 head/time 轴。"
         "各 head 输出合并后经过 output projection 回到残差宽度。E 不必等于 H×D。",
         "mini_transformer/model.py", "四个 head 的 score 是四张 T×T 表，还是一张 4T×4T？"),
    11: ("GQA、MQA、MHA",
         "Query heads 数不变，KV heads 分别为 1、2、H。缓存紧凑 KV，不保存扩展副本。"
         "改变 KV heads 会改变参数形状，不能要求不同随机模型输出相同。",
         "mini_transformer/model.py", "KV 字节数为何与 KV heads 成正比？共享的是权重/表示还是 Q？"),
    12: ("自己的 Cache 与位置状态",
         "Cache 属于请求，不属于模型全局。prefill 后读出 position，decode 只投影最新 token，"
         "查询历史 K/V。拒绝超长输入必须发生在修改 Cache 之前。",
         "mini_transformer/model.py", "将缓存清空却不重置位置会怎样？两条并发请求能共享同一个可变 Cache 吗？"),
    13: ("Transformer Block 与梯度",
         "x1=x+Attention(norm1(x))，x2=x1+FFN(norm2(x1))。"
         "残差保留未归一化的主路径，FFN 内两条投影并行，SiLU(gate)逐通道乘up。",
         "mini_transformer/model.py", "第二次 residual 加的是 x 还是 x1？所有参数是否获得有限梯度？"),
    14: ("Mini Transformer 验收与计时",
         "先验证同权重的 Cache/no-cache 逐 token logits，再计时。warm-up 在计时外，"
         "GPU 前后同步；报告 shape、dtype、重复次数、峰值 allocated memory。",
         "mini_transformer/model.py", "若 JIT 和首次分配包含在计时中会怎样？一次短 prompt 能证明长期吞吐吗？"),
    15: ("CUDA Thread、Block、Grid",
         "i=blockIdx.x*blockDim.x+threadIdx.x；尾部 i>=N 的线程不读写。"
         "C++ 示例独立于 PyTorch ABI，需 nvcc 与适用 GPU 架构。",
         "kernels/vector_reduce.cu", "N=257、block=256 时需多少 blocks？第二个 block 有几个有效线程？"),
    16: ("Shared Memory Reduction",
         "每个 block 将局部元素放 shared memory，树形归约每轮同步；越界线程贡献零。"
         "示例最后在 CPU 合并 block sums，不伪装成全 GPU 最优实现。",
         "kernels/vector_reduce.cu", "为何 __syncthreads 不能只写在部分线程执行的条件里？"),
    17: ("Triton Vector Add 与 RMSNorm",
         "program_id 指向一块工作，arange 构建块内 offset，mask 处理尾部。"
         "RMSNorm 每个 program 处理一行，FP32 累积后恢复输出 dtype。",
         "kernels/triton_ops.py", "BLOCK 大于行宽时 reduction 的补充值应该是什么？"),
    18: ("Softmax、LayerNorm 与数值稳定",
         "softmax 先减行最大值，padding 位置设 -inf。LayerNorm 先求均值，再求居中方差；"
         "RMSNorm 则不减均值。测试非常数长度与高幅值输入。",
         "kernels/triton_ops.py", "softmax padding 用 0 会怎样？常数行 LayerNorm 应输出什么？"),
    19: ("MatMul Tiling",
         "一个 program 计算 BM×BN 输出块，沿 K 分 BK 累积。边缘 tile 读补零、写加 mask。"
         "变一个 tile 参数比较，不把更大 tile 必定更快当成结论。",
         "kernels/triton_ops.py", "M/N/K 不是 tile 的整数倍时哪三处要 mask？"),
    20: ("朴素 Attention 与 SDPA 对照",
         "首先用 PyTorch 分块参考验证因果性，再对照 SDPA。"
         "只调用 SDPA 不等于写出了融合 kernel；本日 GPU 实验另调用教学 Triton kernel。",
         "online_attention.py", "Prefill 的 T×T 与单 token decode 的 1×T 有什么区别？"),
    21: ("GPU Online Attention 与 Fused SwiGLU",
         "Triton kernel 对 Q/K/V 分块，在线维护 m/l/o，不写出完整 score。"
         "它仅支持等长 causal prefill、D=16/32/64、forward，无 backward/GQA/padding。",
         "kernels/triton_ops.py", "为何能称教学融合 Attention，不能称完整生产 FlashAttention？"),
    22: ("外部 vLLM 隔离集成",
         "在独立环境部署兼容 ARM64/GB10 的真实 vLLM，不能覆盖已验证的 torch 环境。"
         "本课程提供 /v1/models 与 /v1/completions 探针；没有服务就记录未执行。",
         "vllm_probe.py", "API 成功能证明 PagedAttention 内核正确性或性能吗？"),
    23: ("分页 Cache 的逻辑与物理映射",
         "logical_position // block_size 找 block table，余数找 offset。"
         "物理 block 可不连续，单个请求的逻辑顺序不变。",
         "mini_vllm/paged_kv_cache.py", "逻辑位置 6、block size 4 应映射到 table 的哪项/哪个 offset？"),
    24: ("Continuous Batching 状态转移",
         "用 arrival、token budget、running slots 约束调度。轮内不能重复执行同一请求；"
         "完成请求释放资源，下一轮可补入等待请求。先比较模拟轨迹，不换算 GPU tokens/s。",
         "mini_vllm/scheduler.py", "等待、公平轮转、完成三个事件如何记录？"),
    25: ("实现分页分配、回收与 OOM",
         "预先为请求预留容量，分配不足要原子失败。释放后的句柄失效，不能读到复用内存。"
         "教学实现显式 gather 页面，再做普通 Attention，不是 paged GPU kernel。",
         "mini_vllm/paged_kv_cache.py", "释放后旧引用还能读取吗？OOM 前后 free list 是否一致？"),
    26: ("按长度分桶的真实执行引擎",
         "Cache position 和本次输入长度相同的请求可以 stack 后调用一次模型。"
         "BatchCache 拆分每层更新回不同请求；prompt 不同但长度相同可以合批。",
         "mini_vllm/engine.py", "为什么不同位置的请求不能不带 mask/位置处理就直接 stack？"),
    27: ("调度模拟与真实 GPU 对照",
         "同权重同输入比较独立 greedy 与真实 engine 输出。"
         "实际 engine 有 page gather、Python 调度开销；结果快慢都如实记录，不保证加速。",
         "mini_vllm/engine.py", "哪些数字是轮数，哪些是同步 wall-clock？是否使用相同工作负载？"),
    28: ("Prefix Cache：先正确再优化",
         "选择精确完整前缀缓存；命中时复用 K/V 与最后 logits，克隆以隔离后续写入。"
         "不做 copy-on-write，不承诺节省内存；权重变动后必须清空 store。",
         "mini_vllm/engine.py", "只缓存前缀最后 logits、不缓存 K/V，能继续 decode 吗？"),
    29: ("报告与可复现性",
         "记录环境、seed、输入 token、dtype、shape、测试结果、源代码基线和已知限制。"
         "生成示例 token 是随机小模型，不是模型质量评测。保留失败和 skipped 项。",
         "curriculum.py", "别人仅凭报告能复现实验吗？哪些文件是训练产物而不应入 Git？"),
    30: ("综合验收",
         "组合模型、紧凑 KV、分页分配、分桶合批、Prefix Cache、正确性与计时。"
         "口述一次请求完整生命周期，并说明与真实 vLLM 的差距。",
         "mini_vllm/README.md", "能区分模型数学、kernel 优化、内存分配、调度和服务协议吗？"),
}


def make_model(device, cfg=Config()):
    torch.manual_seed(123)
    return Model(cfg).to(device).eval()


def _attention_lab(device, gpu=False):
    torch.manual_seed(1)
    q, k, v = [torch.randn(1, 2, 17, 32, device=device) for _ in range(3)]
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    actual = blocked_attention(q, k, v)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    result = {"blocked_pytorch_vs_sdpa": "passed", "shape": list(q.shape),
              "scope": "online-softmax mathematical reference, not a GPU kernel"}
    if gpu:
        from learning.kernels.triton_ops import attention
        actual = attention(q, k, v)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        result.update(triton_forward="passed", triton_latency=measure(lambda: attention(q, k, v), device),
                      sdpa_latency=measure(lambda: F.scaled_dot_product_attention(
                          q, k, v, is_causal=True), device))
    return result


def _cache_lab(device, kv=2):
    model = make_model(device, replace(Config(), kv_heads=kv))
    ids = torch.arange(8, device=device).unsqueeze(0)
    full = model(ids)
    state = Cache(model.cfg.layers)
    outputs = [model(ids[:, :5], state)]
    for i in range(5, 8):
        outputs.append(model(ids[:, i:i + 1], state))
    error = (full - torch.cat(outputs, 1)).abs().max().item()
    torch.testing.assert_close(full, torch.cat(outputs, 1), atol=1e-5, rtol=1e-5)
    return {"same_weight_cache_alignment": "passed", "max_absolute_error": error,
            "compact_cache_shape": list(state.get(0)[0].shape),
            "kv_bytes_per_layer": sum(t.numel() * t.element_size() for t in state.get(0))}


def _requests():
    return [Request("A", [1, 2, 3], 4), Request("B", [4, 5, 6], 2),
            Request("C", [7, 8], 3, arrival=1)]


def _engine_lab(device, timed=False):
    model = make_model(device)
    requests = _requests()
    reference = {r.name: model.generate(torch.tensor([r.prompt], device=device), r.max_tokens)[0].tolist()
                 for r in requests}
    engine = Engine(model, slots=2, blocks=16, block_size=2)
    result = engine.run(requests)
    assert result["outputs"] == reference
    assert len(engine.pool.free) == 16
    result.update(greedy_reference="passed", all_blocks_reclaimed=True)
    if timed:
        def sequential():
            for r in _requests():
                model.generate(torch.tensor([r.prompt], device=device), r.max_tokens)
        result["engine_latency"] = measure(lambda: engine.run(_requests()), device)
        result["sequential_latency"] = measure(sequential, device)
        result["timing_scope"] = "same three prompts/output budgets; no HTTP, no EOS, tiny random model"
    return result


@torch.inference_mode()
def run_day(day, device="cpu"):
    if day not in LESSONS:
        raise ValueError("Day must be between 6 and 30.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA requested but unavailable.")
    result = {"day": day, "title": LESSONS[day][0], "device": str(device), "status": "passed"}
    if day == 6:
        x = torch.tensor([1000., 1001., 998., -999., 1004.], device=device)
        y, history = online_softmax_chunks(x)
        torch.testing.assert_close(y, x.softmax(0))
        result.update(history=history, attention=_attention_lab(device),
                      score_example_bytes=1 * 16 * 4096**2 * 2)
    elif day in (7, 24):
        jobs = [Job("A", 1), Job("B", 5), Job("C", 1, 1), Job("D", 1, 2)]
        result["simulation"] = [simulate(jobs, mode) for mode in ("sequential", "static", "continuous")]
    elif day == 8:
        from reasoning_from_scratch.qwen3 import RMSNorm as RefNorm
        from reasoning_from_scratch.qwen3 import apply_rope, compute_rope_params
        x = torch.randn(1, 2, 7, 8, device=device)
        torch.testing.assert_close(RMSNorm(8).to(device)(x), RefNorm(8).to(device)(x))
        cos, sin = compute_rope_params(8, context_length=32)
        torch.testing.assert_close(rope(x, 3), apply_rope(x, cos.to(device), sin.to(device), 3))
        result.update(norm_reference="passed", rope_reference="passed", offset=3)
    elif day in (9, 10):
        cfg = replace(Config(), heads=1 if day == 9 else 4, kv_heads=1 if day == 9 else 4)
        model = make_model(device, cfg)
        ids = torch.arange(8, device=device).unsqueeze(0)
        changed = ids.clone()
        changed[:, 4:] += 1
        torch.testing.assert_close(model(ids)[:, :4], model(changed)[:, :4])
        result.update(logits_shape=list(model(ids).shape), causal_prefix="passed",
                      attention=_attention_lab(device))
    elif day == 11:
        result["variants"] = {str(kv): _cache_lab(device, kv) for kv in (1, 2, 4)}
    elif day in (12, 13, 14):
        result.update(_cache_lab(device))
        if day == 13:
            with torch.inference_mode(False), torch.enable_grad():
                model = make_model(device)
                scores = model(torch.tensor([[1, 2, 3]], device=device))
                F.cross_entropy(scores.flatten(0, 1), torch.tensor([2, 3, 4], device=device)).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters())
            result["backward"] = "passed"
        if day == 14:
            model = make_model(device)
            ids = torch.arange(8, device=device).unsqueeze(0)
            result["prefill_latency"] = measure(lambda: model(ids), device)
            result["cached_generation_latency"] = measure(lambda: model.generate(ids, 4), device)
    elif day in (15, 16):
        binary = os.environ.get("CUDA_LESSON_BINARY")
        if not binary:
            result.update(status="requires_cuda_compiler",
                          action="Compile kernels/vector_reduce.cu, set CUDA_LESSON_BINARY; see course.")
        else:
            completed = subprocess.run([binary], check=True, capture_output=True, text=True, timeout=30)
            result["cuda_program"] = json.loads(completed.stdout)
    elif day in (17, 18, 19, 21):
        if device.type != "cuda":
            result.update(status="requires_cuda", action="Run this lesson with device='cuda'.")
            return result
        from learning.kernels import triton_ops as ops
        if day == 17:
            x, y = torch.randn(1003, device=device), torch.randn(1003, device=device)
            torch.testing.assert_close(ops.add(x, y), x + y)
            rows = torch.randn(3, 127, device=device)
            expected = rows * torch.rsqrt(rows.square().mean(-1, keepdim=True) + 1e-6)
            torch.testing.assert_close(ops.row_op(rows), expected, atol=1e-5, rtol=1e-5)
            result["add_and_rmsnorm"] = "passed"
        elif day == 18:
            rows = torch.randn(3, 127, device=device)
            torch.testing.assert_close(ops.row_op(rows, "softmax"), rows.softmax(-1))
            torch.testing.assert_close(ops.row_op(rows, "layernorm"),
                                       F.layer_norm(rows, (127,), eps=1e-6), atol=1e-5, rtol=1e-5)
            result["softmax_and_layernorm"] = "passed"
        elif day == 19:
            a, b = torch.randn(65, 33, device=device), torch.randn(33, 49, device=device)
            expected = (a.cpu().double() @ b.cpu().double()).to(device=device, dtype=a.dtype)
            result["tiles"] = {}
            for tile in (16, 32, 64):
                torch.testing.assert_close(ops.matmul(a, b, tile), expected, atol=1e-4, rtol=1e-4)
                result["tiles"][str(tile)] = measure(lambda: ops.matmul(a, b, tile), device)
        else:
            result["attention"] = _attention_lab(device, gpu=True)
            g, u = torch.randn(1003, device=device), torch.randn(1003, device=device)
            torch.testing.assert_close(ops.swiglu(g, u), F.silu(g) * u, atol=1e-5, rtol=1e-5)
            result["fused_swiglu"] = "passed"
    elif day == 20:
        result["attention"] = _attention_lab(device, gpu=device.type == "cuda")
    elif day == 22:
        endpoint = os.environ.get("VLLM_BASE_URL")
        if endpoint:
            from learning.vllm_probe import probe
            result["external_service"] = probe(endpoint)
        else:
            result.update(status="external_service_not_configured",
                          action="Deploy isolated vLLM and set VLLM_BASE_URL; no service success claimed.")
    elif day in (23, 25):
        pool = PagedPool(Config(), blocks=4, block_size=2, device=device)
        pool.reserve("A", 2)
        pool.reserve("B", 2)
        pool.release("A")
        state = pool.reserve("C", 6)
        result.update(block_table=state.table, logical_mapping=[state.locate(i) for i in range(6)])
        try:
            pool.reserve("OOM", 1)
            raise AssertionError("Allocation should fail.")
        except MemoryError:
            result["atomic_oom"] = "passed"
        pool.release("B")
        pool.release("C")
        assert len(pool.free) == 4
        result["reclaimed_blocks"] = len(pool.free)
    elif day in (26, 27, 30):
        result["engine"] = _engine_lab(device, timed=day in (27, 30))
        if day == 30:
            result["cache"] = _cache_lab(device)
            result["scope"] = "integrated reference, not a production vLLM implementation"
    elif day == 28:
        model = make_model(device)
        store = PrefixStore(model)
        ids = torch.tensor([[1, 2, 3]], device=device)
        _, _, first = store.prefill(ids)
        cache, _, second = store.prefill(ids)
        suffix = torch.tensor([[4]], device=device)
        torch.testing.assert_close(model(suffix, cache)[:, -1],
                                   model(torch.cat((ids, suffix), 1))[:, -1], atol=1e-5, rtol=1e-5)
        fresh, _, _ = store.prefill(ids)
        assert fresh.length == 3
        result.update(first_hit=first, second_hit=second, continuation_alignment="passed",
                      isolation="passed", scope="exact prefix snapshots; no COW or weight invalidation")
    elif day == 29:
        result.update(torch=torch.__version__, seed=123, configuration=Config().__dict__,
                      report_template=["question", "hypothesis", "command", "input", "device",
                                       "dtype", "tests", "timing", "limitations", "next_step"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=int, choices=list(LESSONS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.all == (args.day is not None):
        parser.error("Select exactly one of --day or --all.")
    reports = [run_day(day, args.device) for day in (LESSONS if args.all else [args.day])]
    text = json.dumps(reports, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
