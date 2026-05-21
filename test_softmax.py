import sys
import os
import datetime
import torch

# .so 在项目根目录，将其加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softmax_cuda  # 加载 softmax_cuda.cpython-313-x86_64-linux-gnu.so

REPEATS = 100
WARMUP  = 10

# 测试的矩阵尺寸列表 (rows, cols)
SIZES = [
    (128,  1024),
    (128,  4096),
    (128,  16384),
    (1024, 1024),
    (1024, 4096),
    (4096, 4096),
]

# ── 计时函数（CUDA Event，精度 ~0.5 us） ─────────────────────────────────────
def benchmark(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    for i in range(REPEATS):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(start_events, end_events)]  # ms

def median_ms(times):
    return torch.tensor(times).median().item()

# ── 按尺寸循环，收集结果 ──────────────────────────────────────────────────────
times_cuda  = []  # 每个元素对应一个尺寸的 100 次耗时列表
times_torch = []

print(f"{'size':<12}  {'cuda mean':>10}  {'cuda median':>12}  {'torch mean':>10}  {'torch median':>12}  {'torch/cuda(median)':>15}  correctness")
print("-" * 95)

results = []  # 用于最终写入文件

for rows, cols in SIZES:
    x = torch.rand(rows, cols, device="cuda", dtype=torch.float32)

    # 正确性验证
    out = softmax_cuda.softmax_cuda(x)
    ref = torch.softmax(x, dim=-1)
    match = torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    if not match:
        max_diff = (out - ref).abs().max().item()
        print(f"{rows}x{cols:<6}  correctness FAIL  max_diff={max_diff:.2e}")
        times_cuda.append(None)
        times_torch.append(None)
        results.append((rows, cols, None, None, False))
        continue

    # 计时
    t_cuda  = benchmark(lambda: softmax_cuda.softmax_cuda(x))
    t_torch = benchmark(lambda: torch.softmax(x, dim=-1))
    times_cuda.append(t_cuda)
    times_torch.append(t_torch)

    tc = torch.tensor(t_cuda)
    tt = torch.tensor(t_torch)
    torch_over_cuda = tt.median().item() / tc.median().item()

    print(f"{rows}x{cols:<6}"
          f"  {tc.mean().item():>9.3f}ms"
          f"  {tc.median().item():>11.3f}ms"
          f"  {tt.mean().item():>9.3f}ms"
          f"  {tt.median().item():>11.3f}ms"
          f"  {torch_over_cuda:>14.2f}x"
          f"  PASS")
    results.append((rows, cols, t_cuda, t_torch, True))

# ── 追加写入结果文件 ──────────────────────────────────────────────────────────
RESULT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")
RESULT_FILE = os.path.join(RESULT_DIR, "python_test_result.txt")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
lines = [f"\n[{timestamp}]  repeats={REPEATS}"]
for rows, cols, t_cuda, t_torch, ok in results:
    if not ok:
        lines.append(f"  {rows}x{cols:<6}  FAIL")
        continue
    tc = torch.tensor(t_cuda)
    tt = torch.tensor(t_torch)
    torch_over_cuda = tt.median().item() / tc.median().item()
    lines.append(
        f"  {rows}x{cols:<6}"
        f"  cuda  mean={tc.mean().item():.3f}ms  median={tc.median().item():.3f}ms"
        f"  torch mean={tt.mean().item():.3f}ms  median={tt.median().item():.3f}ms"
        f"  torch/cuda={torch_over_cuda:.2f}x"
    )

with open(RESULT_FILE, "a") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nresult appended to: {RESULT_FILE}")
