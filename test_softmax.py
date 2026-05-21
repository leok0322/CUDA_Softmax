import sys
import os
import datetime
import torch

# .so 在项目根目录，将其加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softmax_cuda  # 加载 softmax_cuda.cpython-313-x86_64-linux-gnu.so

ROWS    = 4096
COLS    = 4096
REPEATS = 100
WARMUP  = 10

x = torch.rand(ROWS, COLS, device="cuda", dtype=torch.float32)

# ── 1. 正确性验证 ────────────────────────────────────────────────────────────
out = softmax_cuda.softmax_cuda(x)
ref = torch.softmax(x, dim=-1)

match = torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
print(f"correctness: {'PASS' if match else 'FAIL'}")
if not match:
    max_diff = (out - ref).abs().max().item()
    print(f"  max_diff = {max_diff:.2e}")
    sys.exit(1)

# ── 2. 计时函数（用 CUDA Event，精度 ~0.5 us） ───────────────────────────────
def benchmark(fn, warmup=WARMUP, repeats=REPEATS):
    # 预热：消除 JIT / driver 初始化开销
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(start_events, end_events)]  # ms

# ── 3. 运行 100 次 ───────────────────────────────────────────────────────────
times_cuda  = benchmark(lambda: softmax_cuda.softmax_cuda(x))
times_torch = benchmark(lambda: torch.softmax(x, dim=-1))

# ── 4. 统计 ──────────────────────────────────────────────────────────────────
def stats(times, label):
    t = torch.tensor(times)
    print(f"\n{label} ({REPEATS} runs, {ROWS}x{COLS} float32):")
    print(f"  mean   {t.mean().item():.3f} ms")
    print(f"  median {t.median().item():.3f} ms")
    print(f"  min    {t.min().item():.3f} ms")
    print(f"  max    {t.max().item():.3f} ms")
    return t.median().item()

med_cuda  = stats(times_cuda,  "custom CUDA softmax")
med_torch = stats(times_torch, "torch.softmax")

speedup = med_torch / med_cuda
print(f"\nspeedup (torch / custom): {speedup:.2f}x")

# ── 5. 追加写入比较结果 ───────────────────────────────────────────────────────
RESULT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")
RESULT_FILE = os.path.join(RESULT_DIR, "python_test_result.txt")

def fmt_stats(times, label):
    t = torch.tensor(times)
    return (
        f"  {label}:\n"
        f"    mean={t.mean().item():.3f}ms  median={t.median().item():.3f}ms"
        f"  min={t.min().item():.3f}ms  max={t.max().item():.3f}ms"
    )

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
lines = [
    f"\n[{timestamp}]  {ROWS}x{COLS} float32  repeats={REPEATS}",
    fmt_stats(times_cuda,  "custom CUDA"),
    fmt_stats(times_torch, "torch.softmax"),
    f"  speedup (torch/custom): {speedup:.2f}x",
]
with open(RESULT_FILE, "a") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nresult appended to: {RESULT_FILE}")
