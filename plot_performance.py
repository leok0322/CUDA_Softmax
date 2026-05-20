#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
# 上方 `# /// script` 块是 PEP 723 定义的 inline script metadata：
#   uv run plot_performance.py        → uv 解析此块，自动创建隔离环境并安装 matplotlib，再运行脚本
#   uv run python plot_performance.py → uv 把 python 当可执行命令，不解析此块，不自动安装依赖
#   python plot_performance.py        → 解释器直接忽略此块，需手动 pip install matplotlib
#
# 安装的包缓存在 ~/.cache/uv/，不会自动删除，下次运行直接复用；手动清理：uv cache clean
"""
用法：
  uv run plot_performance.py 512 1024 2048 4096
  uv run plot_performance.py --all
  uv run plot_performance.py            # 默认绘制全部维度
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = Path(__file__).parent / "benchmark_results"
ALL_DIMS = [128, 256, 512, 1024, 2048, 4096]
KERNELS = list(range(0, 11))  # kernel 0（base reference）到 kernel 10

KERNEL_LABELS = {
    0:  "K0  Base",
    1:  "K1  Naive",
    2:  "K2  SMEM Tree",
    3:  "K3  SMEM v2",
    4:  "K4  Warp Shfl",
    5:  "K5  Float4",
    6:  "K6  1024T DblWarp",
    7:  "K7  Unroll+Shfl",
    8:  "K8  Online Softmax",
    9:  "K9  Shfl v2",
    10: "K10 Reg Store",
}


def parse_results(kernel_id: int) -> dict[int, float]:
    """解析 softmax_kernel_{id}_result.txt，返回 {dimension: gflops}。"""
    path = RESULTS_DIR / f"softmax_kernel_{kernel_id}_result.txt"
    if not path.exists():
        return {}
    # 匹配形如 "performance: ( 11.8 ) GFLOPS. size: (128)." 的行
    pattern = re.compile(
        r"performance:\s*\(\s*([\d.]+)\s*\)\s*GFLOPS\.\s*size:\s*\((\d+)\)"
    )
    results = {}
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            gflops, size = float(m.group(1)), int(m.group(2))
            results[size] = gflops  # 同一 size 多次出现时取最后一行
    return results


def main():
    parser = argparse.ArgumentParser(description="绘制各 softmax kernel 在指定维度上的 GFLOPS 折线图")
    parser.add_argument(
        "dims",
        nargs="*",
        type=int,
        metavar="DIM",
        help=f"要绘制的矩阵维度，可选值：{ALL_DIMS}（不填则使用全部维度）",
    )
    parser.add_argument("--all", action="store_true", help="使用全部维度（与不填参数等价）")
    args = parser.parse_args()

    dims = sorted(set(args.dims)) if args.dims and not args.all else ALL_DIMS

    invalid = [d for d in dims if d not in ALL_DIMS]
    if invalid:
        print(f"错误：不支持的维度 {invalid}，可选值为 {ALL_DIMS}", file=sys.stderr)
        sys.exit(1)

    data: dict[int, dict[int, float]] = {}
    for kid in KERNELS:
        parsed = parse_results(kid)
        if parsed:
            data[kid] = parsed
        else:
            print(f"警告：未找到 kernel {kid} 的结果文件，跳过", file=sys.stderr)

    if not data:
        print("错误：没有找到任何结果文件", file=sys.stderr)
        sys.exit(1)

    # X 轴为 kernel 1-10，K0 作为水平基准线
    plot_kernels = list(range(1, 11))
    colors = [plt.cm.tab10(i) for i in range(len(dims))]

    fig, ax = plt.subplots(figsize=(13, 7))

    for i, dim in enumerate(dims):
        color = colors[i]
        ys = [data.get(kid, {}).get(dim) for kid in plot_kernels]
        # 过滤掉 None（该 kernel 无此维度数据），折线在缺失处自动断开
        valid = [(kid, v) for kid, v in zip(plot_kernels, ys) if v is not None]
        if not valid:
            continue
        xs, ys_valid = zip(*valid)
        ax.plot(xs, ys_valid, label=f"{dim}×{dim}", marker="o",
                linewidth=1.8, markersize=5, color=color)

        # K0 base reference：与该维度同色的水平虚线，方便对比各 kernel 与基准的差距
        baseline = data.get(0, {}).get(dim)
        if baseline is not None:
            ax.axhline(baseline, linestyle="--", linewidth=1.0, color=color, alpha=0.5)

    ax.set_xlabel("Kernel", fontsize=12)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=12)
    ax.set_title("CUDA Softmax Kernel Performance", fontsize=14)
    ax.set_xticks(plot_kernels)
    ax.set_xticklabels([KERNEL_LABELS[k] for k in plot_kernels], rotation=25, ha="right")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], linestyle="--", linewidth=1.0, color="gray", alpha=0.7))
    labels.append("K0 base reference")
    ax.legend(handles, labels, loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = Path(__file__).parent / "performance_plot.png"
    plt.savefig(out, dpi=150)
    print(f"已保存：{out}")
    plt.show()


if __name__ == "__main__":
    main()
