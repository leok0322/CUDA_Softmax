#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ── softmax_kernel：朴素实现，每个线程独立扫描整行 ──────────────────────────
//
// 问题定义：对形状 [h, w] 的矩阵 a，对每一行做 softmax，结果写入 b
//   softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
//   先减去行最大值再取 exp，是数值稳定化手段：
//     若不减 max，当 x_i 较大时 exp(x_i) 上溢为 inf，结果为 NaN
//     减去 max 后指数最大为 exp(0)=1，不会上溢，且 softmax 值不变（分子分母同比缩放）
//
// 线程映射（由 softmax_cu 的 dispatcher 决定）：
//   block: (32, 32)       grid: (w/32, h/32)
//   threadIdx.x → col    threadIdx.y → row
//   每个线程负责计算一个输出元素 b[row, col]
//
// template <typename scalar_t>：
//   由 AT_DISPATCH_FLOATING_TYPES 在运行时展开，scalar_t 实际为 float 或 double
//   模板化避免为每种类型写重复代码，编译器为每种类型生成独立的 PTX/SASS
//
// __restrict__：
//   告知编译器 a 和 b 指针不存在内存别名（不指向同一块内存）
//   允许编译器跳过"写 b 是否影响读 a"的保守检查，生成更激进的访存优化代码

template <typename scalar_t>
__global__ void softmax_kernel(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  // col / row：当前线程负责的输出列号和行号
  //   blockIdx.x * blockDim.x + threadIdx.x = (block列号)*32 + 线程列偏移
  //   blockIdx.y * blockDim.y + threadIdx.y = (block行号)*32 + 线程行偏移
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;

  // 边界检查：grid 按 w/32, h/32 整数除法划分（向下截断）
  //   若 w 或 h 不是 32 的倍数，边缘线程从未启动（欠覆盖），此处 if 只保护已启动线程不越界
  //   欠覆盖示例：w=100 → grid.x=3，只覆盖前 96 列，最后 4 列永远不计算
  if (row < h && col < w)
  {
    // ── Pass 1：扫描整行求最大值 ───────────────────────────────────────────
    // a[row*w + i]：第 row 行第 i 列的元素（行优先存储）
    // fmaxf：单精度浮点 max，对应硬件 FMAX 指令，比 if/else 更快（无分支）
    float maxval = a[row*w];
    for (int i = 1; i<w; i++)
    {
      maxval = fmaxf(maxval, a[row*w + i]);
    }

    // ── Pass 2：扫描整行求 exp 之和（softmax 分母）──────────────────────────
    // __expf：CUDA 内置单精度快速近似 exp，精度约 2 ulp，比标准 expf 快约 2-4 倍
    //   对应 GPU 硬件的 MUFU.EX2 指令（2^x 近似，内部将 e^x 转换为 2^(x/ln2)）
    //   --use_fast_math 会将 expf 自动替换为 __expf；本项目直接调用 __expf
    float divisor = 0.f;
    for (int i = 0; i<w; i++)
    {
      divisor += __expf(a[row*w + i] - maxval);
    }

    // ── Pass 3：计算当前线程负责的那一个输出元素 ────────────────────────────
    // 只写 b[row*w + col]，即 (row, col) 这一个位置
    b[row*w + col] = __expf(a[row*w + col]-maxval)/(divisor);
  }
  // ── 性能瓶颈分析 ─────────────────────────────────────────────────────────
  // 每个线程独立执行 Pass1+Pass2，总内存读取量为 O(w²)：
  //   同一行的 w 个线程各自读取整行 w 个元素，共读 w*w 次
  // 同一行内所有线程计算结果完全相同的 maxval 和 divisor，被重复计算 w 次
  // 后续 kernel2~10 通过线程协作 + shared memory / warp shuffle 将此降至 O(w)
}


