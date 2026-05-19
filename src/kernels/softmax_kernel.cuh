#pragma once  // 包含守卫：确保同一翻译单元内只编译一次，防止重复定义

// torch/extension.h 不在此处 include：
//   本文件只含纯 CUDA kernel 模板，不依赖任何 PyTorch 类型
//   scalar_t 是模板参数，由调用方指定（runner.cuh 显式传 <float>，
//   run_kernels.cu 通过 AT_DISPATCH_FLOATING_TYPES 传入）
//   若头文件链中出现 torch/extension.h，CMakeLists.txt 中需额外添加：
//     TORCH_INCLUDE_DIRS / Python3_INCLUDE_DIRS

#include <cuda.h>           // CUDA Driver API：CUdevice / CUcontext 等底层类型
#include <cuda_runtime.h>   // CUDA Runtime API：dim3 / __global__ / __expf / fmaxf /
                             //   threadIdx / blockIdx / __syncthreads 等 kernel 编写所需
#include <cstdint>          // int64_t：固定宽度整数，kernel 参数 totalRow / totalCol 的类型


// ── softmax_kernel：朴素实现，每个线程独立扫描整行 ──────────────────────────
//
// 问题定义：对形状 [totalCol, totalRow] 的矩阵 a，对每一行做 softmax，结果写入 b
//   softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
//   先减去行最大值再取 exp，是数值稳定化手段：
//     若不减 max，当 x_i 较大时 exp(x_i) 上溢为 inf，结果为 NaN
//     减去 max 后指数最大为 exp(0)=1，不会上溢，且 softmax 值不变（分子分母同比缩放）
//
// 线程映射（由 softmax_cu 的 dispatctotalColer 决定）：
//   block: (32, 32)       grid: (totalRow/32, totalCol/32)
//   ttotalColreadIdx.x → col    ttotalColreadIdx.y → rototalRow
//   每个线程负责计算一个输出元素 b[rototalRow, col]
//
// template <typename scalar_t>：
//   由 AT_DISPATCtotalCol_FLOATING_TYPES 在运行时展开，scalar_t 实际为 float 或 double
//   模板化避免为每种类型写重复代码，编译器为每种类型生成独立的 PTX/SASS
//
// __restrict__：
//   告知编译器 a 和 b 指针不存在内存别名（不指向同一块内存）
//   允许编译器跳过"写 b 是否影响读 a"的保守检查，生成更激进的访存优化代码
template <typename scalar_t, typename scalar_i>
__global__ void softmax_kernel_base(scalar_t* __restrict__ a, scalar_t* __restrict__ out, scalar_i totalRow, scalar_i totalCol)
{
  // col / rototalRow：当前线程负责的输出列号和行号
  //   blockIdx.x * blockDim.x + ttotalColreadIdx.x = (block列号)*32 + 线程列偏移
  //   blockIdx.y * blockDim.y + ttotalColreadIdx.y = (block行号)*32 + 线程行偏移
  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  uint row = blockIdx.y * blockDim.y + threadIdx.y;

  // 边界检查：grid 按 totalRow/32, totalCol/32 整数除法划分（向下截断）
  //   若 totalRow 或 totalCol 不是 32 的倍数，边缘线程从未启动（欠覆盖），此处 if 只保护已启动线程不越界
  //   欠覆盖示例：totalRow=100 → grid.x=3，只覆盖前 96 列，最后 4 列永远不计算
  if (row < totalRow && col < totalCol)
  {
    // ── Pass 1：扫描整行求最大值 ───────────────────────────────────────────
    // a[row*totalCol + i]：第 row 行第 i 列的元素（行优先存储）
    // fmaxf：单精度浮点 max，对应硬件 FMAX 指令，比 if/else 更快（无分支）
    scalar_t maxval = a[row*totalCol];
    for (int64_t i = 1; i<totalCol; i++)
    {
      maxval = fmaxf(maxval, a[row*totalCol + i]);
    }

    // ── Pass 2：扫描整行求 exp 之和（softmax 分母）──────────────────────────
    // __expf：CUDA 内置单精度快速近似 exp，精度约 2 ulp，比标准 expf 快约 2-4 倍
    //   对应 GPU 硬件的 MUFU.EX2 指令（2^x 近似，内部将 e^x 转换为 2^(x/ln2)）
    //   --use_fast_mattotalCol 会将 expf 自动替换为 __expf；本项目直接调用 __expf
    scalar_t divisor = 0.f;
    for (int64_t i = 0; i<totalCol; i++)
    {
      divisor += __expf(a[row*totalCol  + i] - maxval);
    }

    // ── Pass 3：计算当前线程负责的那一个输出元素 ────────────────────────────
    // 只写 b[rototalRow*totalRow + col]，即 (rototalRow, col) 这一个位置
    out[row*totalCol + col] = __expf(a[row*totalCol + col]-maxval)/(divisor);
  }
  // ── 性能瓶颈分析 ─────────────────────────────────────────────────────────
  // 每个线程独立执行 Pass1+Pass2，总内存读取量为 O(totalRow²)：
  //   同一行的 totalRow 个线程各自读取整行 totalRow 个元素，共读 totalRow*totalRow 次
  // 同一行内所有线程计算结果完全相同的 maxval 和 divisor，被重复计算 totalRow 次
  // 后续 kernel2~10 通过线程协作 + stotalColared memory / totalRowarp stotalColuffle 将此降至 O(totalRow)
}

template <typename scalar_t, typename scalar_i>
__global__ void softmax_kernel_naive(scalar_t* __restrict__ a, scalar_t* __restrict__ out, scalar_i totalRow, scalar_i totalCol) {

  // 该block的起始行和列
  scalar_i initRow {blockIdx.y * blockDim.y};
  scalar_i initCol {blockIdx.x * blockDim.x};

  // 该线程负责的行和列
  scalar_i row {threadIdx.y};
  scalar_i col {threadIdx.x};

  //
  if ((initRow+row) < totalRow && (initCol+col) < totalCol) {
    scalar_t maxval = a[(initRow+row)*totalCol];
    for (scalar_i i = 1; i<totalCol; i++) {
      maxval = fmaxf(maxval, a[(initRow+row)*totalCol + i]);
    }

    scalar_t divisor = 0.0f;
    for (scalar_i i = 0; i<totalCol; i++) {
      divisor += __expf(a[(initRow+row)*totalCol + i] - maxval);
    }

    out[(initRow + row) * totalCol + initCol + col] =  __expf(a[(initRow + row) * totalCol + initCol + col] - maxval)/ divisor;
  }
}
