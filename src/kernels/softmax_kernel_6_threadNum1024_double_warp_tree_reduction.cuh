#pragma once

#include <cuda_runtime.h>

#include "common.cuh"    // BLOCK_DIM_X / UNROLL_FACTOR / URF，所有 kernel 共用


template <typename scalar_t>
__global__ void softmax_kernel_6(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  int warp_id = ty/32;
  __shared__ float reduction[BLOCK_DIM_X/32];
  if (row < h)
  {
    float maxval = 0;
    for (int i = ty; i<w/4; i+=BLOCK_DIM_X)
    {
      float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
      maxval = fmaxf(maxval, val.x);
      maxval = fmaxf(maxval, val.y);
      maxval = fmaxf(maxval, val.z);
      maxval = fmaxf(maxval, val.w);
    }
    for (int mask = 16; mask>0; mask/=2)
    {
      maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
    }

    if (ty%32 == 0)
    {
      reduction[warp_id] = maxval;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        maxval = ty < BLOCK_DIM_X/32 ? reduction[ty] : 0;
        for (int mask = 16; mask>0; mask/=2)
        {
          maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
        }
    }
    if (ty == 0)
    {
        reduction[0] = maxval;
    }
    __syncthreads();
    maxval = reduction[0];
    float divisor = 0.f;
    for (int i = ty; i<w/4; i+=BLOCK_DIM_X)
    {
      float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
      divisor += __expf(val.x - maxval);
      divisor += __expf(val.y - maxval);
      divisor += __expf(val.z - maxval);
      divisor += __expf(val.w - maxval);
    }
    for (int mask = 16; mask>0; mask/=2)
    {
      divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
    }

    if (ty%32 == 0)
    {
      reduction[warp_id] = divisor;
    }

    __syncthreads();
    if (warp_id == 0)
    {
        divisor = ty < BLOCK_DIM_X/32 ? reduction[ty] : 0;
        for (int mask = 16; mask>0; mask/=2)
        {
          divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
        }
    }
    if (ty == 0)
    {
        reduction[0] = divisor;
    }

    __syncthreads();
    divisor = reduction[0];

    for (int i = ty; i<w/4; i+=BLOCK_DIM_X)
    {
        float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
        val.x = __expf(val.x-maxval)/divisor;
        val.y = __expf(val.y-maxval)/divisor;
        val.z = __expf(val.z-maxval)/divisor;
        val.w = __expf(val.w-maxval)/divisor;
        reinterpret_cast<float4*>(&b[row*w + i*4])[0] = val;
    }
  }
}

template <typename scalar_t,typename scalar_t4, typename scalar_i>
__global__ void softmax_kernel_threadNum1024_double_warp_tree_reduction(scalar_t* __restrict__ a, scalar_t* __restrict__ b, scalar_i totalRow, scalar_i totalCol) {
  // 该block的起始行
  scalar_i initRow {blockDim.y * blockIdx.y};
  // 该线程负责的行
  scalar_i row {threadIdx.y};
  // 该线程在block内的线程号
  scalar_i threadIDX {threadIdx.x};
  // block内的线程数
  scalar_i threadNum {blockDim.x};


  static_assert(BLOCK_DIM_X / 32 == 32 && "reducNum恰好一个warp");

  if (initRow + row < totalRow ) {
    // 静态SMEM
    const uint reducNum {BLOCK_DIM_X / 32};
    __shared__ scalar_t reduction[reducNum];
    // 求最大值
    scalar_t maxVal {-INFINITY};
    for (scalar_i i {threadIDX}; i < totalCol / 4; i+=threadNum) {
      scalar_t4 vecA { reinterpret_cast<scalar_t4*>(&a[(initRow + row) * totalCol + i * 4])[0] };
      maxVal = fmaxf(maxVal,vecA.x);
      maxVal = fmaxf(maxVal,vecA.y);
      maxVal = fmaxf(maxVal,vecA.z);
      maxVal = fmaxf(maxVal,vecA.w);
    }

    // warp内树形规约
    for (scalar_i i {16}; i>=1; i/=2) {
      maxVal = fmaxf(maxVal, __shfl_xor_sync(0xffffffff, maxVal, i, 32));
    }

    // warp规约结果写入smem
    if ( threadIDX % 32 == 0) {
      reduction[threadIDX / 32] = maxVal;
    }

    __syncthreads();

    // 在warp0内树形规约
    // Bug1(fixed): 原代码：
    //   for (i=16; i>=1; i/=2) {
    //     if (threadIDX < 32)
    //       reduction[threadIDX] = fmaxf(maxVal, __shfl_xor_sync(0xffffffff, reduction[threadIDX+i], i, 32));
    //   }
    //   错误1：__shfl_xor_sync 设计用于对寄存器交换，不是对 smem 索引运算
    //          var=reduction[threadIDX+i] 时，lane k 实际收到 lane(k XOR i) 的 var=reduction[(k XOR i)+i]
    //          i=16: lane0 收到 lane16 的 reduction[16+16]=reduction[32] → 越界（reduction 只有 32 个元素）
    //   错误2：用 maxVal（本 warp 的局部 max）而非 reduction[threadIDX] 作为初始值参与 fmaxf
    //   修复：__shfl_xor_sync 只能 shuffle 寄存器；先将 reduction[threadIDX] 读入寄存器，再对寄存器做 shuffle
    if (threadIDX < 32) {
      maxVal = reduction[threadIDX];
    }

    if (threadIDX < 32) {
      for (scalar_i i {16}; i>=1; i/=2) {
        maxVal = fmaxf(maxVal, __shfl_xor_sync(0xffffffff, maxVal, i, 32));
      }
    }

    // warp0规约结果写入SMEM
    if (threadIDX  == 0) {
      reduction[threadIDX] = maxVal;
    }

    __syncthreads();

    // 每个线程拿到结果
    maxVal = reduction[0];

    // 求和
    scalar_t divisor {0.0f};
    for (scalar_i i {threadIDX}; i<totalCol / 4; i+=threadNum) {
      scalar_t4 vecA { reinterpret_cast<scalar_t4*>(&a[(initRow + row) * totalCol + i * 4])[0] };
      divisor += __expf(vecA.x - maxVal);
      divisor += __expf(vecA.y - maxVal);
      divisor += __expf(vecA.z - maxVal);
      divisor += __expf(vecA.w - maxVal);
    }
    // warp内树形规约
    for (scalar_i i {16}; i>=1; i/=2) {
      divisor += __shfl_xor_sync(0xffffffff, divisor, i, 32);
    }
    // warp规约结果写入smem
    if ( threadIDX % 32 == 0) {
      reduction[threadIDX / 32] = divisor;
    }

    // Bug2(fixed): 原缺少此 __syncthreads()
    //              reduction[1..31] 由 warp1~warp31 写入，warp0 读前必须跨 warp 同步
    __syncthreads();

    // Bug3(fixed): 原代码：
    //   for (i=16; i>=1; i/=2) {
    //     if (threadIDX < 32)
    //       reduction[threadIDX] += __expf(reduction[threadIDX+i] - maxVal);
    //   }
    //   错误1：reduction[threadIDX+i] 同 Bug1，lane k 越界读 reduction[(k XOR i)+i]，i=16 时索引最大=47
    //   错误2：divisor 已是 exp 之和，对其再套 __expf() 完全错误，应直接累加
    //   错误3：没有 __shfl_xor_sync，不是 warp shuffle 规约；在 smem 上做树形加法不仅语义错，且越界
    //   修复：循环外先将 reduction[threadIDX] 读入寄存器 divisor，再用 __shfl_xor_sync 对寄存器累加
    if (threadIDX < 32) {
      divisor = reduction[threadIDX];
    }

    if (threadIDX < 32) {
      for (scalar_i i {16}; i>=1; i/=2) {
        divisor += __shfl_xor_sync(0xffffffff, divisor, i, 32);
      }
    }

    // warp0规约结果写入SMEM
    if (threadIDX == 0) {
      reduction[threadIDX] = divisor;
    }

    __syncthreads();

    divisor = reduction[0];

    //求每个元素的softmax的值
    // Bug4(fixed): 原公式 __expf((x-maxVal)/divisor)，应为 __expf(x-maxVal)/divisor
    for (scalar_i i {threadIDX}; i<totalCol / 4; i+=threadNum) {
      scalar_t4 vecA { reinterpret_cast<float4*>(&a[(initRow + row) * totalCol + i * 4])[0] };
      vecA.x = __expf( vecA.x - maxVal) / divisor;
      vecA.y = __expf( vecA.y - maxVal) / divisor;
      vecA.z = __expf( vecA.z - maxVal) / divisor;
      vecA.w = __expf( vecA.w - maxVal) / divisor;
      reinterpret_cast<float4*>(&b[(initRow + row) * totalCol + i * 4])[0] = vecA;
    }
  }
}