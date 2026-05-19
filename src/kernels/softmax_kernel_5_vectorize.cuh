#pragma once

#include <cuda_runtime.h>

#ifndef BLOCK_DIM_X
#define BLOCK_DIM_X 1024
#endif

template <typename scalar_t>
__global__ void softmax_kernel_5(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  __shared__ float reduction[BLOCK_DIM_X/2];
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

    if (ty >= BLOCK_DIM_X/2)
    {
      reduction[ty - BLOCK_DIM_X/2] = maxval;
    }
    for(int stride = BLOCK_DIM_X/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        maxval = fmaxf(maxval, reduction[ty]);
        if (ty >= stride/2)
        {
          reduction[ty - stride/2] = maxval;
        }
      }
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

    if (ty >= BLOCK_DIM_X/2)
    {
      reduction[ty - BLOCK_DIM_X/2] = divisor;
    }

    for(int stride = BLOCK_DIM_X/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        divisor = divisor + reduction[ty];
        if (ty >= stride/2)
        {
          reduction[ty - stride/2] = divisor;
        }
      }
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


template <typename scalar_t, typename scalar_t4,typename scalar_i>
__global__ void softmax_kernel_vectorize(scalar_t* __restrict__ a, scalar_t* __restrict__ b, scalar_i totalRow, scalar_i totalCol) {
  // 该block负责的起始行
  scalar_i initRow {blockIdx.y * blockDim.y};
  // 该线程负责的行
  scalar_i row {threadIdx.y};
  // 该线程的线程号
  scalar_i threadIDX {threadIdx.x};
  // 该block的线程总数
  scalar_i threadNum {blockDim.x};


  if (initRow+ row < totalRow) {
    // 该block的静态SMEM
    static_assert(BLOCK_DIM_X % 32 == 0 && "线程数不能包含完整的warp");
    // constexpr scalar_i reducNum 不能用作数组大小：
    //   scalar_i 是模板类型参数，编译器将 constexpr scalar_i 视为"依赖类型的常量"，
    //   无法在实例化前确认其是整型常量表达式（integer constant expression），
    //   nvcc 因此把 reduction[reducNum] 当作 VLA（运行期可变长数组）；
    //   而 __shared__ 具有静态存储期，C++/CUDA 均禁止 VLA 拥有静态存储期。
    //   改用 constexpr uint（非模板具体类型）后，编译器可确认其为整型常量表达式，
    //   VLA 问题消失。
    constexpr uint reducNum {BLOCK_DIM_X / 32};
    __shared__ scalar_t reduction[reducNum];

    // 求行最大值
    scalar_t maxval {-INFINITY};
    // Bug1(fixed): 地址原为 threadIDX*4（常数），所有迭代读同一块；改为 i*4
    //   注：totalCol <= 4*threadNum 时循环只跑一次，i==threadIDX，两者相等，此 bug 不显现
    for (scalar_i i {threadIDX}; i  < totalCol / 4; i+=threadNum) {
      // 向量化加载的优点和要求：
      // · 每条 load 指令传输 16 bytes（float4）而非 4 bytes（float），相同数据量指令数 ÷4
      // · 指令数减少 → 调度器空出 issue slot → 更多 FMA 得以发射 → FMA 掩盖 load 延迟
      // · 循环次数 ÷4 → loop branch/counter 开销降低
      // · 地址计算次数 ÷4 → AGEN 压力降低
      // 要求：totalCol % 4 == 0（保证每行起始地址 16 字节对齐）且访问连续（合并事务）
      scalar_t4 vecA { reinterpret_cast<scalar_t4*>(&a[(initRow + row) * totalCol + i * 4])[0] };
      // Bug2(fixed): 原为 maxval = vecA.x，覆盖上一轮累积的 maxval；改为 fmaxf
      maxval = fmaxf(maxval, vecA.x);
      maxval = fmaxf(maxval, vecA.y);
      maxval = fmaxf(maxval, vecA.z);
      maxval = fmaxf(maxval, vecA.w);
    }

    for (scalar_i i {16}; i>=1; i/=2) {
      maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff,maxval,i,32));
    }

    if (threadIDX % 32 == 0) {
      reduction[threadIDX / 32] = maxval;
    }

    __syncthreads();

    for (scalar_i i {reducNum / 2}; i>=1; i/=2) {
      if (threadIDX < i) {
        reduction[threadIDX] = fmaxf(reduction[threadIDX], reduction[threadIDX + i]);
      }
      __syncthreads();
    }

    maxval = reduction[0];

    // 求和
    scalar_t divisor {0.0f};
    // Bug3(fixed): 地址原为 threadNum*4（block总线程数×4，固定偏移）；改为 i*4
    //   totalCol=128 时 threadNum*4=4096，越界读到第 32 行以后的数据，
    //   divisor 为错误值，归一化结果趋近 0 → "Is 0.00"
    for (scalar_i i {threadIDX}; i < totalCol / 4; i+=threadNum) {
      scalar_t4 vecA {reinterpret_cast<scalar_t4*>(&a[(initRow + row) * totalCol + i * 4])[0]};
      divisor += __expf(vecA.x - maxval);
      divisor += __expf(vecA.y - maxval);
      divisor += __expf(vecA.z - maxval);
      divisor += __expf(vecA.w - maxval);
    }

    for (scalar_i i {16}; i>=1; i/=2) {
      divisor += __shfl_xor_sync(0xffffffff,divisor,i,32);
    }

    if (threadIDX % 32 == 0) {
      reduction[threadIDX / 32] = divisor;
    }

    __syncthreads();

    for (scalar_i i {reducNum / 2}; i>=1; i/=2) {
      if (threadIDX < i) {
        reduction[threadIDX] += reduction[threadIDX + i];
      }
      __syncthreads();
    }

    divisor = reduction[0];

    // 计算每个元素的softmax
    for (scalar_i i {threadIDX}; i <totalCol / 4; i+=threadNum) {
      // Bug4(fixed): 原从 b（未初始化的输出）读数据，exp(垃圾值)/divisor ≈ 0；改为从 a 读
      //              地址原为 threadIDX*4（常数），同 Bug1；改为 i*4
      scalar_t4 vecA { reinterpret_cast<scalar_t4*>(&a[(initRow + row) * totalCol + i * 4])[0] };
      vecA.x = __expf(vecA.x - maxval)/ divisor;
      vecA.y = __expf(vecA.y - maxval)/ divisor;
      vecA.z = __expf(vecA.z - maxval)/ divisor;
      vecA.w = __expf(vecA.w - maxval)/ divisor;
      // Bug5: 写出地址用 threadIDX*4（常数），循环多次时所有迭代覆盖同一位置，仅最后一次生效；应为 i*4
      reinterpret_cast<scalar_t4*>(&b[(initRow + row) * totalCol + i * 4])[0] =  vecA;
    }
  }
}