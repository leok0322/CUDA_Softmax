#pragma once

#include <cuda_runtime.h>
#include <cuda/cmath>

#ifndef BLOCK_DIM_X
#define BLOCK_DIM_X 1024
#endif

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 4
#endif

#ifndef URF_DEFINED
#define URF_DEFINED
constexpr int URF {UNROLL_FACTOR};
#endif



template <typename scalar_t>
__global__ void softmax_kernel_7(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  // kernel7 相比 kernel6 的改进：
  // 1. row=blockIdx.x：每个 block 专属一行，全部 BLOCK_DIM_X 线程服务同一行
  // 2. __shfl_down_sync 替代 __shfl_xor_sync：语义上只需 lane0 持有结果写 smem，
  //    但 SIMT 模型下 32 个 lane 仍执行相同指令，与 xor 指令数相同，无性能差异
  // 3. shuffle 手动展开（5条立即数指令）：delta 是编译期常数，生成最优 PTX 指令
  // 4. #pragma unroll URF：主循环展开，occupancy低时MLP补偿TLP不足；始终减少循环开销
  int row = blockIdx.x;
  int ty = threadIdx.y;
  int warp_id = ty/32;
  __shared__ float reduction[BLOCK_DIM_X/32];
  if (row < h)
  {
    float maxval = 0;
    // Pass1：float4 向量化加载，#pragma unroll URF 展开提高 ILP
#pragma unroll URF
    for (int i = ty; i<w/4; i+=BLOCK_DIM_X)
    {
        float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
        maxval = fmaxf(maxval, val.x);
        maxval = fmaxf(maxval, val.y);
        maxval = fmaxf(maxval, val.z);
        maxval = fmaxf(maxval, val.w);
    }
    // Level1：warp 内 __shfl_down_sync 规约，5步后 lane0 持有 warp 内最大值
    // 手动展开保证 delta 是编译期立即数，等价于加 #pragma unroll 的循环
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 16, 32));
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 8, 32));
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 4, 32));
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 2, 32));
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 1, 32));

    // 只有 lane0（ty%32==0）持有 warp 结果，写入 smem
    if (ty%32 == 0)
    {
      reduction[warp_id] = maxval;
    }
    // 跨 warp 同步：保证所有 warp 的 lane0 写完 smem 后 warp0 才读
    __syncthreads();
    // Level2：warp0 读 32 个 warp 结果，再做一轮 __shfl_down_sync
    // BLOCK_DIM_X=1024 → 32个warp结果，恰好填满 warp0 的 32 个 lane
    if (warp_id == 0)
    {
        maxval = ty < BLOCK_DIM_X/32 ? reduction[ty] : 0;
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 16, 32));
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 8, 32));
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 4, 32));
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 2, 32));
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 1, 32));
    }
    // ty==0（全局 lane0）将全局最大值写回 smem，供所有线程读取
    if (ty == 0)
    {
        reduction[0] = maxval;
    }
    // 跨 warp 同步：lane0 写完后所有线程才能读 reduction[0]
    __syncthreads();
    maxval = reduction[0];

    // Pass2：计算 exp 之和（divisor），结构与 max 规约完全对称
    float divisor = 0.f;
#pragma unroll URF
    for (int i = ty; i<w/4; i+=BLOCK_DIM_X)
    {
        float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
        divisor += __expf(val.x - maxval);
        divisor += __expf(val.y - maxval);
        divisor += __expf(val.z - maxval);
        divisor += __expf(val.w - maxval);
    }

    divisor += __shfl_down_sync(0xffffffff, divisor, 16, 32);
    divisor += __shfl_down_sync(0xffffffff, divisor, 8, 32);
    divisor += __shfl_down_sync(0xffffffff, divisor, 4, 32);
    divisor += __shfl_down_sync(0xffffffff, divisor, 2, 32);
    divisor += __shfl_down_sync(0xffffffff, divisor, 1, 32);

    if (ty%32 == 0)
    {
      reduction[warp_id] = divisor;
    }

    __syncthreads();
    if (warp_id == 0)
    {
        divisor = ty < BLOCK_DIM_X/32 ? reduction[ty] : 0;
        divisor += __shfl_down_sync(0xffffffff, divisor, 16, 32);
        divisor += __shfl_down_sync(0xffffffff, divisor, 8, 32);
        divisor += __shfl_down_sync(0xffffffff, divisor, 4, 32);
        divisor += __shfl_down_sync(0xffffffff, divisor, 2, 32);
        divisor += __shfl_down_sync(0xffffffff, divisor, 1, 32);
    }
    if (ty == 0)
    {
        reduction[0] = divisor;
    }

    __syncthreads();
    divisor = reduction[0];

    // Pass3：写出 softmax 结果，float4 向量化写，#pragma unroll URF 展开
#pragma unroll URF
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

template <typename scalar_t,typename scalar_t4,typename scalar_i>
__global__ void softmax_kernel_using_shfl_down_sync_and_unroll(scalar_t* __restrict__ a, scalar_t* __restrict__ b, const scalar_i totalRow, const scalar_i totalCol) {
    // 该block负责的起始行
    const scalar_i initRow { blockIdx.y * blockDim.y };
    // 该线程负责的行
    const scalar_i row { threadIdx.y };
    // 该线程的线程号
    const scalar_i threadIDX { threadIdx.x };
    // 该block的线程数
    const scalar_i threadNum { blockDim.x };
    // 静态SMEM的大小
    const uint reduc { BLOCK_DIM_X / 32 };

    static_assert(BLOCK_DIM_X / 32 == 32 && "线程数不是1024");

    if (initRow + row < totalRow) {
        // 静态SMEM
        __shared__ scalar_t reduction[reduc];

        // 求最大值
        scalar_t maxVal {-INFINITY};
        // 串行规约到每个线程的maxVal变量
#pragma unroll URF
        for (scalar_i i {threadIDX}; i < totalCol / 4; i+=threadNum) {
            // 16字节对齐
            scalar_t4 vecA { reinterpret_cast<scalar_t4*>(&a[(initRow + row) * totalCol + i * 4])[0] };
            maxVal = fmaxf(maxVal, vecA.x);
            maxVal = fmaxf(maxVal, vecA.y);
            maxVal = fmaxf(maxVal, vecA.z);
            maxVal = fmaxf(maxVal, vecA.w);
        }

        // warp树形规约
        // #pragma unroll 只作用于紧跟其后的第一个循环，不影响其他循环
        // 作用：展开后 i 变成编译期字面量（16,8,4,2,1），__shfl_down_sync 的 delta
        //       必须是编译期常数才能生成 shfl.sync.down 立即数 PTX 指令；
        //       若不展开，i 是运行期变量，编译器无法生成最优指令
#pragma unroll
        for (scalar_i i {16}; i>=1; i/=2) {
            maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff,maxVal,i,32));
        }
        // 规约结果写入SMEM
        if (threadIDX % 32 == 0) {
            reduction[threadIDX / 32] = maxVal;
        }
        __syncthreads();

        // warp0规约
        // 一个warp内不需要同步
        if (threadIDX < 32) {
            maxVal = reduction[threadIDX];
        }

        if (threadIDX < 32) {
#pragma unroll
            for (scalar_i i {16}; i>=1; i/=2) {
                maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff,maxVal,i,32));
            }
        }

        if (threadIDX == 0) {
            reduction[threadIDX] = maxVal;
        }

        __syncthreads();

        // 所有线程拿到最大值
        maxVal = reduction[0];

        // 求和
        scalar_t divisor {0.0f};
        // 每个线程串行规约
        for (scalar_i i {threadIDX}; i<totalCol/4; i+=threadNum) {
            // 16字节
            scalar_t4 vecA { reinterpret_cast<scalar_t4*>(&a[(initRow + row) * totalCol + i * 4])[0]};
            divisor += __expf(vecA.x - maxVal);
            divisor += __expf(vecA.y - maxVal);
            divisor += __expf(vecA.z - maxVal);
            divisor += __expf(vecA.w - maxVal);
        }
        // warp树形规约，不需要同步
#pragma unroll
        for (scalar_i i {16}; i>=1; i/=2) {
            divisor += __shfl_down_sync(0xffffffff,divisor,i,32);
        }
        // 规约结果写入SMEM
        if (threadIDX % 32 == 0) {
            reduction[threadIDX / 32] = divisor;
        }

        __syncthreads();

        if (threadIDX < 32) {
            divisor = reduction[threadIDX];
        }

        if (threadIDX < 32) {
#pragma unroll
            for (scalar_i i {16}; i>=1; i/=2) {
                divisor += __shfl_down_sync(0xffffffff,divisor,i,32);
            }
        }

        if (threadIDX == 0) {
            reduction[threadIDX] = divisor;
        }
        __syncthreads();

        // 每个线程读取求和
        divisor = reduction[0];


        // 每个线程计算softmax
#pragma unroll URF
        for (scalar_i i {threadIDX}; i<totalCol / 4; i+=threadNum) {
            scalar_t4 vecA { reinterpret_cast<scalar_t4*>(&a[(initRow + row) * totalCol + i * 4])[0] };
            vecA.x = __expf(vecA.x-maxVal)/divisor;
            vecA.y = __expf(vecA.y-maxVal)/divisor;
            vecA.z = __expf(vecA.z-maxVal)/divisor;
            vecA.w = __expf(vecA.w-maxVal)/divisor;
            reinterpret_cast<scalar_t4*>(&b[(initRow + row) * totalCol + i * 4])[0] = vecA;
        }
    }
}