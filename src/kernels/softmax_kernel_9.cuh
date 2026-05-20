#pragma once

#include <cuda_runtime.h>

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
__global__ void softmax_kernel_9(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
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
    #pragma unroll
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
#pragma unroll URF
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

#pragma unroll URF
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
