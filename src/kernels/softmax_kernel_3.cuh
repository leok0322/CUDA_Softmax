#pragma once


#include <cuda_runtime.h>

#include "common.cuh"    // BLOCK_DIM_X / UNROLL_FACTOR / URF，所有 kernel 共用


template <typename scalar_t>
__global__ void softmax_kernel_3(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  __shared__ float reduction[BLOCK_DIM_X];
  if (row < h)
  {
    float maxval = 0;
    for (int i = ty; i<w; i+=BLOCK_DIM_X)
    {
      maxval = fmaxf(maxval, a[row*w + i]);
    }

    reduction[ty] = maxval;
    for(int stride = BLOCK_DIM_X/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        reduction[ty] = fmaxf(reduction[ty], reduction[ty+stride]);
      }
    }

    __syncthreads();
    maxval = reduction[0];

    float divisor = 0.f;
    for (int i = ty; i<w; i+=BLOCK_DIM_X)
    {
      divisor += __expf(a[row*w + i] - maxval);
    }
    reduction[ty] = divisor;
    for(int stride = BLOCK_DIM_X/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        reduction[ty] = reduction[ty] + reduction[ty+stride];
      }
    }
    __syncthreads();
    divisor = reduction[0];

    for (int i = ty; i<w; i+=BLOCK_DIM_X)
    {
      b[row*w + i] = __expf(a[row*w + i]-maxval)/divisor;
    }
  }
}


