#pragma once

#include <cuda_runtime.h>   // dim3：三维网格/block 形状结构体，kernel 启动语法 <<<>>> 所需
                             // uint（unsigned int）：经由 cuda_runtime.h → vector_types.h 定义
#include <cuda/cmath>       // cuda::ceil_div(a, b)：向上整除，等价于 (a + b - 1) / b
                             //   来自 libcudacxx（CUDA 11+ 附带的 C++ 标准库移植）
                             //   不属于经典 CUDA Runtime API（cuda_runtime.h 不含此函数）
                             //   若 CUDA 版本 < 11，需改为手写 (N + 31) / 32

#include "kernels.cuh"      // 转发头：#include "kernels/softmax_kernel.cuh"
                             //   引入所有 softmax kernel 函数模板的定义
                             //   （softmax_kernel_base / softmax_kernel_naive 等）
                             //   kernel 函数模板必须在调用点可见，不能只有声明，
                             //   因此通过头文件而非 .cu 文件暴露
#include "error_check.cuh"  // cudaCheck 宏：包装 CUDA API 调用，失败时打印位置并终止
                             //   cudaCheck(cudaGetLastError()) 用于检测 kernel 启动错误

void run_softmax_kernel_base(int M, int N, float* A, float* out) {
      dim3 block_size = dim3(32, 32, 1);
      uint grid_x =  cuda::ceil_div(M,32);
      uint grid_y =  cuda::ceil_div(N,32);
      dim3 grid_size = dim3(grid_x, grid_y, 1);
      softmax_kernel_base<float><<<grid_size,block_size>>>(A, out,M, N);
      cudaCheck(cudaGetLastError());
}

void run_softmax_kernel_naive(int M, int N, float* A, float* out) {
      dim3 block_size = dim3(32, 32, 1);
      uint grid_x =  cuda::ceil_div(M,32);
      uint grid_y =  cuda::ceil_div(N,32);
      dim3 grid_size = dim3(grid_x, grid_y, 1);
      softmax_kernel_naive<float><<<grid_size,block_size>>>(A, out,M, N);
      cudaCheck(cudaGetLastError());
}