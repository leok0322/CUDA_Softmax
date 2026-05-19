#include "runner.cuh"        // 引入函数声明，让编译器验证此文件中的定义与声明签名一致
                             // 链接层面不强制要求（validation.cu 已包含声明，链接器能找到定义），
                             // 但 include 后若签名不一致，编译阶段即报错，比链接阶段的 undefined reference 更易定位

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


#include <cudnn.h>

void run_softmax_kernel_base(uint totalRow, uint totalCol, float* A, float* out) {
  // dim3 block_size = dim3(32, 32, 1);
  // uint grid_x =  cuda::ceil_div(totalRow,32);
  // uint grid_y =  cuda::ceil_div(totalCol,32);
  // dim3 grid_size = dim3(grid_x, grid_y, 1);
  // softmax_kernel_base<float, uint><<<grid_size,block_size>>>(A, out,totalRow, totalCol);
  // cudaCheck(cudaGetLastError());


  // ── 1. 创建 handle ──────────────────────────────────────────────────────
  cudnnHandle_t handle;
  cudnnCreate(&handle);

  // ── 2. 创建并设置 tensor descriptor ────────────────────────────────────
  // 将 m×n 矩阵映射为 NCHW：N=m, C=n, H=1, W=1
  // mode=CHANNEL → 对每个 (N=row, H=1, W=1) 在 C=n 维度做 softmax
  // 即每行 n 个元素独立做 softmax ✓
  cudnnTensorDescriptor_t desc;
  cudnnCreateTensorDescriptor(&desc);
  cudnnSetTensor4dDescriptor(
      desc,
      CUDNN_TENSOR_NCHW,   // 内存布局
      CUDNN_DATA_FLOAT,    // 数据类型
      totalRow,            // N：行数
      totalCol,            // C：列数（softmax 在此维度规约）
      1,                   // H
      1                    // W
  );

  // ── 3. 调用 softmax ─────────────────────────────────────────────────────
  float alpha = 1.0f, beta = 0.0f;
  cudnnSoftmaxForward(
      handle,
      CUDNN_SOFTMAX_ACCURATE,       // 数值稳定算法
      CUDNN_SOFTMAX_MODE_CHANNEL,   // 按行（C 维度）规约
      &alpha,
      desc, A,
      &beta,
      desc, out
  );

  // ── 4. 释放资源 ─────────────────────────────────────────────────────────
  cudnnDestroyTensorDescriptor(desc);
  cudnnDestroy(handle);
}

void run_softmax_kernel_naive(uint totalRow, uint totalCol, float* A, float* out) {
  dim3 block_size = dim3(32, 32, 1);
  uint grid_x =  cuda::ceil_div(totalRow,32);
  uint grid_y =  cuda::ceil_div(totalCol,32);
  dim3 grid_size = dim3(grid_x, grid_y, 1);
  softmax_kernel_naive<float, uint><<<grid_size,block_size>>>(A, out,totalRow, totalCol);
  cudaCheck(cudaGetLastError());
}

void run_softmax_kernel_tree_reduction(uint totalRow, uint totalCol, float* A, float* out) {
  dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
  dim3 grid_size = dim3(1, totalRow, 1);
  softmax_kernel_tree_reduction<float, uint><<<grid_size,block_size>>>(A, out,totalRow, totalCol);
  cudaCheck(cudaGetLastError());
}


void run_softmax_kernel_warp_tree_reduction(uint totalRow, uint totalCol, float* A, float* out) {
  dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
  dim3 grid_size = dim3(1, totalRow, 1);
  softmax_kernel_warp_tree_reduction<float, uint><<<grid_size,block_size>>>(A, out,totalRow, totalCol);
  cudaCheck(cudaGetLastError());
}