#pragma once

// 函数声明（无函数体）：只告知编译器函数签名，定义在 runner.cu
// 头文件放声明而非定义：多个翻译单元 #include 同一头文件时，
//   定义（有函数体）会产生多份，违反 ODR，链接报 "multiple definition"
//   声明（无函数体）不产生目标代码，多份声明合法
void run_softmax_kernel_base(uint totalRow, uint totalCol, float* A, float* out);

void run_softmax_kernel_naive(uint totalRow, uint totalCol, float* A, float* out);

void run_softmax_kernel_tree_reduction(uint totalRow, uint totalCol, float* A, float* out);

void run_softmax_kernel_warp_tree_reduction(uint totalRow, uint totalCol, float* A, float* out);


void run_softmax_kernel_vectorize(uint totalRow, uint totalCol, float* A, float* out);

void run_softmax_kernel_threadNum1024_double_warp_tree_reduction(uint totalRow, uint totalCol, float* A, float* out);

void run_softmax_kernel_using_shfl_down_sync_and_unroll(uint totalRow, uint totalCol, float* A, float* out);

void run_softmax_kernel_using_online_softmax(uint totalRow, uint totalCol, float* A, float* out);

void run_softmax_kernel_reg_store(uint totalRow, uint totalCol, float* A, float* out);