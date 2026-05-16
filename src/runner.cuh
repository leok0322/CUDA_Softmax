#pragma once

// 函数声明（无函数体）：只告知编译器函数签名，定义在 runner.cu
// 头文件放声明而非定义：多个翻译单元 #include 同一头文件时，
//   定义（有函数体）会产生多份，违反 ODR，链接报 "multiple definition"
//   声明（无函数体）不产生目标代码，多份声明合法
void run_softmax_kernel_base(int M, int N, float* A, float* out);

void run_softmax_kernel_naive(int M, int N, float* A, float* out);