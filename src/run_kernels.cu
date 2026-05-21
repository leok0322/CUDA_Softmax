#include "kernels.cuh"
#include "error_check.cuh"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda/cmath>


#ifndef SOFTMAX_VARIANT
#define SOFTMAX_VARIANT 8
#endif



// torch::Tensor：句柄（handle），不是数据本身
//
//   内部结构：
//     torch::Tensor
//       └─ intrusive_ptr<TensorImpl>   ← 引用计数智能指针，类似 shared_ptr 但侵入式
//            └─ TensorImpl
//                 ├─ Storage           ← 实际 GPU 内存块（data_ptr 指向这里）
//                 ├─ sizes[]           ← 各维度大小，如 [h, w]
//                 ├─ strides[]         ← 各维度步长（字节数），决定内存布局
//                 ├─ dtype             ← 元素类型（float / double 等）
//                 ├─ device            ← cuda:0 / cpu 等
//                 └─ ref_count         ← 引用计数，降为 0 时释放 Storage
//
//   拷贝语义（浅拷贝）：
//     torch::Tensor b = a;   // 只复制句柄，ref_count++，a 和 b 指向同一块 GPU 内存
//     按值传参同理：softmax_cu(torch::Tensor x) 传入时拷贝句柄，开销 O(1)
//
//   深拷贝（需要独立副本时）：
//     torch::Tensor b = a.clone();   // 分配新内存并复制数据，ref_count 各自独立
torch::Tensor softmax_cu(torch::Tensor x)
{
  // auto 推断为 torch::Tensor，与 x 共享同一 TensorImpl 结构（shape/dtype/device），
  // 但 empty_like 分配独立的 Storage（新的 GPU 内存块），ref_count 从 1 开始
  auto out = torch::empty_like(x);
  int64_t dim = x.dim();
  if (dim == 2) {
    int64_t totalRow = x.size(0);
    int64_t totalCol = x.size(1);
    // ── kernel1 的 dispatcher（位于 kernels.cu 的 softmax_cu 函数中）────────────
    //
    // #if SOFTMAX_VARIANT == 1 ... #endif
    //   预处理器条件编译，在编译阶段求值（不是运行时 if）
    //   SOFTMAX_VARIANT 由 target_compile_definitions 注入为 -DSOFTMAX_VARIANT=N
    //   只有值为 1 时这段代码才被编译进二进制，其余变体代码在同一次编译中完全丢弃
    //
    // dim3(32, 32, 1) / dim3(w/32, h/32, 1)
    //   dim3 是 CUDA 内置结构体，三个字段 x, y, z 对应三个维度：
    //     block_size = (32, 32, 1) → 每个 block 有 32×32 = 1024 个线程
    //     grid_size  = (w/32, h/32, 1) → 整个 grid 有 (w/32)×(h/32) 个 block
    //   w/32 是 C++ 整数除法，向下截断，w 不是 32 倍数时产生欠覆盖（见上方边界检查注释）
    #if SOFTMAX_VARIANT == 1
      dim3 block_size = dim3(32, 32, 1);
      uint grid_x =  cuda::ceil_div(totalRow,32);
      uint grid_y =  cuda::ceil_div(totalCol,32);
      dim3 grid_size = dim3(grid_x, grid_y, 1);
      // AT_DISPATCH_FLOATING_TYPES(运行时 dtype, 调试名, lambda)
      // 展开为 switch(x.scalar_type())，每个 case 分支注入编译期类型别名 scalar_t：
      //   case Float:  { using scalar_t = float;  /* lambda 体 */ break; }
      //   case Double: { using scalar_t = double; /* lambda 体 */ break; }
      //
      // x.scalar_type() 是运行时值，但每个 case 分支里的 scalar_t 是编译期常量。
      // 编译器在编译阶段把两个分支都编译，各自实例化一份模板：
      //   softmax_kernel_naive<float>  → 编译进二进制
      //   softmax_kernel_naive<double> → 编译进二进制
      // 运行时 switch 只决定跳到哪个已编译好的实例执行，不存在运行时模板实例化。
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
            softmax_kernel_naive<scalar_t, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
            cudaCheck(cudaGetLastError());
      }));
    #endif

    #if SOFTMAX_VARIANT == 2
      dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
      dim3 grid_size = dim3(1, totalRow, 1);
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
            softmax_kernel_tree_reduction<scalar_t, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
            cudaCheck(cudaGetLastError());
      }));
    #endif

    #if SOFTMAX_VARIANT == 3
      dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
      dim3 grid_size = dim3(1, totalRow, 1);
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
          softmax_kernel_warp_tree_reduction<scalar_t, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
          cudaCheck(cudaGetLastError());
      }));
    #endif

    #if SOFTMAX_VARIANT == 4
      dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
      dim3 grid_size = dim3(1, totalRow, 1);
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
          softmax_kernel_vectorize<scalar_t, float4, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
          cudaCheck(cudaGetLastError());
    }));
    #endif

    #if SOFTMAX_VARIANT == 5
      dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
      dim3 grid_size = dim3(1, totalRow, 1);
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
          softmax_kernel_threadNum1024_double_warp_tree_reduction<scalar_t, float4, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
          cudaCheck(cudaGetLastError());
    }));

    #endif
    #if SOFTMAX_VARIANT == 6
      dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
      dim3 grid_size = dim3(1, totalRow, 1);
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
          softmax_kernel_using_shfl_down_sync_and_unroll<scalar_t, float4, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
          cudaCheck(cudaGetLastError());
    }));
    #endif

    #if SOFTMAX_VARIANT == 7
      dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
      dim3 grid_size = dim3(1, totalRow, 1);
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
          softmax_kernel_online_softmax<scalar_t, float4, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
          cudaCheck(cudaGetLastError());
    }));
    #endif


  }
  if (dim == 3) {
    assert(x.is_contiguous() && "x的内存必须连续");
    int64_t totalBatch {x.size(0)};
    int64_t totalRow {x.size(1)};
    int64_t totalCol {x.size(2)};
    #if SOFTMAX_VARIANT == 1
      block_size = dim3(32, 32, 1);
      uint grid_x { cuda::ceil_div(totalCol,32) };
      uint grid_y { cuda::ceil_div(totalBatch * totalRow,32)};
      grid_size = dim3(grid_x, grid_y, 1);
      // AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
      //   softmax_kernel_naive<scalar_t, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
      //   cudaCheck(cudaGetLastError());
      }));
    #endif

    #if SOFTMAX_VARIANT == 2
      dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
      dim3 grid_size = dim3(1, totalBatch * totalRow, 1);
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
            softmax_kernel_tree_reduction<scalar_t, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
            cudaCheck(cudaGetLastError());
      }));
    #endif

    #if SOFTMAX_VARIANT == 3
      dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
      dim3 grid_size = dim3(1, totalBatch * totalRow, 1);
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
          softmax_kernel_warp_tree_reduction<scalar_t, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
          cudaCheck(cudaGetLastError());
      }));
    #endif

    #if SOFTMAX_VARIANT == 4
      dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
      dim3 grid_size = dim3(1, totalBatch * totalRow, 1);
      assert(totalCol % 4 == 0 && "线程需要刚好能用flot4完全覆盖，并且是16字节对齐");
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
          softmax_kernel_vectorize<scalar_t, float4, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
          cudaCheck(cudaGetLastError());
    }));
    #endif

    #if SOFTMAX_VARIANT == 5
      dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
      dim3 grid_size = dim3(1, totalBatch * totalRow, 1);
      assert(totalCol % 4 == 0 && "线程需要刚好能用flot4完全覆盖，并且是16字节对齐");
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
          softmax_kernel_threadNum1024_double_warp_tree_reduction<scalar_t, float4, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
          cudaCheck(cudaGetLastError());
    }));

    #endif
    #if SOFTMAX_VARIANT == 6
      dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
      dim3 grid_size = dim3(1, totalBatch * totalRow, 1);
      assert(totalCol % 4 == 0 && "线程需要刚好能用flot4完全覆盖，并且是16字节对齐");
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
          softmax_kernel_using_shfl_down_sync_and_unroll<scalar_t, float4, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
          cudaCheck(cudaGetLastError());
    }));
    #endif

    #if SOFTMAX_VARIANT == 7
      dim3 block_size = dim3(BLOCK_DIM_X, 1, 1);
      dim3 grid_size = dim3(1, totalBatch * totalRow, 1);
      assert(totalCol % 4 == 0 && "线程需要刚好能用flot4完全覆盖，并且是16字节对齐");
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
          softmax_kernel_online_softmax<scalar_t, float4, int64_t><<<grid_size,block_size>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),totalRow, totalCol);
          cudaCheck(cudaGetLastError());
    }));
    #endif
  }
  return out;
}
