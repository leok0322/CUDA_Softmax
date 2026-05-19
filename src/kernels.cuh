#pragma once

#include "kernels/softmax_kernel.cuh"
#include "kernels/softmax_kernel_2_SMEM_tree_reduction.cuh"
#include "kernels/softmax_kernel_3.cuh"
#include "kernels/softmax_kernel_4_warp_tree_reduction.cuh"
#include "kernels/softmax_kernel_5_vectorize.cuh"
#include "kernels/softmax_kernel_6_threadNum1024_double_warp_tree_reduction.cuh"
#include "kernels/softmax_kernel_7_using_shfl_down_sync.cuh"
#include "kernels/softmax_kernel_8.cuh"
#include "kernels/softmax_kernel_9.cuh"
#include "kernels/softmax_kernel_10.cuh"
