#pragma once

#include <cuda_runtime.h>



#ifndef BLOCK_DIM_X
#define BLOCK_DIM_X 1024
#endif

// ── kernel4 改进：warp shuffle 替换 shared memory 树形规约 ──────────────────
//
// __shfl_xor_sync 用法：
//   T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=32)
//   · mask    ：位图，bit i=1 表示 lane i 参与此次 shuffle；未参与的 lane 不能作为数据源
//               0xffffffff = 32 个 lane 全部参与（最常见用法）
//               部分 lane 不活跃时（如 if 分支）需将对应 bit 清零，否则行为未定义
//   · var     ：当前 lane 贡献给规约的值（存于寄存器）
//               返回值是对方 lane 的 var，不是自己的；需在返回值上做累加：
//               val += __shfl_xor_sync(0xffffffff, val, 16)  // 收到对方值后自行累加
//   · laneMask：当前 lane ID XOR laneMask = 对方 lane ID（XOR 对称，双方互为来源）
//   · width   ：将 warp 切成大小为 width 的独立子组，shuffle 不跨子组边界
//               width=32（默认）：整个 warp 为一组；width=16：分为 [0~15][16~31] 两组
//               mask 控制"谁参与"，width 控制"交换边界"，两者不重复：
//               mask=0x0000ffff, width=16：只有 lane 0~15 参与，且交换限制在 16 个 lane 内
//               mask=0xffffffff, width=16：32 个 lane 全参与，但交换限制在各自 16 个 lane 内
//   · 效果    ：每个 lane 直接读取对方寄存器中的 var，无需经过 shared memory
//   · 无需 __syncthreads()：warp 内所有线程天然同步执行（lockstep）
//
// XOR 蝶形规约原理（以 8 线程为例）：
//   laneMask 翻转二进制某一位，将 warp 分成两组配对交换：
//     mask=4(100)：翻转 bit2，lane 0↔4, 1↔5, 2↔6, 3↔7
//     mask=2(010)：翻转 bit1，lane 0↔2, 1↔3, 4↔6, 5↔7
//     mask=1(001)：翻转 bit0，lane 0↔1, 2↔3, 4↔5, 6↔7
//   每步覆盖范围翻倍，log₂(32)=5 步后所有 lane 均持有 warp 内全局结果
//
// 两级规约流程（BLOCK_DIM_Y=1024，共 32 个 warp）：
//   Level 1：每个 warp 内 shuffle 规约（5步）→ 各 lane 均得本 warp 结果
//   Level 2：32 个 warp 的 lane0 写入 shared memory → warp0 再做 shuffle
//   共 2 次 __syncthreads()（kernel3 需要 20 次）
// ───────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void softmax_kernel_4(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  int warp_id = ty/32;
  // shared memory 仅需 32 槽（每 warp 1 个），kernel3 需要 BLOCK_DIM_Y=1024 槽
  __shared__ float reduction[BLOCK_DIM_X/32];
  if (row < h)
  {
    float maxval = 0;
    // Pass 1 串行阶段：每线程负责 w/BLOCK_DIM_Y 列，与 kernel3 相同
    for (int i = ty; i<w; i+=BLOCK_DIM_X)
    {
      maxval = fmaxf(maxval, a[row*w + i]);
    }
    // Level 1：warp 内 shuffle 规约求最大值
    // mask=16,8,4,2,1 共 5 步，每步覆盖范围翻倍
    // 结束后 warp 内所有 32 个 lane 均持有本 warp 的最大值
    for (int mask = 16; mask>0; mask/=2)
    {
      // 当前 lane 从 (ty XOR mask) 号 lane 取 maxval，与自身做 fmaxf
      // XOR 对称：若 lane A 的交换对象是 lane B，则 lane B 的交换对象也是 lane A
      // 双方各自执行 fmaxf(自身值, 对方值)，结果相同，无需额外同步
      maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
    }

    // Level 2 第一步：每个 warp 的 lane0 将本 warp 结果写入 shared memory
    // 其余 lane 也持有相同值，但只需 lane0 写入
    if (ty%32 == 0)
    {
      reduction[warp_id] = maxval;
    }
    __syncthreads();  // 确保所有 warp 写入完毕再让 warp0 读取
    // Level 2 第二步：warp0 对 32 个 warp 结果再做一轮 shuffle 规约
    if (warp_id == 0)
    {
        // warp0 的 lane ty 读取 reduction[ty]（第 ty 个 warp 的最大值）
        // ty >= BLOCK_DIM_Y/32 的 lane 不对应真实 warp，填 0（不影响 fmaxf）
        maxval = ty < BLOCK_DIM_X/32 ? reduction[ty] : 0;
        for (int mask = 16; mask>0; mask/=2)
        {
          maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
        }
        // 结束后 warp0 内所有 lane 均持有全局最大值
    }
    // lane0（ty==0）将全局最大值写回 reduction[0]，供所有线程读取
    if (ty == 0)
    {
        reduction[0] = maxval;
    }
    __syncthreads();  // 确保 reduction[0] 写入完毕
    maxval = reduction[0];

    float divisor = 0.f;
    // Pass 2 串行阶段：每线程累加自己负责列的 exp(x - max)
    for (int i = ty; i<w; i+=BLOCK_DIM_X)
    {
      divisor += __expf(a[row*w + i] - maxval);
    }
    // Level 1：warp 内 shuffle 规约求 exp 之和（累加而非 fmaxf）
    for (int mask = 16; mask>0; mask/=2)
    {
      divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
    }

    if (ty%32 == 0)
    {
      reduction[warp_id] = divisor;
    }

    __syncthreads();
    if (warp_id == 0)
    {
        divisor = ty < BLOCK_DIM_X/32 ? reduction[ty] : 0;
        for (int mask = 16; mask>0; mask/=2)
        {
          divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
        }
    }
    if (ty == 0)
    {
        reduction[0] = divisor;
    }

    __syncthreads();
    divisor = reduction[0];

    // Pass 3：写出归一化结果，访问模式与 kernel3 相同（coalesced）
    for (int i = ty; i<w; i+=BLOCK_DIM_X)
    {
      b[row*w + i] = __expf(a[row*w + i]-maxval)/divisor;
    }
  }
}


template <typename scalar_t, typename scalar_i>
__global__ void softmax_kernel_warp_tree_reduction(scalar_t* __restrict__ a, scalar_t* __restrict__ b, scalar_i totalRow, scalar_i totalCol) {
  // block起始的行
  scalar_i initRow { blockIdx.y * blockDim.y };
  scalar_i threadIDX { threadIdx.x };
  scalar_i row { threadIdx.y };
  scalar_i threadNum { blockDim.x };

  // block静态SMEM
  static_assert(BLOCK_DIM_X % 32 == 0 && "一个block的线程数不是warp线程数的倍数");
  // reduc一定小于block中的线程数
  const uint reduc {BLOCK_DIM_X / 32};
  // static_assert(reduc % 32 == 0 && "一个block的warp数warp线程数的倍数");
  __shared__ scalar_t reduction[reduc];

  // 条件判断行是否越界
  if (row + initRow < totalRow) {
    // 求max
    scalar_t maxval = { -INFINITY };
    // 每个线程计算totalCol / threadNum个元素的最大值
    for (scalar_i i {threadIDX}; i<totalCol; i+=threadNum) {
      maxval = fmaxf(maxval, a[(row + initRow) * totalCol + i]);
    }

    // warp 内蝶形规约：5步（i=16,8,4,2,1），每步所有 lane 同时执行
    // 每个 lane：自己的 maxval = fmax(自己的 maxval, 同warp内 threadIdx XOR i 号线程的 maxval)
    // XOR 配对由硬件保证不跨 warp；5步后每个 lane 均持有 warp 内全局最大值
    //
    // 以4个lane（初始 lane0=1, lane1=3, lane2=2, lane3=4）为例：
    // Step1 i=2(10)，翻转bit1，lane0↔lane2, lane1↔lane3：
    //   lane0=fmax(1,2)=2  lane1=fmax(3,4)=4  lane2=fmax(2,1)=2  lane3=fmax(4,3)=4
    //   每个lane持有自己和配对lane的最大值
    // Step2 i=1(01)，翻转bit0，lane0↔lane1, lane2↔lane3：
    //   lane0=fmax(2,4)=4  lane1=fmax(4,2)=4  lane2=fmax(2,4)=4  lane3=fmax(4,2)=4
    //   lane0从lane1读到4（lane1在Step1已汇聚了lane3的值），间接获得从未直接接触的lane3的值
    // 每步覆盖范围翻倍：2步覆盖4个lane，5步覆盖32个lane
    for (scalar_i i {16}; i>=1; i/=2) {
      maxval = fmax(maxval,__shfl_xor_sync(0xffffffff,maxval,i,32));
    }

    if (threadIDX % 32 == 0) {
      reduction[threadIDX / 32] = maxval;
    }

    __syncthreads();

    // 树形规约
    // Bug2(fixed): 原条件 i<totalCol → i/=2 最终到 0，0<totalCol 恒真，无限循环；改为 i>=1
    // Bug3(fixed): 原条件 threadIDX<reduc → threadIDX=reduc-1 时访问 reduction[reduc] 越界；改为 threadIDX<i
    for (scalar_i i {reduc/2}; i>=1; i/=2) {
      if (threadIDX < i) {
        reduction[threadIDX] = fmax(reduction[threadIDX],reduction[threadIDX+i]);
      }
      __syncthreads();
    }

    maxval = reduction[0];

    // 求和
    scalar_t divisor = 0.0f;
    // Bug1(fixed): 原为 a[...+threadIDX]，所有迭代读同一列；改为 +i
    for (scalar_i i {threadIDX}; i<totalCol; i+=threadNum) {
      divisor += __expf(a[(row + initRow) * totalCol + i]-maxval);
    }
    // Bug4(fixed): 原条件 i<totalCol，同 Bug2，无限循环；改为 i>=1
    for (scalar_i i {16}; i>=1; i/=2) {
      divisor += __shfl_xor_sync(0xffffffff,divisor,i,32);
    }

    if (threadIDX % 32 == 0) {
      reduction[threadIDX / 32] = divisor;
    }

    // Bug5(fixed): 原缺此 __syncthreads()，树形规约可能读到未写入的 reduction 值
    __syncthreads();

    // Bug6(fixed): 同 Bug2+Bug3，原条件 i<totalCol 无限循环，原 threadIDX<reduc 越界
    for (scalar_i i {reduc/2}; i>=1; i/=2) {
      if (threadIDX < i) {
        reduction[threadIDX] += reduction[threadIDX + i];
      }
      __syncthreads();
    }

    divisor = reduction[0];

    // Bug7(fixed): 原为 b[...+threadIDX] / a[...+threadIDX]，所有迭代读写同一列；改为 +i
    for (scalar_i i {threadIDX}; i<totalCol; i+=threadNum) {
      b[(row + initRow) * totalCol + i] = __expf(a[(row + initRow) * totalCol + i] - maxval) / divisor;
    }
  }
}
