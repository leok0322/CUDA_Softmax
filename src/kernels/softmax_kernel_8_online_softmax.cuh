#pragma once

#include <cuda_runtime.h>

#ifndef BLOCK_DIM_X
#define BLOCK_DIM_X 1024
#endif


// ── URF 重复定义问题 ────────────────────────────────────────────────────────
// kernels.cuh 将所有 kernel 头文件包含进同一翻译单元（TU）。
// kernel7 和 kernel8 各自在文件作用域定义 constexpr int URF {4}，
// 预处理展开后同一 TU 内出现两次相同名字的定义 → 编译器报重复定义错误。
//
// 【为什么 inline constexpr int URF {4} 不行】
//   inline 允许同一定义出现在多个 TU 中（链接器合并），
//   但不允许同一 TU 同一作用域内出现两次定义，问题依然存在。
//
// 【三种解决方案】
//   方案A（推荐）：函数内定义
//     将 constexpr int URF {4}; 移入每个 kernel 函数体内，
//     #pragma unroll 可识别函数内的 constexpr int，各 kernel 完全独立。
//
//   方案B：公共头文件
//     新建 kernels_common.cuh，集中定义一次 URF，
//     所有 kernel 头文件 include 它；依赖 #pragma once 保证只展开一次。
//
//   方案C（当前采用）：宏 guard 保护
//     第一个被包含的头文件定义 URF 并设置宏标志 URF_DEFINED，
//     后续头文件检测到标志已存在则跳过定义，避免重复。
// ────────────────────────────────────────────────────────────────────────────
#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 4
#endif

#ifndef URF_DEFINED
#define URF_DEFINED
constexpr int URF {UNROLL_FACTOR};
#endif



template <typename scalar_t>
__global__ void softmax_kernel_8(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x;
  int ty = threadIdx.y;
  int warp_id = ty/32;
  __shared__ float reduction_max[BLOCK_DIM_X/32];
  __shared__ float reduction_div[BLOCK_DIM_X/32];
  if (row < h)
  {
    float maxval = 0;
    float divisor = 0;
    float old_maxval = 0;
#pragma unroll URF
    for (int i = ty; i<w/4; i+=BLOCK_DIM_X)
    {
        float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
        maxval = fmaxf(maxval, val.x);
        maxval = fmaxf(maxval, val.y);
        maxval = fmaxf(maxval, val.z);
        maxval = fmaxf(maxval, val.w);
        if (maxval > old_maxval)
        {
          divisor *= __expf(old_maxval - maxval);
          old_maxval = maxval;
        }
        divisor += __expf(val.x - maxval);
        divisor += __expf(val.y - maxval);
        divisor += __expf(val.z - maxval);
        divisor += __expf(val.w - maxval);
    }
    float incoming_divisor = 0;
    float incoming_maxval = 0;
#pragma unroll URF
    for (int mask = 16; mask>0; mask/=2)
    {
      incoming_maxval = __shfl_xor_sync(0xffffffff, maxval, mask, 32);
      incoming_divisor = __shfl_xor_sync(0xffffffff, divisor, mask, 32);
      if (incoming_maxval > maxval)
      {
        divisor *= __expf(maxval - incoming_maxval);
        maxval = incoming_maxval;
      }
      else
      {
        incoming_divisor *= __expf(incoming_maxval - maxval);
      }
      divisor += incoming_divisor;
    }

    if (ty%32 == 0)
    {
      reduction_max[warp_id] = maxval;
      reduction_div[warp_id] = divisor;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        maxval = ty < BLOCK_DIM_X/32 ? reduction_max[ty] : 0;
        divisor = ty < BLOCK_DIM_X/32 ? reduction_div[ty] : 0;
#pragma unroll URF
        for (int mask = 16; mask>0; mask/=2)
        {
          incoming_maxval = __shfl_xor_sync(0xffffffff, maxval, mask, 32);
          incoming_divisor = __shfl_xor_sync(0xffffffff, divisor, mask, 32);
          if (incoming_maxval > maxval)
          {
            divisor *= __expf(maxval - incoming_maxval);
            maxval = incoming_maxval;
          }
          else
          {
            incoming_divisor *= __expf(incoming_maxval - maxval);
          }
          divisor += incoming_divisor;
        }
    }
    if (ty == 0)
    {
        reduction_max[0] = maxval;
        reduction_div[0] = divisor;
    }
    __syncthreads();
    maxval = reduction_max[0];
    divisor = reduction_div[0];

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

template <typename scalar_t, typename scalar_t4, typename scalar_i>
__global__ void softmax_kernel_online_softmax(scalar_t* __restrict__ a, scalar_t* __restrict__ b, scalar_i totalRow, scalar_i totalCol) {
  // 该block负责的起始行
  scalar_i initRow { blockIdx.y * blockDim.y };
  // 该线程负责的行
  scalar_i row { threadIdx.y };
  // 该线程的线程号
  scalar_i theadIDX { threadIdx.x };
  // 该block中的所有线程数
  scalar_i threadNum { blockDim.x };

  // 静态SMEM元素个数
  static_assert(BLOCK_DIM_X % 32 == 0, "线程需要覆盖完整的warp，实现warp树形规约");
  constexpr uint reducNum { BLOCK_DIM_X / 32 };

  static_assert(BLOCK_DIM_X / 32 == 32, "需要1024个线程实时warp0的树形规约");


  if (initRow + row < totalRow) {
    // 分配静态SMEM
    __shared__ scalar_t reduction_max[reducNum];
    __shared__ scalar_t reduction_sum[reducNum];

    // {-INFINITY}：-INFINITY 是 float 类型宏，若 scalar_t 为 int 等整数类型，
    // 花括号初始化禁止窄化转换（float→int），会报编译错误。
    // 此处 scalar_t 经 AT_DISPATCH_FLOATING_TYPES 保证只为 float/double，
    // float→double 属拓宽转换不算窄化，故实际安全；但写法上不够通用。
    // 更类型安全的写法：-std::numeric_limits<scalar_t>::infinity()
    scalar_t maxVar {-std::numeric_limits<scalar_t>::infinity()};
    scalar_t maxVar_switch {-std::numeric_limits<scalar_t>::infinity()};
    scalar_t maxVar_compare {-std::numeric_limits<scalar_t>::infinity()};
    // {0.0f}：float 字面量初始化 scalar_t。若 scalar_t 为 int，花括号初始化
    // 同样禁止 float→int 的窄化转换而报错。
    // 应使用 {} 触发值初始化（value initialization）：对所有标量类型零初始化，
    // int{} → 0，float{} → 0.0f，double{} → 0.0，类型无关且永远合法。
    scalar_t divisor {};
    scalar_t divisor_switch {};

#pragma unroll URF
    for (scalar_i i {theadIDX}; i < totalCol / 4; i+=threadNum) {
      scalar_t4 vecA { reinterpret_cast<scalar_t4*>(&a[(initRow + row) * totalCol + i * 4])[0] };
      maxVar_switch = fmaxf(maxVar_switch, vecA.x);
      maxVar_switch = fmaxf(maxVar_switch, vecA.y);
      maxVar_switch = fmaxf(maxVar_switch, vecA.z);
      maxVar_switch = fmaxf(maxVar_switch, vecA.w);

      divisor_switch += __expf(vecA.x - maxVar_switch);
      divisor_switch += __expf(vecA.y - maxVar_switch);
      divisor_switch += __expf(vecA.z - maxVar_switch);
      divisor_switch += __expf(vecA.w - maxVar_switch);

      // 冗余 if 已删除：原代码在首次迭代时用 if (i==theadIDX) 初始化 maxVar。
      // 合并公式本身已处理 maxVar=-INF 的情况（情况3）：
      //   divisor *= exp(-INF - m) = 0，divisor += divisor_switch 得正确结果，
      //   maxVar = maxVar_compare = m，无需额外分支。
      // if (i == theadIDX) {
      //   maxVar = maxVar_switch;
      // }
      // 每轮迭代合并两批数据：
      //   批次A（历史）：maxVar    = 前几轮积累的全局最大值，divisor    = 对应的 exp 之和
      //   批次B（当前）：maxVar_switch = 本轮 4 个元素的局部最大值，divisor_switch = 对应的 exp 之和
      //
      // 合并公式需要三个值：m_A、m_B、new_max = max(m_A, m_B)
      //   divisor_switch *= exp(m_B    - new_max)  ← 当前批次缩放到 new_max 基准
      //   divisor        *= exp(m_A    - new_max)  ← 历史批次缩放到 new_max 基准
      //
      // 必须引入 maxVar_compare 单独保存 new_max：
      //   若直接用 maxVar_switch = fmaxf(maxVar_switch, maxVar)，
      //   maxVar_switch 被覆盖为 new_max，m_B 丢失，
      //   divisor_switch 的缩放因子 exp(maxVar_switch - maxVar_compare)
      //   = exp(new_max - new_max) = 1，而非正确的 exp(m_B - new_max)。
      maxVar_compare = fmaxf(maxVar_switch, maxVar);
      divisor_switch *= __expf(maxVar_switch - maxVar_compare);  // exp(m_B - new_max)
      divisor        *= __expf(maxVar        - maxVar_compare);  // exp(m_A - new_max)
      divisor += divisor_switch;
      maxVar = maxVar_compare;
      // divisor_switch = static_cast<scalar_t>( 0.0f);
      divisor_switch = scalar_t{};
      maxVar_switch = -std::numeric_limits<scalar_t>::infinity();
    }

    // ── warp 内在线规约（合并 (maxVar, divisor) 对）────────────────────────────
    // maxVar_compare = max(自身, 对方) = 本步规约后的新最大值基准
    //
    // 【为什么必须用 __shfl_xor_sync，不能用 __shfl_down_sync】
    //
    // xor_sync（蝶形）：低 lane 与高 lane 互换 maxVar，双方都能计算
    //   new_max = fmaxf(m_低, m_高)，各自将 divisor 缩放到 new_max 基准，
    //   再互读对方【已缩放好】的 divisor 相加，结果正确。
    //
    // down_sync（单向）：低 lane 读高 lane，高 lane 读越界位置。
    //   高 lane 执行：
    //     maxVar_compare = fmaxf(maxVar, __shfl_down_sync(..., maxVar, i, 32))
    //   shfl_down 源 lane = 高lane + i，超出 warp 范围（≥32），
    //   越界时 __shfl_down_sync 返回高 lane 自身的 maxVar，即：
    //     maxVar_compare = fmaxf(m_高, m_高) = m_高   ← 看不到低 lane 的 max
    //   于是高 lane 执行：
    //     divisor *= exp(m_高 - m_高) = 1             ← divisor 完全未缩放
    //   低 lane 随后读到高 lane 未缩放的 divisor，相加结果错误。
    //
    // 具体失败场景（低 lane max > 高 lane max，高 lane divisor 应缩小）：
    //   低 lane0: (max=3.0, div=d₀)   高 lane2: (max=2.0, div=d₂)   new_max=3.0
    //
    //   xor_sync（正确）：
    //     高 lane2: new_max=max(2,3)=3，divisor×=exp(2-3)=exp(-1) → d₂·exp(-1)（已缩放）
    //     低 lane0: 读已缩放的 d₂·exp(-1)，结果 = d₀·exp(0) + d₂·exp(-1) ✓
    //
    //   down_sync（错误）：
    //     高 lane2: new_max=fmaxf(2, 越界→2)=2，divisor×=exp(0)=1 → d₂（未缩放）
    //     低 lane0: 读未缩放的 d₂，结果 = d₀·exp(0) + d₂ ✗（d₂ 少乘了 exp(-1)）
    //
    // 简单 max/sum 规约：高 lane 不需要缩放，down_sync 可用
    // Online softmax 规约：高 lane 必须将 divisor 缩放到 new_max 基准 → 只能用 xor_sync
    //
    // 情况1：两端均有数据（均 ≠ -INFINITY）
    //   maxVar_compare 为两端中较大的实数
    //   exp(maxVar - maxVar_compare) ≤ 1（负指数），正常缩放，无 NaN 风险
    //
    // 情况2：两端均无数据（均为 -INFINITY）
    //   maxVar_compare = fmaxf(-INF, -INF) = -INF
    //   若不处理：exp(-INF - (-INF)) = exp(NaN) = NaN，0 × NaN = NaN ← 必须跳过
    //   处理方式：检测到 maxVar_compare == -INF 时将两端置 0
    //             exp(0 - 0) = 1，divisor = 0 × 1 = 0，结果正确
    //
    // 情况3：一端无数据（-INFINITY），另一端有数据（实数）
    //   maxVar_compare = fmaxf(-INF, real) = real（≠ -INF，不触发 if）
    //   无数据端：exp(-INF - real) = exp(-∞) = 0，0 × 0 = 0，正确
    //   有数据端：exp(real - real) = 1，divisor 不变，正确
    //   无需特殊处理，exp 不产生 NaN
    for (scalar_i i {16}; i>=1; i/=2) {
      maxVar_compare = fmaxf(maxVar,__shfl_xor_sync(0xffffffff,maxVar,i,32));
      if (maxVar_compare == -std::numeric_limits<scalar_t>::infinity()) {   // 情况2：两端均无数据
        // maxVar = maxVar_compare = static_cast<scalar_t>( 0.0f);
        maxVar = maxVar_compare = scalar_t{};
      }
      divisor *= __expf(maxVar - maxVar_compare);
      divisor += __shfl_xor_sync(0xffffffff,divisor,i,32);
      maxVar = maxVar_compare;
    }

    if (theadIDX % 32 == 0) {
      reduction_max[theadIDX / 32] = maxVar;
      reduction_sum[theadIDX / 32] = divisor;
    }

    __syncthreads();  // 跨 warp 屏障：等待所有 warp 的 lane0 写完 smem 后 warp0 才能读


    if (theadIDX < 32) {
      maxVar = reduction_max[theadIDX];
      divisor = reduction_sum[theadIDX];
    }
    // ── warp0 读 smem + 跨 warp 规约（合并为一个 if 块）────────────────────────
    // 读 smem 和 Level2 规约的主体均为 theadIDX < 32（即 warp0），
    // warp 内天然 lockstep 同步，两者之间无需 __syncthreads()，合并为同一 if 块。
    // 三种情况同 Level1，注释见上。
    // __syncthreads();


    if (theadIDX < 32) {
      for (scalar_i i {16}; i>=1; i/=2) {
        maxVar_compare = fmaxf(maxVar,__shfl_xor_sync(0xffffffff,maxVar,i,32));
        if (maxVar_compare == -std::numeric_limits<scalar_t>::infinity()) {   // 情况2：两端均无数据
          // maxVar = maxVar_compare = static_cast<scalar_t>( 0.0f);
          maxVar = maxVar_compare = scalar_t{};
        }
        divisor *= __expf(maxVar - maxVar_compare);
        divisor += __shfl_xor_sync(0xffffffff,divisor,i,32);
        maxVar = maxVar_compare;
      }
    }

    if (theadIDX == 0) {
      reduction_max[theadIDX] = maxVar;
      reduction_sum[theadIDX] = divisor;
    }

    __syncthreads();

    maxVar = reduction_max[0];
    divisor = reduction_sum[0];


    // 除法优化：预先计算 inv_divisor，将循环内 4 次除法改为 1 次除法 + 4 次乘法。
    // FP 除法约 20 cycle，FP 乘法约 4 cycle；节省 3 次除法，迭代次数多时收益显著。
    scalar_t inv_divisor { scalar_t{1} / divisor };
#pragma unroll URF
    for (scalar_i i {theadIDX}; i<totalCol / 4; i+=threadNum) {
      scalar_t4 vecA { reinterpret_cast<scalar_t4*>(&a[(initRow + row) * totalCol + i * 4])[0] };
      vecA.x = __expf(vecA.x-maxVar) * inv_divisor;
      vecA.y = __expf(vecA.y-maxVar) * inv_divisor;
      vecA.z = __expf(vecA.z-maxVar) * inv_divisor;
      vecA.w = __expf(vecA.w-maxVar) * inv_divisor;
      reinterpret_cast<scalar_t4*>(&b[(initRow + row) * totalCol + i * 4])[0] = vecA;
    }
  }
}