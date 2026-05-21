#pragma once

// ── URF 重复定义问题 ────────────────────────────────────────────────────────
// kernels.cuh 将所有 kernel 头文件包含进同一翻译单元（TU）。
// kernel7 和 kernel8 各自在文件作用域定义 constexpr int URF {8};
// 预处理展开后同一 TU 内出现两次相同名字的定义 → 编译器报重复定义错误。
//
// 【为什么 inline constexpr int URF {8};
//   inline 允许同一定义出现在多个 TU 中（链接器合并），
//   但不允许同一 TU 同一作用域内出现两次定义，问题依然存在。
//
// 【三种解决方案】
//   方案A（推荐）：函数内定义
//     将 constexpr int URF {8};
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


// 所有 kernel 共用的编译期常量，统一在此定义。
// 各 kernel 头文件 #include "common.cuh"，不再各自重复定义。
//
// 优先级（高 → 低）：
//   cmake -DBLOCK_DIM_X=512  →  编译器 -D 标志（最高）
//   #ifndef 保护的默认值      →  下方 #define（最低）
//
// cmake target_compile_definitions 注入 -DBLOCK_DIM_X=X 时，
// 预处理器在文件顶部隐式插入 #define BLOCK_DIM_X X，
// 后续 #ifndef BLOCK_DIM_X 条件为假，跳过默认值，使用 cmake 注入值。

// 每个 block 在 X 维的线程数（即每行处理的线程数）
// kernel 2~10 均使用此值；kernel 1 使用 2D block，不依赖此宏
#ifndef BLOCK_DIM_X
#define BLOCK_DIM_X 1024
#endif

// 主循环展开因子，控制 #pragma unroll URF 的展开次数
// 影响 kernel 6~10；值必须满足 UNROLL_FACTOR * BLOCK_DIM_X * 4 <= totalCol
#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 4
#endif

// URF：编译期常量，供 #pragma unroll 使用
// #pragma unroll 要求操作数为编译期常量；constexpr int 满足此要求
// 集中定义一次，消除 kernel 6/7/8/9/10 各自定义时的重复定义冲突
constexpr int URF {8};


#ifndef WIDTH
#define WIDTH 4096
#endif
