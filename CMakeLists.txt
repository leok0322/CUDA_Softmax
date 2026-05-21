cmake_minimum_required(VERSION 3.20)

# 必须在 project() 之前设置，锁定编译器路径，避免多版本共存时误用错误版本
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_COMPILER  "/usr/bin/g++")

# project(<项目名> LANGUAGES <语言列表>)
#   - 声明项目名称为 Fast_Softmax，并启用 CXX（C++）和 CUDA 两种语言
#   - CMake 会在此阶段探测并配置对应的编译器（g++ / nvcc）
#   - LANGUAGES 决定了后续 add_library / add_executable 能编译的文件类型：
#       .cpp  -> g++ 编译为纯主机代码（interface.cpp 中的 pybind11 模块定义）
#       .cu   -> nvcc 编译为主机 + 设备混合代码（kernels.cu 中的 softmax kernel）
project(Fast_Softmax LANGUAGES CXX CUDA)

# 导出编译命令到 compile_commands.json，供 clangd / clang-tidy 做代码分析和补全
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find_package(CUDAToolkit REQUIRED)：现代 CMake 方式（CMake 3.17+）
#   提供 IMPORTED target：CUDA::cudart / CUDA::cublas 等
#   相比旧式 find_package(CUDA)，路径查找更可靠，变量名更规范
find_package(CUDAToolkit REQUIRED)

# find_package(<包名> [REQUIRED] [COMPONENTS <组件列表>])
#
#   Python3    : CMake 内置模块（FindPython3.cmake），专门用于查找 Python3 相关组件
#                与旧式 FindPythonInterp.cmake 相比，支持 venv / 多版本共存，更可靠
#
#   REQUIRED   : 找不到则立即报错终止，不会静默跳过
#
#   COMPONENTS Interpreter : 只查找 Python 解释器（python3 可执行文件）
#     不加 COMPONENTS 时默认查找所有组件，包括：
#       Interpreter : python3 可执行文件 → Python3_EXECUTABLE
#       Development : Python.h 头文件 + libpython.so（编译 C 扩展时需要）
#       NumPy       : numpy 头文件路径
#     本项目只需要解释器路径（用于 execute_process 查询 EXT_SUFFIX），无需编译链接 Python，
#     只指定 Interpreter 可加快 cmake 配置速度，避免搜索不需要的组件
#
#   成功后设置的变量：
#     Python3_EXECUTABLE : venv 解释器的完整路径
#                          如 /home/liam/python_linux/python_venv/.venv/bin/python3
#     Python3_VERSION    : 解释器版本号（如 3.13.5）
#
#   CMake 默认在系统 PATH 中搜索 python3，找到的是系统解释器而非 venv 解释器
#   需通过以下任一方式将搜索范围锁定到 venv：
#     -DPython3_ROOT_DIR=/home/liam/python_linux/python_venv/.venv
#                                       CMake 在 .venv/bin/ 下搜索 python3，找到 venv 解释器
#     -DPython3_EXECUTABLE=/home/liam/python_linux/python_venv/.venv/bin/python3
#                                       直接锁定到具体可执行文件，跳过任何搜索逻辑，最精确
#                                       路径来源：venv 激活后执行 which python3 或 readlink -f $(which python3)
#   两者均在 CLion CMake options 或命令行 cmake -D 中传入；同时指定时 EXECUTABLE 优先
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# Development 组件额外提供：
#   Python3_INCLUDE_DIRS : Python.h 所在目录
#   Python3_LIBRARIES    : libpython.so
# 编译时需要 Python.h 的原因：
#   torch/extension.h → torch/csrc/python_headers.h → <Python.h>
message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")

# ── GPU 架构检测（必须在 find_package(Torch) 之前，使 _GPU_CC 可用）────────────
# execute_process(<参数列表>)：在 cmake 配置阶段（而非编译阶段）运行外部命令
#   与 add_custom_command 的区别：
#     execute_process   : cmake -S ... -B ... 时立即执行，结果作为配置信息使用
#     add_custom_command: make / ninja 时执行，属于构建阶段的步骤
#
#   COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
#     nvidia-smi              : NVIDIA 系统管理接口，随 GPU 驱动安装，与 CUDA Toolkit 无关
#     --query-gpu=compute_cap : 查询 GPU 的 Compute Capability，输出格式为 "8.6"
#     --format=csv,noheader   : 以 CSV 格式输出，去掉表头行，只保留数值
#     多卡机器每张卡输出一行，此处取第一行（_GPU_CC 只读第一个值）
#
#   OUTPUT_VARIABLE _GPU_CC          : 将命令的 stdout 存入变量 _GPU_CC（值为 "8.6"）
#   OUTPUT_STRIP_TRAILING_WHITESPACE : 去掉末尾换行符，否则 _GPU_CC 为 "8.6\n"
#   ERROR_QUIET                      : nvidia-smi 不存在时静默失败，_GPU_CC 保持为空
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _GPU_CC
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# string(REPLACE <匹配串> <替换串> <结果变量> <输入串>)
#   将 _GPU_CC（"8.6"）中的 "." 替换为 ""，得到 "86" 存入 _GPU_CC_NUM
#   nvcc 的 -gencode 参数需要无小数点的整数形式（如 86），不接受 "8.6"
string(REPLACE "." "" _GPU_CC_NUM "${_GPU_CC}")

if(NOT _GPU_CC_NUM)
    message(FATAL_ERROR "无法检测 GPU 架构，请确认 nvidia-smi 可用")
endif()

# ── find_package(Torch) 前的准备：TORCH_CUDA_ARCH_LIST + unset ───────────────
#
# 顺序约束（三者缺一不可）：
#   1. execute_process(nvidia-smi) 先于此处运行，_GPU_CC 已有值（如 "8.6"）
#   2. TORCH_CUDA_ARCH_LIST 在 find_package(Torch) 之前设置，供 PyTorch cmake 读取
#   3. unset(CMAKE_CUDA_ARCHITECTURES CACHE) 在 find_package(Torch) 之前清除，
#      避免 PyTorch cuda.cmake 第 322 行 if(DEFINED CMAKE_CUDA_ARCHITECTURES) 触发 Warning 2
#      根因：project(... LANGUAGES CUDA) 初始化 CUDA 语言时，CMake 3.20（CMP0104）
#            会自动将 CMAKE_CUDA_ARCHITECTURES 赋为默认值（"native" 或空串）；
#            不 unset 则 find_package(Torch) 执行时变量已定义，Warning 2 触发
#   4. set(CMAKE_CUDA_ARCHITECTURES) 在 find_package(Torch) 之后、check_language(CUDA)
#      之前设置：check_language 编译测试程序时需要 CMAKE_CUDA_ARCHITECTURES 有值，
#      若为空则报 "CUDA_ARCHITECTURES is empty for target" 错误
#
#   TORCH_CUDA_ARCH_LIST 格式：带点版本号 "8.6"（PyTorch 专用）
#   CMAKE_CUDA_ARCHITECTURES 格式：无点整数 "86"（CMake / nvcc 专用）
set(TORCH_CUDA_ARCH_LIST "${_GPU_CC}")
unset(CMAKE_CUDA_ARCHITECTURES CACHE)
# find_package(Torch REQUIRED)：查找 LibTorch（PyTorch 的 C++ 后端）
#   设置以下变量：
#     TORCH_INCLUDE_DIRS   : torch/extension.h 等头文件路径
#     TORCH_LIBRARIES      : libtorch.so / libtorch_cpu.so 等
#     TORCH_CXX_FLAGS      : PyTorch 要求的 ABI 标志（如 -D_GLIBCXX_USE_CXX11_ABI=1）
#     TORCH_INSTALL_PREFIX : LibTorch 安装根目录
#   需通过 CMAKE_PREFIX_PATH 指定 torch 的 cmake 目录，在 CLion CMake options 中添加：
#     -DCMAKE_PREFIX_PATH=/home/liam/python_linux/python_venv/.venv/lib/python3.13/site-packages/torch/share/cmake
find_package(Torch REQUIRED)

# find_package(Torch) 执行时来自 PyTorch 自身 cmake 文件的三个 Warning：
#
#   Warning 1：Failed to compute shorthash for libnvrtc.so
#     来源：cuda.cmake，PyTorch 对 NVRTC 库做完整性校验时找不到哈希值
#     影响：无，NVRTC 是运行时编译库，本项目不使用，可忽略
#
#   Warning 2：pytorch is not compatible with CMAKE_CUDA_ARCHITECTURES
#     见上方顺序约束说明，已通过 unset + 延迟 set 修复
#
#   Warning 3：static library kineto_LIBRARY-NOTFOUND not found
#     来源：TorchConfig.cmake，Kineto 是 PyTorch 的性能分析库
#     原因：此 PyTorch 安装包未编译 Kineto，find_library 找不到
#     影响：无，Kineto 仅用于 PyTorch 自身的性能分析，不影响 CUDA kernel 编译和运行，可忽略

# set(CMAKE_CUDA_ARCHITECTURES)：find_package(Torch) 之后、check_language(CUDA) 之前设置

# set(CMAKE_CUDA_ARCHITECTURES <架构号>)：全局设置目标 GPU 的计算能力版本
#   触发 Warning 并强制将其覆盖为 OFF（见上方 TORCH_CUDA_ARCH_LIST 注释）
#
#   作用范围：对之后所有 add_library / add_executable 创建的 target 全局生效，
#             各 target 自动继承，无需逐个调用 set_target_properties(... CUDA_ARCHITECTURES ...)
#   CMake 内部将其展开为 nvcc 的 -gencode 标志：
#     -gencode arch=compute_86,code=[sm_86,compute_86]
#       code=sm_86      : 生成 SASS 二进制，在当前 GPU 上直接执行，性能最优
#       code=compute_86 : 保留 PTX 中间码，供未来更新架构的 GPU JIT 编译（向前兼容）
#   常见架构号对照：
#     75=Turing(RTX20)  80=Ampere(A100)  86=Ampere(RTX30)  89=Ada(RTX40)  90=Hopper(H100)
#   不设置全局值时的替代方式（效果相同，但需逐个 target 指定，繁琐）：
#     set_target_properties(softmax_cuda PROPERTIES CUDA_ARCHITECTURES 86)
#     cmake -DCMAKE_CUDA_ARCHITECTURES=86 ...
set(CMAKE_CUDA_ARCHITECTURES ${_GPU_CC_NUM})
message(STATUS "GPU Compute Capability: ${_GPU_CC} -> CMAKE_CUDA_ARCHITECTURES=${_GPU_CC_NUM}  TORCH_CUDA_ARCH_LIST=${_GPU_CC}")

# find_library(<变量名> <库名> [PATHS <路径>] [REQUIRED])
#   与 find_package 的区别：
#     find_package : 高层接口，依赖包自带的 XxxConfig.cmake，一次调用同时设置
#                    头文件路径、库路径、编译标志、IMPORTED target 等所有信息
#     find_library : 低层接口，只在指定目录下搜索单个 .so / .a 文件，
#                    把完整路径存入变量，不设置任何头文件路径或编译标志
#
#   torch_python 用 find_library 而非 find_package 的原因：
#     libtorch_python.so 是 PyTorch 的内部子库，没有自己的 TorchPythonConfig.cmake，
#     find_package(Torch) 也故意未将其包含在 TORCH_LIBRARIES 中（让用户按需链接）；
#     它没有 CMake 包的身份，只是一个裸 .so，因此只能用 find_library 直接定位文件
#
#   参数说明：
#     TORCH_PYTHON_LIBRARY        : 结果变量，成功后存储 libtorch_python.so 的完整路径
#     torch_python                : 库名（不含 lib 前缀和 .so 后缀），
#                                   find_library 自动拼接为 libtorch_python.so 去搜索
#     PATHS "${TORCH_INSTALL_PREFIX}/lib" : 搜索目录，由 find_package(Torch) 设置的
#                                           TORCH_INSTALL_PREFIX 展开为 LibTorch 安装根目录
#     REQUIRED                    : 找不到则报错终止
find_library(TORCH_PYTHON_LIBRARY torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    REQUIRED
)

# include(CheckLanguage)：加载 CMake 内置模块，引入 check_language() 函数
#   CMake 内置模块位于 /usr/share/cmake-*/Modules/CheckLanguage.cmake
#   类似 C++ 中的 #include，不加载则 check_language() 未定义，调用会报错
include(CheckLanguage)

# check_language(CUDA)：探测当前系统是否具备完整可用的 CUDA 编译环境
#   实际编译一段最小 CUDA 代码来验证 nvcc 能否正常工作
#   需要 CMAKE_CUDA_ARCHITECTURES 已设置（见上方），否则报 "CUDA_ARCHITECTURES is empty" 错误
#   结果写入缓存变量 CMAKE_CUDA_COMPILER：
#     有效路径（如 /usr/local/cuda/bin/nvcc）：探测成功
#     NOTFOUND                               ：探测失败
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found")
endif()

# PyTorch 2.x 要求至少 C++17；C++20 作为超集向后兼容，可直接使用
#   注意：PyTorch 某些旧头文件的 volatile 复合赋值在 C++20 下会触发 deprecated 警告
#         （C++20 弃用了部分 volatile 用法），不影响编译和运行，可忽略
set(CMAKE_CXX_STANDARD 20)
# 编译器不支持 C++20 时直接报错终止，而非静默降级到更低标准
#   不设置（默认）：编译器不支持时悄悄降级（如降到 C++17），你以为在用 C++20 实际没有
#   设置为 ON    ：不支持则 fatal error，确保编译标志确实是 -std=c++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── 可配置的 kernel 编译参数 ──────────────────────────────────────────────────
# set(<变量名> <默认值> CACHE <类型> "<描述>")
#   与普通 set() 的区别：
#     普通 set()      : 变量只存在于当前 cmake 运行的内存中，每次 Reload 都重新赋值
#     CACHE set()     : 变量写入 CMakeCache.txt（构建目录下），跨 Reload 持久保存；
#                       若用户已通过 -D 传入值，CACHE set() 不会覆盖，保留用户的值
#   set(... CACHE ...) 与 -D 的分工：
#     set(SOFTMAX_VARIANT 8 CACHE ...) : 提供兜底默认值，cache 里没有值时才生效
#     -DSOFTMAX_VARIANT=6              : 用户指定值，由 CLion CMake options 或命令行传入
#     set() 永远不会覆盖已在 cache 中的值，两者优先级：-D > cache 已有值 > set() 默认值
#
#   三层默认值的优先级（高 → 低）：
#     1. cmake -DSOFTMAX_VARIANT=6          → 最高，写入 cache，覆盖一切
#     2. set(SOFTMAX_VARIANT 8 CACHE ...)   → cache 无值时写入，cmake 层的兜底
#     3. #ifndef SOFTMAX_VARIANT            → build.ninja 无 -D 时的编译器层兜底
#        #define SOFTMAX_VARIANT 1
#        #endif
#     只要 cmake 通过 target_compile_definitions 把 -DSOFTMAX_VARIANT 写入 build.ninja，
#     第 3 层的 #ifndef 条件就为假，修改源文件中的默认值 1 完全没有效果。
#
#   三种场景下的行为：
#     首次 Reload，未传 -D : cache 无值 → set() 写入默认值 8 → 编译用 8
#     Reload 时传 -DSOFTMAX_VARIANT=6 : CMake 先把 6 写入 cache → set() 检测到已有值不覆盖 → 编译用 6
#     再次 Reload，未传 -D : cache 仍有 6（上次写入）→ set() 不覆盖 → 编译仍用 6
#     重置为默认值         : 删除 cmake-build-release/CMakeCache.txt 后 Reload，或显式传 -DSOFTMAX_VARIANT=8
#
#   【修改 CMakeLists.txt 中的默认值不会改变 cache 中已有的值】
#     若 cache 中已有 SOFTMAX_VARIANT=8，把文件里的默认值从 8 改为 6，
#     重新 cmake（Reload）后 cache 中仍是 8，因为 set(... CACHE ...) 检测到已有值就跳过。
#     必须显式传 -D 或删除 CMakeCache.txt 才能让修改生效。
#
#   类型 STRING：在 CLion / cmake-gui 中以文本框展示，可直接编辑
#     其他常用类型：BOOL（复选框）、PATH（目录选择器）、FILEPATH（文件选择器）
#     类型仅影响 GUI 展示方式，不影响 CMake 内部的变量处理
#
#   描述字符串：显示在 cmake-gui / ccmake 的变量旁边，说明该变量的用途和合法值范围
#
#   这四个宏对应 kernels.cu 顶部的 #ifndef 默认值，通过 cmake -D 可在配置阶段覆盖，
#   等价于 benchmark.py 中 torch.utils.cpp_extension.load 的 extra_cuda_cflags / extra_cflags：
#     SOFTMAX_VARIANT : 选择 kernel 实现（1~9），对应 softmax_kernel1~9
#     BLOCK_DIM_X     : 每个 block 在 Y 维度的线程数，必须是 2 的幂次（128~1024）
#     UNROLL_FACTOR   : 循环展开因子（1/2/4/8），用于 kernel7/8/9 的向量化循环
#     WIDTH           : kernel10 专用的编译期行宽，必须与实际矩阵的 totalCol 完全一致。
#                       kernel10 用 WIDTH 宏作为循环上界和 reg_array 大小，不读运行时参数 w，
#                       两者不一致时循环越界或提前结束，结果静默错误。
#                       totalCol 是运行时变量，无法直接用作编译期常量，
#                       因此 kernel10 是为特定列数硬编译的 kernel：
#                         cmake -DWIDTH=4096 → 只在 totalCol=4096 时结果正确
#                         cmake -DWIDTH=0（默认）→ #ifndef WIDTH 跳过 4096，WIDTH=0，
#                           循环条件 i<0 不执行，输出缓冲区为未初始化的垃圾值
#
#   用法示例（选择 variant=6，block_y=512）：
#     cmake -DSOFTMAX_VARIANT=6 -DBLOCK_DIM_X=512 ..
set(SOFTMAX_VARIANT  8    CACHE STRING "Softmax kernel variant (1-9), matches softmax_kernelN")
set(BLOCK_DIM_X      1024 CACHE STRING "Block dim Y, power of 2 in [128, 1024]")
set(UNROLL_FACTOR    8    CACHE STRING "Loop unroll factor for kernel7+: 1/2/4/8")
# WIDTH 是 kernel10 专用编译期行宽，必须与实际totalCol 一致，而totalCol是编译期变量，会随时改变。
set(WIDTH            4096    CACHE STRING "kernel10 static row width; must equal actual totalCol; 0 = loop does not execute, output is uninitialized")

message(STATUS "Kernel config: SOFTMAX_VARIANT=${SOFTMAX_VARIANT}  BLOCK_DIM_X=${BLOCK_DIM_X}  UNROLL_FACTOR=${UNROLL_FACTOR}  WIDTH=${WIDTH}")

# ── 全局编译选项 ──────────────────────────────────────────────────────────────
# add_compile_options 中的生成器表达式在 cmake 配置阶段求值，按条件决定是否将选项写入 build.ninja
# 各选项对应的 nvcc 实际命令行（以架构 86 为例）：
#
#   Debug  模式：nvcc -G -lineinfo --ptxas-options=-v --use_fast_math 不加
#   Release模式：nvcc    -lineinfo --ptxas-options=-v --use_fast_math
#
# Debug 模式：-G 为 GPU 设备代码生成调试符号，供 cuda-gdb 单步调试 kernel
#   副作用：禁用几乎所有 GPU 优化，性能大幅下降，不可用于性能测量
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")

# Release 模式：-lineinfo 保留源码行号，让 ncu 能将性能数据映射回源码行
#   与 -G 的区别：不禁止优化，Release 下 ncu 分析才能反映真实性能
add_compile_options("$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")

# --use_fast_math：Release 下将标准数学函数替换为 GPU 硬件内置近似实现，提升性能
#   实际展开为三个子选项：
#     -ftz=true       : 把非规格化浮点数（denormal）刷成零，避免硬件慢速处理路径
#     -prec-div=false : 用快速近似除法替代精确除法
#     -prec-sqrt=false: 用快速近似平方根替代精确平方根
#   同时把标准函数自动替换为硬件内置近似版本：
#     exp()  → __expf()   sin()  → __sinf()   sqrt() → __fsqrt_rn()  ...
#
#   本项目 kernels.cu 已直接调用 __expf()，对这部分无额外影响；
#   主要影响是 -ftz / -prec-div / -prec-sqrt 以及编译器自动替换其他标准函数调用
#
#   加此选项的原因：与 benchmark.py 的 JIT 编译参数保持一致
#     benchmark.py: extra_cuda_cflags=[..., "--use_fast_math", ...]
#     两种构建路径（JIT / CMake）编译出的 kernel 行为相同，benchmark 结果才可互相对比；
#     若一个加了一个没加，softmax 数值可能有细微差异，torch.allclose 断言可能因此失败
add_compile_options("$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--use_fast_math>")

# --ptxas-options=-v：两种模式均开启，输出每个 kernel 的寄存器 / 共享内存用量
#   Debug 下看未优化代码的资源占用，Release 下看真实资源占用（影响 occupancy）
#
#   不加 $<CONFIG:xxx> 条件则两种模式都触发，加了则只在对应模式触发：
#     两种模式均开启：$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
#     只开启 Debug  ：$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:--ptxas-options=-v>
#     只开启 Release：$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--ptxas-options=-v>
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>")

# TORCH_CXX_FLAGS 由 find_package(Torch) 设置，通常包含 ABI 标志，如：
#   "-D_GLIBCXX_USE_CXX11_ABI=1"
#   必须应用到所有编译单元，否则与 libtorch.so 的 ABI 不匹配，链接时报 undefined reference
#   原因：PyTorch 编译时选定了一个 ABI 版本，扩展必须用同一版本，否则符号名不同无法链接
#
# separate_arguments(<结果变量> <模式> "<输入字符串>")
#   将空格分隔的字符串拆分为 CMake list（分号分隔），存入结果变量
#   必须拆分的原因：
#     TORCH_CXX_FLAGS 是一个普通字符串，如 "-D_GLIBCXX_USE_CXX11_ABI=1 -Wall"
#     直接传给 add_compile_options("${TORCH_CXX_FLAGS}") 会被当作一个整体选项，
#     编译器收到的是 "-D_GLIBCXX_USE_CXX11_ABI=1 -Wall"（带空格的单个参数），报错
#     拆分后变为 CMake list："-D_GLIBCXX_USE_CXX11_ABI=1;-Wall"
#     add_compile_options 展开 list 后，编译器收到两个独立参数，正确解析
#
#   NATIVE_COMMAND：按当前平台的 shell 规则拆分
#     Linux/macOS : 以空格为分隔符（等价于 UNIX_COMMAND）
#     Windows     : 以空格为分隔符并处理引号（等价于 WINDOWS_COMMAND）
#     使用 NATIVE_COMMAND 无需关心平台差异，CMake 自动选择正确规则
# 本机实际值（PyTorch 以 CXX11 ABI 编译，torch.compiled_with_cxx11_abi() == True）：
#   TORCH_CXX_FLAGS     = "-D_GLIBCXX_USE_CXX11_ABI=1"        ← 空格分隔的普通字符串
#   TORCH_CXX_FLAGS_LIST = "-D_GLIBCXX_USE_CXX11_ABI=1"       ← 拆分后的 CMake list（此处只有一项）
#
#   若 PyTorch 以旧 ABI 编译（compiled_with_cxx11_abi() == False）：
#   TORCH_CXX_FLAGS     = "-D_GLIBCXX_USE_CXX11_ABI=0 -Wall"  ← 假设含多个标志
#   TORCH_CXX_FLAGS_LIST = "-D_GLIBCXX_USE_CXX11_ABI=0;-Wall" ← 拆分为两项的 list
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

# ${TORCH_CXX_FLAGS_LIST} 展开为多个独立参数，每个编译单元（.cu / .cpp）均追加这些标志
# 本机等价于对所有编译命令追加：-D_GLIBCXX_USE_CXX11_ABI=1
#   nvcc kernels.cu    ... -D_GLIBCXX_USE_CXX11_ABI=1
#   g++  interface.cpp ... -D_GLIBCXX_USE_CXX11_ABI=1
add_compile_options(${TORCH_CXX_FLAGS_LIST})

# ── .o / .so / 可执行文件 的关系 ─────────────────────────────────────────────
# 三者均为 ELF（Executable and Linkable Format）二进制，ELF 头中的 type 字段区分身份：
#
#   .o（目标文件，ET_REL）：单个源文件编译后的中间产物
#     - 内含机器码片段，但外部符号（如 cudaMalloc / torch::xxx）地址未填，只有占位符
#     - 不能独立运行，存在意义是增量编译：只有改动的源文件重新生成 .o，
#       没改的复用旧 .o，最后统一链接，避免每次全量重编
#
#   .so（共享对象，ET_DYN）：多个 .o 链接后的完整产物
#     - 所有符号地址已解析填入，是完整可加载的二进制
#     - 没有 main() 入口，不能独立运行，须由可执行文件在运行时通过 dlopen() 加载
#     - PTX 作为数据段嵌入 .o 后随链接进入 .so，驱动运行时从 .so 中读取并 JIT 编译
#
#   可执行文件（ET_EXEC / ET_DYN）：有 main() 入口，可独立运行
#     - 本项目中 python3 是可执行文件，softmax_cuda.so 是它加载的库，两者平行关系：
#         python3 → import softmax_cuda → dlopen(softmax_cuda.so) → 函数可调用
#     - .so 不会"变成"可执行文件，也不存在从 .so 到可执行文件的转换路径
#
#   完整构建流程对比：
#
#   本项目 add_library(softmax_cuda SHARED ...)：
#     interface.cpp  ──(g++)──▶  interface.cpp.o  ─┐
#                                                   ├─(ld)──▶  softmax_cuda.so  ← 无入口，须由 python3 dlopen 加载
#     kernels.cu     ──(nvcc)─▶  kernels.cu.o     ─┘
#       含 SASS（sm_86）和 PTX（compute_86）两段，均嵌入 .o，链接后进入 .so
#
#   参考 GEMM 项目 add_executable(gemm ...)：
#     gemm.cu        ──(nvcc)──▶  gemm.cu.o       ─┐
#     sgemm_v1.cu    ──(nvcc)──▶  sgemm_v1.cu.o   ─┤
#     sgemm_v2.cu    ──(nvcc)──▶  sgemm_v2.cu.o   ─┼─(ld)──▶  gemm  ← 有 main() 入口，./gemm 直接运行
#     utils.cpp      ──(g++)───▶  utils.cpp.o     ─┘
#
#   两者编译阶段完全相同，区别仅在链接阶段：
#     add_library SHARED → ld 生成无入口的共享库（ET_DYN，无 main）
#     add_executable     → ld 生成有入口的可执行文件（ET_EXEC / ET_DYN，有 main）
#
# aux_source_directory(<目录> <变量>)：扫描指定目录下所有已知扩展名的源文件（.c/.cpp/.cu 等），
#   将完整路径列表存入变量，非递归（不扫子目录）
#   等价于手动写：set(KERNELS src/kernels/softmax_kernel.cu ...)
#   缺点：新增文件后需手动 Reload CMake 才能被感知（配置阶段执行一次，之后不追踪变化）
# aux_source_directory 只收集有效源文件扩展名（.c/.cpp/.cu 等），.cuh 不在其中
# src/kernels/ 目前只有 softmax_kernel.cuh，因此 ${KERNELS} 实际为空列表
# 保留此变量为将来在 src/kernels/ 下添加 .cu 实现文件预留
aux_source_directory(${PROJECT_SOURCE_DIR}/src/kernels KERNELS)

# ── softmax_cuda 共享库（Python 扩展模块）────────────────────────────────────
# 构建为 SHARED 库而非可执行文件，Python import 时动态加载 .so
#
# 源文件：能独立编译成 .o 目标文件的文件（.cpp / .cu）
#   头文件（.h / .cuh）不是源文件，不单独编译，不产生 .o，
#   通过 #include 展开进源文件，由编译器自动处理，不需要列在此处
#
# CMake 根据每个源文件的扩展名决定编译器，各文件独立编译为 .o，最后链接为 .so：
#   .cpp → g++   src/interface.cpp  → interface.cpp.o
#   .cu  → nvcc  src/run_kernels.cu → run_kernels.cu.o
#                     ↓ ld（链接）
#   softmax_cuda.cpython-313-x86_64-linux-gnu.so
#     ├── interface.cpp.o
#     └── run_kernels.cu.o
#
# 增量编译：只有内容变化的源文件重新走编译流程，其余复用已有 .o
#   修改 run_kernels.cu → 只重编 run_kernels.cu.o，interface.cpp.o 不变
add_library(softmax_cuda SHARED
    src/interface.cpp
    src/run_kernels.cu
    # src/kernels.cuh      ← 头文件，不产生 .o，不需要列在此处
    # ${KERNELS}           ← src/kernels/ 下无 .cu 源文件，当前为空列表
)

# Python C 扩展的文件命名规范：
#   Python 解释器 import 时按固定规则搜索文件名，格式为：
#     <模块名>.<python版本>-<平台>.so
#   本机实际值（Python 3.13，x86_64 Linux）：
#     softmax_cuda.cpython-313-x86_64-linux-gnu.so
#   若文件名不符合此规范，import softmax_cuda 找不到文件，报 ModuleNotFoundError
#
# execute_process 调用 venv 解释器查询本机的扩展后缀：
#   sysconfig.get_config_var('EXT_SUFFIX') 返回当前解释器对应的标准后缀
#   本机返回：.cpython-313-x86_64-linux-gnu.so
#   结果存入 _PY_EXT_SUFFIX，OUTPUT_STRIP_TRAILING_WHITESPACE 去掉末尾换行
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE _PY_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# set_target_properties(<target> PROPERTIES <属性名> <值>)：设置 target 的文件名属性
#   执行前的默认文件名（CMake 对 SHARED 库的默认规则）：
#     PREFIX = "lib"，SUFFIX = ".so"  →  libsoftmax_cuda.so
#   执行后：
#     PREFIX = ""，SUFFIX = ".cpython-313-x86_64-linux-gnu.so"  →  softmax_cuda.cpython-313-x86_64-linux-gnu.so
#
#   PREFIX ""  : 去掉 CMake 对共享库默认添加的 "lib" 前缀
#                不设置则产物为 libsoftmax_cuda.so，Python 无法识别
#   SUFFIX ... : 将后缀替换为 Python 标准扩展名，覆盖默认的裸 ".so"
#                两者合并后产物名为：softmax_cuda.cpython-313-x86_64-linux-gnu.so
#
#   查询失败（ERROR_QUIET 静默，_PY_EXT_SUFFIX 为空）时只去前缀，保留裸 .so，
#   产物为 softmax_cuda.so，Python 无法直接 import，但仍可用 ctypes.CDLL() 加载
if(_PY_EXT_SUFFIX)
    set_target_properties(softmax_cuda PROPERTIES PREFIX "" SUFFIX "${_PY_EXT_SUFFIX}")
    message(STATUS "Python extension suffix: ${_PY_EXT_SUFFIX}")
else()
    set_target_properties(softmax_cuda PROPERTIES PREFIX "")
    message(WARNING "无法检测 Python 扩展后缀，生成裸 .so；若需直接 import，请手动重命名")
endif()

# target_compile_definitions(<target> <范围> <宏定义列表>)
#   桥梁作用：读取 CMake 变量（配置阶段的 -D），将其写成编译器标志 -D 进入 build.ninja，
#             编译阶段 nvcc/g++ 读取后，-D 才真正成为 C 预处理器宏
#   将宏定义注入指定 target 的编译命令，等价于给编译器追加 -D<宏名>=<值>
#
#   执行阶段说明：
#     set(SOFTMAX_VARIANT 8 CACHE ...)      ← 配置阶段（Reload）：把值存入 CMakeCache.txt
#     target_compile_definitions(... ${SOFTMAX_VARIANT}) ← 配置阶段（Reload）：读取 CMake 变量，
#                                                           把 -DSOFTMAX_VARIANT=8 写入 build.ninja
#     nvcc 编译 kernels.cu                  ← 编译阶段（Build）预处理步骤：
#                                             读取 build.ninja 中的 -DSOFTMAX_VARIANT=8，
#                                             #ifndef SOFTMAX_VARIANT 条件为假，跳过默认值，
#                                             使用传入的 8，后续编译 / 汇编 / 链接均使用此值
#   即：target_compile_definitions 本身在配置阶段执行（写 build.ninja），
#       对 kernels.cu 的实际影响发生在编译阶段的预处理器步骤，两者不在同一阶段
#
#   【cmake --build 不需要重新传宏】
#   cmake --build cmake-build-release --target softmax_cuda -- -j 18
#     编译阶段直接读取已生成的 build.ninja，-DSOFTMAX_VARIANT=8 已在其中，
#     无需也无法通过 --build 命令行再传宏；要改宏值必须重新跑配置阶段：
#       cmake -DSOFTMAX_VARIANT=6 -B cmake-build-release   ← 重新配置，更新 build.ninja
#       cmake --build cmake-build-release                  ← 编译阶段用新值
#
#   ── 宏的作用范围 ──────────────────────────────────────────────────────────────
#
#   target_compile_definitions(softmax_cuda PRIVATE ...) 只对 softmax_cuda target
#   的直接编译单元（.cpp / .cu）生效：
#
#     g++  src/interface.cpp    → 收到所有 -D，只实际用到 TORCH_EXTENSION_NAME
#     nvcc src/run_kernels.cu   → 收到所有 -D，实际用到见下表
#
#   src/kernels/ 下的 .cuh 头文件不是编译单元，本身不收到任何 -D；
#   预处理器先将 #include 展开，把 .cuh 内容文本合并进 run_kernels.cu，
#   之后编译器处理合并后的翻译单元，-D 对整个翻译单元生效，.cuh 中的宏由此可用。
#
#   【各宏在源文件中的实际使用情况】
#
#   宏名                  interface.cpp   run_kernels.cu  说明
#   ─────────────────────────────────────────────────────────────────────────────
#   TORCH_EXTENSION_NAME  ✓ 使用          ✗ 不用         PYBIND11_MODULE 展开时用
#   SOFTMAX_VARIANT       ✗ 不用          ✓ 使用         #if SOFTMAX_VARIANT==N 选择 kernel
#   BLOCK_DIM_X           ✗ 不用          ✓ 使用         kernel1~6 的 block.y 维线程数；
#                                                         kernel7/8/10 用各自 .cuh 内部的
#                                                         #ifndef BLOCK_DIM_X 定义，
#                                                         不依赖此处注入的 BLOCK_DIM_X
#   UNROLL_FACTOR         ✗ 不用          ✓ 使用         kernel7+ 的循环展开因子
#   WIDTH                 ✗ 不用          ✓ 使用         kernel10 的静态行宽提示
#
#   【validation target 不继承这些宏】
#   validation 是独立的 add_executable target，有自己的 target_compile_definitions
#   （或没有），与 softmax_cuda 的 PRIVATE 定义完全隔离，不会收到上述任何 -D。
#
#   与 add_compile_options 的区别：
#     add_compile_options        : 对所有 target 全局生效
#     target_compile_definitions : 只对指定 target（softmax_cuda）生效，不影响其他 target
#
#   PRIVATE：宏定义只作用于 softmax_cuda 自身的编译
#     PUBLIC    : 自身编译 + 传递给链接了此 target 的其他 target
#     INTERFACE : 只传递，自身不用（用于纯头文件库）
#     本项目是叶子 target（无其他 target 链接它），PRIVATE / PUBLIC 效果相同，
#     用 PRIVATE 更准确，明确表示不对外传递
#
#   各宏定义展开为编译器标志（本机默认值示例）：
#     TORCH_EXTENSION_NAME=softmax_cuda  →  -DTORCH_EXTENSION_NAME=softmax_cuda
#     BLOCK_DIM_X=${BLOCK_DIM_X}         →  -DBLOCK_DIM_X=1024
#     UNROLL_FACTOR=${UNROLL_FACTOR}     →  -DUNROLL_FACTOR=8
#     SOFTMAX_VARIANT=${SOFTMAX_VARIANT} →  -DSOFTMAX_VARIANT=8
#     WIDTH=${WIDTH}                     →  -DWIDTH=0
#
#   在 kernels.cu 中，#ifndef 保护被这些 -D 标志覆盖：
#     #ifndef BLOCK_DIM_X
#     #define BLOCK_DIM_X 1024   ← 若未传 -D，用此默认值
#     #endif
#     编译时收到 -DBLOCK_DIM_X=1024，#ifndef 条件为假，跳过此块，直接使用传入值
#
#   TORCH_EXTENSION_NAME：PYBIND11_MODULE 宏展开时使用此名称作为 Python 模块名
#     必须与库文件名（softmax_cuda）一致，否则 import 时报 ImportError：
#       import softmax_cuda  →  Python 加载 softmax_cuda.cpython-313-...so
#                            →  寻找模块入口 PyInit_softmax_cuda
#                            →  若 TORCH_EXTENSION_NAME 不同，入口名不匹配，加载失败

#   ── cmake -D 与 编译器 -D（C 预处理器宏）的区别 ──────────────────────────────
#
#   相同写法（-D），不同系统，含义完全不同：
#
#     cmake -DSOFTMAX_VARIANT=6
#       属于 CMake 变量系统；配置阶段把 CMake 变量 "6" 写入 CMakeCache.txt
#       此时源码的 #ifndef 看不到它，它只是 CMake 的一个字符串
#
#     nvcc/g++ -DSOFTMAX_VARIANT=6
#       属于 C 预处理器系统；等价于在源文件第一行插入 #define SOFTMAX_VARIANT 6
#       #ifndef SOFTMAX_VARIANT 条件为假，跳过默认值
#
#   target_compile_definitions 是两者之间的桥梁：
#     读取 CMake 变量值（"6"），写成编译器 -D 标志进入 build.ninja
#     编译阶段 nvcc/g++ 读取后，才真正成为 C 预处理器宏
#
#   完整流水线（以 cmake -DSOFTMAX_VARIANT=6 -DBLOCK_DIM_X=512 为例）：
#
#   ┌─ 配置阶段（cmake 命令） ─────────────────────────────────────────────────────
#   │  触发：cmake -DSOFTMAX_VARIANT=6 -DBLOCK_DIM_X=512 -G Ninja -S . -B build/
#   │
#   │  CMakeLists.txt 命令                    → 配置阶段执行的操作
#   │  ─────────────────────────────────────────────────────────────────────────
#   │  set(CMAKE_CUDA_COMPILER ...)          锁定 nvcc 路径，写入 CMakeCache.txt
#   │  project(... LANGUAGES CXX CUDA)       探测 g++ / nvcc，初始化两种语言
#   │  set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 标记：配置后生成 compile_commands.json
#   │  find_package(CUDAToolkit)             设置 CUDAToolkit_INCLUDE_DIRS
#   │  find_package(Python3 ...)             设置 Python3_EXECUTABLE / INCLUDE_DIRS
#   │  execute_process(nvidia-smi)           _GPU_CC="8.6", _GPU_CC_NUM="86"
#   │  find_package(Torch)                   设置 TORCH_INCLUDE_DIRS / LIBRARIES
#   │  set(CMAKE_CUDA_ARCHITECTURES 86)      nvcc 编译时生成 -gencode arch=compute_86,...
#   │  find_library(TORCH_PYTHON_LIBRARY)    找到 libtorch_python.so
#   │  set(SOFTMAX_VARIANT 8 CACHE ...)      cache 已有 "6"（-D 传入），不覆盖
#   │  set(BLOCK_DIM_X 1024 CACHE ...)       cache 已有 "512"，不覆盖
#   │  add_compile_options(-lineinfo ...)    向 build.ninja 写入全局编译标志
#   │  add_library(softmax_cuda SHARED ...)  注册 target，声明三个源文件
#   │  target_compile_definitions(...)  ←── 关键桥梁：读取 CMake 变量 → 写编译器 -D
#   │    SOFTMAX_VARIANT=${SOFTMAX_VARIANT}  build.ninja: -DSOFTMAX_VARIANT=6
#   │    BLOCK_DIM_X=${BLOCK_DIM_X}          build.ninja: -DBLOCK_DIM_X=512
#   │    TORCH_EXTENSION_NAME=softmax_cuda   build.ninja: -DTORCH_EXTENSION_NAME=softmax_cuda
#   │  target_include_directories(...)       build.ninja: -I/cuda/include -I/torch/include ...
#   │  target_link_libraries(...)            build.ninja: -lcudart -ltorch -ltorch_python ...
#   │
#   │  输出：CMakeCache.txt / build.ninja / compile_commands.json
#   └─────────────────────────────────────────────────────────────────────────────
#
#   ┌─ 编译阶段（ninja / cmake --build） ─────────────────────────────────────────
#   │  触发：ninja -C build/ softmax_cuda -j 18
#   │       （或 cmake --build build/ --target softmax_cuda -- -j 18）
#   │
#   │  ninja 读取 build.ninja，并行执行：
#   │
#   │  g++ src/interface.cpp
#   │      -DTORCH_EXTENSION_NAME=softmax_cuda    ← 编译器 -D → 预处理器宏
#   │      -DBLOCK_DIM_X=512                      ← 编译器 -D → 预处理器宏
#   │      -D_GLIBCXX_USE_CXX11_ABI=1  -std=c++20  -I...
#   │      -o interface.cpp.o
#   │      预处理：PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)→PYBIND11_MODULE(softmax_cuda,m)
#   │
#   │  nvcc src/run_kernels.cu
#   │       -DSOFTMAX_VARIANT=6                   ← 编译器 -D → 预处理器宏
#   │       -DBLOCK_DIM_X=512                     ← 编译器 -D → 预处理器宏
#   │       -DUNROLL_FACTOR=8  -DWIDTH=0  -D_GLIBCXX_USE_CXX11_ABI=1
#   │       -std=c++20  -gencode arch=compute_86,...  -lineinfo  --use_fast_math
#   │       -o run_kernels.cu.o
#   │       预处理：#define SOFTMAX_VARIANT 6（-D 注入）
#   │               #ifndef SOFTMAX_VARIANT → 已定义，跳过默认值 8
#   │               softmax_kernel<SOFTMAX_VARIANT> → softmax_kernel<6>
#   │               #define BLOCK_DIM_X 512（-D 注入）
#   │               dim3 block_size(1, BLOCK_DIM_X, 1) → dim3 block_size(1, 512, 1)
#   │
#   │  ld -shared  interface.cpp.o  run_kernels.cu.o  softmax_kernel.cu.o
#   │     -lcudart -ltorch -ltorch_cpu -lc10 -ltorch_python
#   │     -o softmax_cuda.cpython-313-x86_64-linux-gnu.so
#   └─────────────────────────────────────────────────────────────────────────────
target_compile_definitions(softmax_cuda PRIVATE
    TORCH_EXTENSION_NAME=softmax_cuda
    BLOCK_DIM_X=${BLOCK_DIM_X}
    UNROLL_FACTOR=${UNROLL_FACTOR}
    SOFTMAX_VARIANT=${SOFTMAX_VARIANT}
    WIDTH=${WIDTH}
)

# 头文件搜索路径
#   CUDAToolkit_INCLUDE_DIRS : /usr/local/cuda/include（cuda_runtime.h 等）
#   TORCH_INCLUDE_DIRS       : torch/extension.h / torch/torch.h / pybind11 头文件
#   Python3_INCLUDE_DIRS     : Python.h，由 find_package(Python3 ... Development) 提供
#                              torch/extension.h → python_headers.h → <Python.h> 所需
target_include_directories(softmax_cuda PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# 链接库
#   CUDA::cudart      : CUDA 运行时（cudaMalloc / kernel 调度等）
#   ${TORCH_LIBRARIES}: libtorch.so / libtorch_cpu.so / libc10.so 等
#   ${TORCH_PYTHON_LIBRARY}: libtorch_python.so，PYBIND11_MODULE 展开时需要
target_link_libraries(softmax_cuda PRIVATE
    CUDA::cudart
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)

# set_target_properties 与全局 set(CMAKE_CXX_STANDARD 20) 的区别：
#   作用范围：
#     set(CMAKE_CXX_STANDARD 20)                          : 全局，所有 target 继承
#     set_target_properties(softmax_cuda PROPERTIES ...) : 只作用于 softmax_cuda，
#                                                          优先级高于全局值，可单独覆盖
#   CUDA_STANDARD 无对应全局变量：
#     CMAKE_CXX_STANDARD 只影响 .cpp 的编译，CMake 没有全局 CMAKE_CUDA_STANDARD 自动同步机制；
#     不显式设置 CUDA_STANDARD 时，nvcc 编译 kernels.cu 的设备代码退回默认标准（C++14），
#     与主机代码标准不一致，模板实例化时出现难以定位的错误
#   CXX_STANDARD 20 在本项目中是冗余的：
#     全局已有 CMAKE_CXX_STANDARD 20，softmax_cuda 会自动继承，
#     此处显式写出只是让意图更清晰；真正必要的只有 CUDA_STANDARD 20
set_target_properties(softmax_cuda PROPERTIES
    CXX_STANDARD  20   # 冗余但明确：interface.cpp 用 C++20 编译（继承自 CMAKE_CXX_STANDARD）
    CUDA_STANDARD 20   # 必要：kernels.cu 设备代码也用 C++20，无全局变量可继承
    # LIBRARY_OUTPUT_DIRECTORY：共享库（.so）的输出目录
    #   默认输出到构建目录（cmake-build-release/），改为项目根目录后
    #   Python 脚本可直接 import，无需在 sys.path 中添加构建目录
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# ── validation 可执行文件（独立 kernel 性能测试程序）──────────────────────────
# 与 softmax_cuda（SHARED 库）的区别：
#   softmax_cuda : 无 main()，由 python3 dlopen 加载，用于 Python 侧调用
#   validation   : 有 main()，./validation <kernel_num> 直接运行，用于 C++ 侧基准测试
#
# validation.cu 通过 runner.cuh → kernels.cuh → softmax_kernel.cuh
# softmax_kernel.cuh 只包含纯 CUDA kernel 模板，不依赖 torch/extension.h，
# 因此编译和链接均只需要 CUDA 相关头文件和库
add_executable(validation
    src/validation.cu
    src/runner.cu    # 提供 run_softmax_kernel_base / run_softmax_kernel_naive 的定义
                     # validation.cu 通过 runner.cuh 看到声明，链接时从 runner.cu.o 找到定义
                     # 不加此行：链接报 undefined reference to run_softmax_kernel_base
)

target_include_directories(validation PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}        # cuda_runtime.h / device_functions.h 等
    /usr/include/x86_64-linux-gnu      # cudnn.h：apt 安装 cuDNN 时头文件位于此路径
                                       # /usr/local/cuda/include 不含 cudnn.h，需单独指定
    # 若头文件链中出现 #include <torch/extension.h>，需额外添加：
    #   ${TORCH_INCLUDE_DIRS}          # torch/extension.h 本身所在路径
    #   ${Python3_INCLUDE_DIRS}        # torch/extension.h → python_headers.h → Python.h 所需
)

target_link_libraries(validation PRIVATE
    CUDA::cudart   # cudaMalloc / cudaFree / cudaMemcpy / cudaEvent_t 等运行时符号
    cudnn          # cuDNN 运行时库（libcudnn.so），提供 cudnnCreate / cudnnSoftmaxForward 等符号
                   # apt 安装路径：/usr/lib/x86_64-linux-gnu/libcudnn.so.9
                   # 链接器在默认搜索路径中可直接找到，无需显式指定 PATHS
)

set_target_properties(validation PROPERTIES
    CXX_STANDARD  20
    CUDA_STANDARD 20
    # RUNTIME_OUTPUT_DIRECTORY：可执行文件的输出目录
    #   默认输出到构建目录（cmake-build-release/），改为项目根目录后
    #   run.sh 中 EXEC="./validation" 可直接找到，无需指定构建目录路径
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# ════════════════════════════════════════════════════════════════════════════
# 【CLion / CMake 四个操作的含义与对应命令行】
#
# ── Reload CMake Project ─────────────────────────────────────────────────
#   含义：重新执行 CMakeLists.txt，更新构建规则文件（build.ninja / Makefile）
#         不编译任何源文件，不删除已有 .o 文件
#   触发时机：修改 CMakeLists.txt 后必须执行，否则新的编译选项不会写入构建规则
#             修改 SOFTMAX_VARIANT / BLOCK_DIM_X 等 -D 参数后同样需要 Reload
#   命令行：
#     cmake \
#       -DCMAKE_BUILD_TYPE=Release \                                                  # 构建类型：Release（-O3 -lineinfo）/ Debug（-O0 -G）
#       -DCMAKE_PREFIX_PATH=.venv/lib/python3.13/site-packages/torch/share/cmake \   # find_package(Torch) 从此目录读取 TorchConfig.cmake
#       -DPython3_ROOT_DIR=/home/liam/python_linux/python_venv/.venv \               # find_package(Python3) 在此 venv 目录下搜索解释器
#       -DSOFTMAX_VARIANT=8 \                                                         # 选择 softmax_kernel8，注入为编译期宏
#       -DBLOCK_DIM_X=1024 \                                                          # block Y 维线程数，注入为编译期宏
#       -DUNROLL_FACTOR=8 \                                                           # 循环展开因子，注入为编译期宏
#       -DWIDTH=0 \                                                                   # 行宽静态提示，0 = 运行时由参数决定
#       -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \                                         # 指定底层构建工具路径为 ninja 可执行文件
#       -G Ninja \                                                                    # 生成器：生成 build.ninja 而非 Makefile
#                                                                                     # 选 Ninja 而非 Make 的原因：
#                                                                                     #   Make  : 每次构建递归扫描完整依赖树再决定重建目标，启动开销大
#                                                                                     #   Ninja : 依赖图在配置阶段由 CMake 生成完毕，构建时直接查图，
#                                                                                     #           启动开销极低，并行调度更精确，增量编译速度更快
#                                                                                     #   本项目 kernels.cu 编译慢，只改 interface.cpp 时
#                                                                                     #   Ninja 能更快识别并只重编该文件，减少等待
#                                                                                     #   两者最终产物完全相同，cmake --build 对两者通用
#       -S /home/liam/cpp_linux/Fast_Softmax \                                       # 源码目录：CMake 从这里读取 CMakeLists.txt
#                                                                                     #   -S 是 CMake 3.13 引入的显式标志；早期惯用 cd build && cmake ..
#                                                                                     #   用相对路径 .. 隐式指向源码目录，-S 让路径一目了然，无需 cd
#       -B /home/liam/cpp_linux/Fast_Softmax/cmake-build-release                     # 构建目录：build.ninja / CMakeCache.txt / .o / .so 均输出到此
#
#   CLion 中等价配置（Settings → Build, Execution, Deployment → CMake → CMake options）：
#     -DCMAKE_PREFIX_PATH=/home/liam/python_linux/python_venv/.venv/lib/python3.13/site-packages/torch/share/cmake
#     -DPython3_ROOT_DIR=/home/liam/python_linux/python_venv/.venv
#
# ── Clean ────────────────────────────────────────────────────────────────
#   含义：删除所有编译产物（.o 文件、.so 共享库），保留 CMake 配置（build.ninja）
#         下次 Build 时所有源文件强制重新编译
#   命令行：
#     cmake --build cmake-build-release --target clean
#
# ── Build（增量编译）────────────────────────────────────────────────────
#   含义：只重新编译有变化的源文件（kernels.cu / interface.cpp 内容变了），
#         其余复用已有 .o，最终链接生成 softmax_cuda.cpython-3XX-...so
#         Ninja 通过文件时间戳 / 内容哈希判断哪些文件需要重编译
#   注意：CMakeLists.txt 变了但未 Reload，Build 不会感知编译选项的变化
#         修改了 SOFTMAX_VARIANT 等 -D 参数后，必须先 Reload 再 Build，
#         否则 kernels.cu 中的 #ifndef 保护不会生效，仍用旧的宏值编译
#   命令行：
#     cmake --build cmake-build-release --target softmax_cuda -- -j 18
#     │      │      │                   │       │              │   │
#     │      │      │                   │       │              │   └─ 传给底层构建工具（Ninja/Make）的并行参数
#     │      │      │                   │       │              │      -j 18：同时启动 18 个编译进程
#     │      │      │                   │       │              └─ "--" 之后的所有参数透传给底层构建工具
#     │      │      │                   │       │                 cmake 自身不再解析 "--" 之后的内容
#     │      │      │                   │       └─ 只构建名为 softmax_cuda 的 target
#     │      │      │                   │          对应 add_library(softmax_cuda SHARED ...)
#     │      │      │                   └─ --target：指定要构建的目标名称
#     │      │      └─ 构建目录：cmake 在此目录下寻找 build.ninja / Makefile
#     │      │         由 cmake -B cmake-build-release 生成（Reload CMake Project 时创建）
#     │      └─ --build：cmake 进入构建模式（而非配置模式）
#     └─ cmake 主程序
#
#   cmake --build 与 make 的关系：
#     cmake --build 是对底层构建工具的封装，本项目生成器为 Ninja，
#     cmake 读取构建目录后实际执行：ninja -C cmake-build-release softmax_cuda -j 18
#     两者本质等价，区别在于：
#       cmake --build：生成器无关（Ninja/Make 均适用），可在任意目录执行（路径作参数）
#       ninja / make ：只对各自生成器有效，必须在含 build.ninja / Makefile 的目录下执行
#     ninja 的三种等价写法：
#       cd cmake-build-release && ninja softmax_cuda -j 18   # cd 进去再执行
#       ninja -C cmake-build-release softmax_cuda -j 18      # -C 指定目录，无需 cd
#       cmake --build cmake-build-release ...                 # cmake 内部处理 cd
#
# ── Rebuild ──────────────────────────────────────────────────────────────
#   含义：等价于 Clean + Build，先删除所有产物，再完整重新编译
#         用于解决增量编译状态不一致、头文件依赖混乱等问题
#   命令行：
#     cmake --build cmake-build-release --target clean
#     cmake --build cmake-build-release --target softmax_cuda -- -j 18
#
# ── 自动触发说明 ────────────────────────────────────────────────────────
#   CLion 不会自动执行 Reload 或 Build，两者均需手动触发：
#     修改 CMakeLists.txt 后：CLion 顶部弹出提示栏，点击 "Reload CMake Project"
#                             （Settings → CMake → "Reload CMake project on changes" 可开启自动 Reload，
#                               但默认关闭——因为 Reload 会执行 execute_process，如 nvidia-smi / python，有耗时）
#     修改 .cu / .cpp 后    ：保存文件，手动点击 Build 或按快捷键，才会触发增量编译
#                             CLion 不在保存时自动编译，CUDA 单文件编译耗时长，自动触发不合适
#
# ── 常见工作流 ──────────────────────────────────────────────────────────
#   修改 CMakeLists.txt                → 手动 Reload → 手动 Build
#   修改 SOFTMAX_VARIANT / BLOCK_DIM_X → 手动 Reload → 手动 Build（宏值变化必须重新配置）
#   只修改 kernels.cu / interface.cpp  → 手动 Build（增量编译，只重编改动的文件）
#   编译状态混乱 / 链接报奇怪错误      → 手动 Rebuild
#   彻底清除重来                        → 删除 cmake-build-release 目录 → 手动 Reload → 手动 Build
#
# ── 构建产物与 Python 使用 ──────────────────────────────────────────────
#   产物路径：cmake-build-release/softmax_cuda.cpython-3XX-x86_64-linux-gnu.so
#   Python 中加载：
#     import sys
#     sys.path.insert(0, '/home/liam/cpp_linux/Fast_Softmax/cmake-build-release')
#     import softmax_cuda
#     y = softmax_cuda.softmax_cuda(x)
# ════════════════════════════════════════════════════════════════════════════
