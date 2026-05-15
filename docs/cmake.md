# Fast_Softmax CMakeLists.txt 注解

---

## 目录

1. [项目初始化](#1-项目初始化)
2. [工具链与依赖查找](#2-工具链与依赖查找)
3. [GPU 架构检测](#3-gpu-架构检测)
4. [Kernel 编译参数](#4-kernel-编译参数)
5. [全局编译选项](#5-全局编译选项)
6. [共享库构建](#6-共享库构建)
7. [ELF 二进制类型](#7-elf-二进制类型)
8. [CLion 操作与命令行对应](#8-clion-操作与命令行对应)

---

## 1. 项目初始化

### 编译器路径锁定

```cmake
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_COMPILER  "/usr/bin/g++")
```

必须在 `project()` 之前设置，避免多版本共存时 CMake 选到错误的编译器。

### project()

```cmake
project(Fast_Softmax LANGUAGES CXX CUDA)
```

| 参数 | 作用 |
|------|------|
| `Fast_Softmax` | 项目名称 |
| `LANGUAGES CXX CUDA` | 启用 C++ 和 CUDA 两种编译器 |

`LANGUAGES` 决定后续 `add_library` / `add_executable` 能编译的文件类型：

- `.cpp` → `g++` 编译为纯主机代码（`interface.cpp` 中的 pybind11 模块定义）
- `.cu` → `nvcc` 编译为主机 + 设备混合代码（`kernels.cu` 中的 softmax kernel）

### 导出编译命令

```cmake
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
```

生成 `compile_commands.json`，供 clangd / clang-tidy 做代码分析和补全。

### C++ 标准

```cmake
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

- PyTorch 2.x 要求至少 C++17；C++20 作为超集向后兼容，可直接使用
- `CMAKE_CXX_STANDARD_REQUIRED ON`：编译器不支持 C++20 时直接报错终止，而非静默降级
  - 不设置（默认）：悄悄降级，你以为在用 C++20 实际没有
  - 设置为 ON：不支持则 fatal error，确保编译标志确实是 `-std=c++20`

---

## 2. 工具链与依赖查找

### find_package(CUDAToolkit)

```cmake
find_package(CUDAToolkit REQUIRED)
```

CMake 3.17+ 的现代方式，提供 IMPORTED target：`CUDA::cudart` / `CUDA::cublas` 等。
相比旧式 `find_package(CUDA)`，路径查找更可靠，变量名更规范。

### find_package(Python3)

```cmake
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
```

| 参数 | 说明 |
|------|------|
| `REQUIRED` | 找不到则立即报错终止 |
| `COMPONENTS Interpreter` | 查找 Python 解释器（`python3` 可执行文件） |
| `COMPONENTS Development` | 查找 Python 开发头文件和库（`Python.h` / `libpython.so`） |

**为什么需要 `Development` 组件：**

```
torch/extension.h → torch/csrc/python_headers.h → <Python.h>
```

编译 `interface.cpp` 时 `torch/extension.h` 的包含链最终需要 `Python.h`。若只指定 `Interpreter`，`Python3_INCLUDE_DIRS` 不会被设置，编译报：

```
fatal error: Python.h: No such file or directory
```

**成功后设置的变量：**

- `Python3_EXECUTABLE`：venv 解释器的完整路径，如 `/home/liam/python_linux/python_venv/.venv/bin/python3`
- `Python3_VERSION`：解释器版本号，如 `3.13.5`
- `Python3_INCLUDE_DIRS`：`Python.h` 所在目录，如 `/usr/include/python3.13`
- `Python3_LIBRARIES`：`libpython.so` 完整路径

**指定 venv 解释器的两种方式（在 CLion CMake options 或命令行 `-D` 中传入）：**

```
-DPython3_ROOT_DIR=/home/liam/python_linux/python_venv/.venv
    CMake 在 .venv/bin/ 下搜索 python3

-DPython3_EXECUTABLE=/home/liam/python_linux/python_venv/.venv/bin/python3
    直接锁定到具体可执行文件，跳过所有搜索逻辑，最精确
    路径来源：venv 激活后执行 which python3 或 readlink -f $(which python3)
```

两者同时指定时 `EXECUTABLE` 优先。

### find_package(Torch)

```cmake
set(TORCH_CUDA_ARCH_LIST "${_GPU_CC}")   # 必须在 find_package(Torch) 之前
find_package(Torch REQUIRED)
```

**成功后设置的变量：**

| 变量 | 内容 |
|------|------|
| `TORCH_INCLUDE_DIRS` | `torch/extension.h` 等头文件路径 |
| `TORCH_LIBRARIES` | `libtorch.so` / `libtorch_cpu.so` 等 |
| `TORCH_CXX_FLAGS` | PyTorch 要求的 ABI 标志（如 `-D_GLIBCXX_USE_CXX11_ABI=1`） |
| `TORCH_INSTALL_PREFIX` | LibTorch 安装根目录 |

需在 CLion CMake options 中添加：

```
-DCMAKE_PREFIX_PATH=/home/liam/python_linux/python_venv/.venv/lib/python3.13/site-packages/torch/share/cmake
```

**`TORCH_CUDA_ARCH_LIST` 与 `CMAKE_CUDA_ARCHITECTURES` 的顺序要求：**

PyTorch 的 `cuda.cmake` 第 322 行检查：

```cmake
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(WARNING "pytorch is not compatible with CMAKE_CUDA_ARCHITECTURES...")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()
```

只要 `CMAKE_CUDA_ARCHITECTURES` 在 `find_package(Torch)` 执行时已定义，就触发 Warning 并强制覆盖为 OFF。

**修复方法（顺序关键）：**

```cmake
set(TORCH_CUDA_ARCH_LIST "8.6")         # find_package(Torch) 之前 ← PyTorch 读取此变量
find_package(Torch REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 86)        # find_package(Torch) 之后 ← 执行时变量尚未定义
```

- `TORCH_CUDA_ARCH_LIST`：带点版本号 `"8.6"`（PyTorch 专用格式）
- `CMAKE_CUDA_ARCHITECTURES`：无点整数 `86`（CMake / nvcc 格式）

#### find_package(Torch) 产生的三个 Warning

| Warning | 来源 | 影响 | 处理方式 |
|---------|------|------|----------|
| `Failed to compute shorthash for libnvrtc.so` | `cuda.cmake`，对 NVRTC 库做完整性校验 | 无，本项目不使用 NVRTC | 可忽略 |
| `pytorch is not compatible with CMAKE_CUDA_ARCHITECTURES` | `cuda.cmake` 第 322 行 | 会将 `CMAKE_CUDA_ARCHITECTURES` 强制设为 OFF | 将 `CMAKE_CUDA_ARCHITECTURES` 移到 `find_package(Torch)` 之后设置 |
| `static library kineto_LIBRARY-NOTFOUND not found` | `TorchConfig.cmake`，Kineto 性能分析库未编译 | 无，不影响 CUDA kernel 编译和运行 | 可忽略 |

### find_library(TORCH_PYTHON_LIBRARY)

```cmake
find_library(TORCH_PYTHON_LIBRARY torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    REQUIRED
)
```

**`find_library` 与 `find_package` 的区别：**

| | `find_package` | `find_library` |
|-|----------------|----------------|
| 层级 | 高层接口 | 低层接口 |
| 依赖 | 包自带的 `XxxConfig.cmake` | 无 |
| 输出 | 头文件路径、库路径、编译标志、IMPORTED target 等全套信息 | 只设置 `.so` 完整路径变量 |

`libtorch_python.so` 用 `find_library` 的原因：它是 PyTorch 的内部子库，没有自己的 cmake 包文件，`find_package(Torch)` 也故意未将其包含在 `TORCH_LIBRARIES` 中（让用户按需链接），只能用 `find_library` 直接定位文件。

**参数说明：**

- `TORCH_PYTHON_LIBRARY`：结果变量，存储 `libtorch_python.so` 的完整路径
- `torch_python`：库名（不含 `lib` 前缀和 `.so` 后缀），`find_library` 自动拼接为 `libtorch_python.so`
- `PATHS "${TORCH_INSTALL_PREFIX}/lib"`：搜索目录
- `REQUIRED`：找不到则报错终止

### check_language(CUDA)

```cmake
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found")
endif()
```

- `include(CheckLanguage)`：加载 CMake 内置模块，引入 `check_language()` 函数，不加载则调用会报错
- `check_language(CUDA)`：实际编译一段最小 CUDA 代码来验证 nvcc 能否正常工作
  - 结果写入缓存变量 `CMAKE_CUDA_COMPILER`：有效路径表示成功，`NOTFOUND` 表示失败
- `project(... LANGUAGES CUDA)` 已探测编译器，此处是额外的防御性验证，不可用时主动报 `FATAL_ERROR`，比等编译命令失败时的报错更易定位

**错误：`CUDA_ARCHITECTURES is empty for target "cmTC_0c802"`**

```
CMake Error in CMakeLists.txt:
  CUDA_ARCHITECTURES is empty for target "cmTC_0c802".
```

- `cmTC_0c802`：`check_language(CUDA)` 在内部创建的临时测试 target，名称由 CMake 随机生成
- 报错原因：`check_language(CUDA)` 编译测试程序时，`CMAKE_CUDA_ARCHITECTURES` 为空，nvcc 不知道目标架构，编译失败
- 触发场景：`unset(CMAKE_CUDA_ARCHITECTURES CACHE)` 清除了架构值，而 `set(CMAKE_CUDA_ARCHITECTURES ...)` 的位置在 `check_language(CUDA)` 之后，执行到 `check_language` 时变量仍为空

**修复：严格的四步顺序**

```cmake
# 1. 查询 GPU 架构 → _GPU_CC = "8.6"，_GPU_CC_NUM = "86"
execute_process(COMMAND nvidia-smi ...)
string(REPLACE ".." "" _GPU_CC_NUM "${_GPU_CC}")

# 2. 告知 PyTorch 目标架构（find_package(Torch) 之前）
set(TORCH_CUDA_ARCH_LIST "${_GPU_CC}")
# 3. 清除旧缓存值，防止 pytorch 的 Warning 2（find_package(Torch) 之前）
unset(CMAKE_CUDA_ARCHITECTURES CACHE)

# 4. PyTorch 读取 TORCH_CUDA_ARCH_LIST，此时 CMAKE_CUDA_ARCHITECTURES 未定义，不触发 Warning
find_package(Torch REQUIRED)

# 5. 设置 CMAKE_CUDA_ARCHITECTURES（find_package(Torch) 之后，check_language 之前）
set(CMAKE_CUDA_ARCHITECTURES ${_GPU_CC_NUM})

# 6. check_language 编译测试时 CMAKE_CUDA_ARCHITECTURES 已有值，不报错
check_language(CUDA)
```

| 约束 | 原因 |
|------|------|
| `nvidia-smi` 在 `find_package(Torch)` 之前 | `TORCH_CUDA_ARCH_LIST` 须在 `find_package(Torch)` 前设好，依赖 `_GPU_CC` |
| `unset(CACHE)` 在 `find_package(Torch)` 之前 | PyTorch cuda.cmake 检查时 `CMAKE_CUDA_ARCHITECTURES` 必须未定义 |
| `set(CMAKE_CUDA_ARCHITECTURES)` 在 `find_package(Torch)` 之后 | 若在之前设置，触发 PyTorch Warning 2 并被强制覆盖为 OFF |
| `set(CMAKE_CUDA_ARCHITECTURES)` 在 `check_language` 之前 | `check_language` 编译测试 target 时需要非空的架构值 |

---

## 3. GPU 架构检测

### nvidia-smi 查询 Compute Capability

```cmake
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _GPU_CC
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
```

**`execute_process` 与 `add_custom_command` 的区别：**

| | `execute_process` | `add_custom_command` |
|-|-------------------|----------------------|
| 执行时机 | cmake 配置阶段（`cmake -S ... -B ...`）立即执行 | 构建阶段（`make` / `ninja`）执行 |
| 用途 | 查询系统信息，结果作为配置信息使用 | 构建步骤（生成文件、运行脚本等） |

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--query-gpu=compute_cap` | 查询 GPU Compute Capability，输出格式为 `"8.6"` |
| `--format=csv,noheader` | 去掉表头，只保留数值（多卡时每张卡一行） |
| `OUTPUT_VARIABLE _GPU_CC` | 将 stdout 存入变量 `_GPU_CC` |
| `OUTPUT_STRIP_TRAILING_WHITESPACE` | 去掉末尾换行符，否则 `_GPU_CC` 为 `"8.6\n"` |
| `ERROR_QUIET` | nvidia-smi 不存在时静默失败，不中断配置 |

### 格式转换与错误处理

```cmake
string(REPLACE "." "" _GPU_CC_NUM "${_GPU_CC}")
if(NOT _GPU_CC_NUM)
    message(FATAL_ERROR "无法检测 GPU 架构，请确认 nvidia-smi 可用")
endif()
```

- `string(REPLACE "." "" ...)` 将 `"8.6"` 转为 `"86"`，nvcc 的 `-gencode` 参数需要无小数点的整数形式
- `if(NOT _GPU_CC_NUM)` 在变量为空字符串或未定义时条件为真（两种情况：nvidia-smi 不存在，或输出异常）

### CMAKE_CUDA_ARCHITECTURES

```cmake
set(CMAKE_CUDA_ARCHITECTURES ${_GPU_CC_NUM})
```

CMake 内部将其展开为 nvcc 的 `-gencode` 标志：

```
-gencode arch=compute_86,code=[sm_86,compute_86]
```

| 标志 | 作用 |
|------|------|
| `code=sm_86` | 生成 SASS 二进制，在当前 GPU 上直接执行，性能最优 |
| `code=compute_86` | 保留 PTX 中间码，供未来更新架构的 GPU JIT 编译（向前兼容） |

**常见架构号对照：**

| 号码 | 架构 | 代表 GPU |
|------|------|----------|
| 75 | Turing | RTX 20 系列 |
| 80 | Ampere | A100 |
| 86 | Ampere | RTX 30 系列 |
| 89 | Ada | RTX 40 系列 |
| 90 | Hopper | H100 |

**作用范围：** 对之后所有 `add_library` / `add_executable` 创建的 target 全局生效，各 target 自动继承，无需逐个调用 `set_target_properties(... CUDA_ARCHITECTURES ...)`。

**不设置全局值时的替代方式（效果相同，但需逐个 target 指定）：**

```cmake
set_target_properties(softmax_cuda PROPERTIES CUDA_ARCHITECTURES 86)
# 或命令行
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ...
```

---

## 4. Kernel 编译参数

### CACHE 变量

```cmake
set(SOFTMAX_VARIANT  8    CACHE STRING "Softmax kernel variant (1-9), matches softmax_kernelN")
set(BLOCK_DIM_Y      1024 CACHE STRING "Block dim Y, power of 2 in [128, 1024]")
set(UNROLL_FACTOR    8    CACHE STRING "Loop unroll factor for kernel7+: 1/2/4/8")
set(WIDTH            0    CACHE STRING "Static row width hint; 0 means use runtime argument")
```

**`CACHE` 变量与普通变量的区别：**

| | 普通 `set()` | `CACHE set()` |
|-|--------------|---------------|
| 生命周期 | 只存在于当前 cmake 运行内存，每次 Reload 重新赋值 | 写入 `CMakeCache.txt`，跨 Reload 持久保存 |
| 与 `-D` 的关系 | 每次覆盖 | 若用户已通过 `-D` 传入值，不会覆盖，保留用户的值 |

**优先级：`-D` > cache 已有值 > `set()` 默认值**

**三种场景下的行为：**

| 场景 | 结果 |
|------|------|
| 首次 Reload，未传 `-D` | cache 无值 → `set()` 写入默认值 8 → 编译用 8 |
| Reload 时传 `-DSOFTMAX_VARIANT=6` | CMake 把 6 写入 cache → `set()` 检测到已有值不覆盖 → 编译用 6 |
| 再次 Reload，未传 `-D` | cache 仍有 6（上次写入）→ `set()` 不覆盖 → 编译仍用 6 |
| 重置为默认值 | 删除 `cmake-build-release/CMakeCache.txt` 后 Reload，或显式传 `-DSOFTMAX_VARIANT=8` |

**`STRING` 类型**：在 CLion / cmake-gui 中以文本框展示，类型仅影响 GUI 展示方式，不影响 CMake 内部的变量处理。

**四个宏的含义：**

| 宏 | 作用 |
|----|------|
| `SOFTMAX_VARIANT` | 选择 kernel 实现（1~9），对应 `softmax_kernel1~9` |
| `BLOCK_DIM_Y` | 每个 block 在 Y 维度的线程数，必须是 2 的幂次（128~1024） |
| `UNROLL_FACTOR` | 循环展开因子（1/2/4/8），用于 kernel7/8/9 的向量化循环 |
| `WIDTH` | 行宽静态提示（0 = 运行时由参数决定），非零时允许编译器静态优化 |

这四个宏等价于 `benchmark.py` 中 `torch.utils.cpp_extension.load` 的 `extra_cuda_cflags` / `extra_cflags`。

**用法示例（选择 variant=6，block_y=512）：**

```bash
cmake -DSOFTMAX_VARIANT=6 -DBLOCK_DIM_Y=512 ..
```

---

## 5. 全局编译选项

### 生成器表达式

`add_compile_options` 中的生成器表达式（`$<...>`）在 cmake 配置阶段求值，按条件决定是否将选项写入 `build.ninja`。

**各模式实际 nvcc 命令行（以架构 86 为例）：**

```
Debug  模式：nvcc -G -lineinfo（不加）--ptxas-options=-v --use_fast_math（不加）
Release模式：nvcc    -lineinfo          --ptxas-options=-v --use_fast_math
```

### -G（Debug 模式）

```cmake
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
```

为 GPU 设备代码生成调试符号，供 `cuda-gdb` 单步调试 kernel。

副作用：禁用几乎所有 GPU 优化，性能大幅下降，不可用于性能测量。

### -lineinfo（Release 模式）

```cmake
add_compile_options("$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")
```

保留源码行号，让 `ncu` 能将性能数据映射回源码行。与 `-G` 的区别：不禁止优化，Release 下 `ncu` 分析才能反映真实性能。

### --use_fast_math（Release 模式）

```cmake
add_compile_options("$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--use_fast_math>")
```

将标准数学函数替换为 GPU 硬件内置近似实现，实际展开为三个子选项：

| 子选项 | 作用 |
|--------|------|
| `-ftz=true` | 把非规格化浮点数（denormal）刷成零，避免硬件慢速处理路径 |
| `-prec-div=false` | 用快速近似除法替代精确除法 |
| `-prec-sqrt=false` | 用快速近似平方根替代精确平方根 |

同时将标准函数自动替换为硬件内置近似版本：`exp()` → `__expf()`，`sin()` → `__sinf()`，等。

**加此选项的原因：** 与 `benchmark.py` 的 JIT 编译参数保持一致，确保两种构建路径（JIT / CMake）编译出的 kernel 行为相同，benchmark 结果才可互相对比。

### --ptxas-options=-v（两种模式均开启）

```cmake
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>")
```

输出每个 kernel 的寄存器 / 共享内存用量，用于分析 occupancy。

**按模式控制的写法对比：**

```cmake
# 两种模式均开启（当前写法）：
$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>

# 只开启 Debug：
$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:--ptxas-options=-v>

# 只开启 Release：
$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--ptxas-options=-v>
```

不加 `$<CONFIG:xxx>` 条件则两种模式都触发。

### TORCH_CXX_FLAGS（ABI 兼容）

```cmake
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
add_compile_options(${TORCH_CXX_FLAGS_LIST})
```

`TORCH_CXX_FLAGS` 由 `find_package(Torch)` 设置，包含 ABI 标志，必须应用到所有编译单元，否则与 `libtorch.so` 的 ABI 不匹配，链接时报 `undefined reference`。

**为什么需要 `separate_arguments`：**

- `TORCH_CXX_FLAGS` 是空格分隔的普通字符串，如 `"-D_GLIBCXX_USE_CXX11_ABI=1 -Wall"`
- 直接传给 `add_compile_options("${TORCH_CXX_FLAGS}")` 会被当作一个整体选项（含空格的单参数），编译器报错
- 拆分后变为 CMake list（分号分隔），`add_compile_options` 展开 list 后编译器收到两个独立参数，正确解析

`NATIVE_COMMAND`：按当前平台的 shell 规则拆分（Linux 以空格为分隔符），无需关心平台差异。

**本机实际值（PyTorch 以 CXX11 ABI 编译）：**

```
TORCH_CXX_FLAGS      = "-D_GLIBCXX_USE_CXX11_ABI=1"       ← 空格分隔的普通字符串
TORCH_CXX_FLAGS_LIST = "-D_GLIBCXX_USE_CXX11_ABI=1"       ← 拆分后的 CMake list（此处只有一项）
```

等价于对所有编译命令追加：

```
nvcc src/kernels.cu    ... -D_GLIBCXX_USE_CXX11_ABI=1
g++  src/interface.cpp ... -D_GLIBCXX_USE_CXX11_ABI=1
```

---

## 6. 共享库构建

### aux_source_directory

```cmake
aux_source_directory(${PROJECT_SOURCE_DIR}/src/kernels KERNELS)
```

扫描指定目录，将其中所有已知扩展名的源文件（`.cu` / `.cpp` / `.c` 等）收集到变量中。

**参数说明：**

- 第一个参数：要扫描的目录路径
- 第二个参数：结果变量名，收集到的文件列表以分号分隔存入该变量（CMake list 格式）

**特性与注意事项：**

| 特性 | 说明 |
|------|------|
| 非递归 | 只扫描指定目录本身，不递归进入子目录 |
| 已知扩展名 | CMake 内置列表：`.c` `.cpp` `.cxx` `.cc` `.cu` 等；`.cuh` / `.h` 不会被收集 |
| 新增文件需 Reload | `aux_source_directory` 在配置阶段（Reload）执行一次，之后新增的文件不会自动加入，必须再次 Reload |

**与手动列举源文件的区别：**

```cmake
# 手动列举（新增文件必须手动加）：
add_library(softmax_cuda SHARED src/kernels/softmax_kernel.cu)

# aux_source_directory（新增 .cu 文件后 Reload 即可自动收集）：
aux_source_directory(${PROJECT_SOURCE_DIR}/src/kernels KERNELS)
add_library(softmax_cuda SHARED ${KERNELS})
```

### add_library

```cmake
aux_source_directory(${PROJECT_SOURCE_DIR}/src/kernels KERNELS)

add_library(softmax_cuda SHARED
    src/interface.cpp   # g++  编译：pybind11 模块定义，纯主机代码
    src/kernels.cu      # nvcc 编译：kernel 调度函数 softmax_cu()，主机 + 设备混合代码
    ${KERNELS}          # nvcc 编译：src/kernels/ 下的各 kernel 实现（如 softmax_kernel.cu）
)
```

构建为 SHARED 库而非可执行文件，Python `import` 时动态加载 `.so`。每个源文件由 CMake 根据扩展名选择编译器独立编译：

| 源文件 | 编译器 | 产物 |
|--------|--------|------|
| `src/interface.cpp` | `g++` | `interface.cpp.o`（纯主机代码） |
| `src/kernels.cu` | `nvcc` | `kernels.cu.o`（主机 + 设备混合） |
| `src/kernels/softmax_kernel.cu` | `nvcc` | `softmax_kernel.cu.o`（kernel 实现） |

三个 `.o` 最终由链接器合并为 `softmax_cuda.so`。

### Python 扩展文件名

```cmake
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE _PY_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
```

Python C 扩展的文件命名规范：`<模块名>.<python版本>-<平台>.so`

本机实际值（Python 3.13，x86_64 Linux）：

```
softmax_cuda.cpython-313-x86_64-linux-gnu.so
```

若文件名不符合此规范，`import softmax_cuda` 找不到文件，报 `ModuleNotFoundError`。

### 设置文件名属性

```cmake
set_target_properties(softmax_cuda PROPERTIES PREFIX "" SUFFIX "${_PY_EXT_SUFFIX}")
```

| 属性 | 执行前（CMake 默认） | 执行后 |
|------|---------------------|--------|
| `PREFIX` | `"lib"` | `""` |
| `SUFFIX` | `".so"` | `".cpython-313-x86_64-linux-gnu.so"` |
| 最终文件名 | `libsoftmax_cuda.so` | `softmax_cuda.cpython-313-x86_64-linux-gnu.so` |

### target_compile_definitions

```cmake
target_compile_definitions(softmax_cuda PRIVATE
    TORCH_EXTENSION_NAME=softmax_cuda
    BLOCK_DIM_Y=${BLOCK_DIM_Y}
    UNROLL_FACTOR=${UNROLL_FACTOR}
    SOFTMAX_VARIANT=${SOFTMAX_VARIANT}
    WIDTH=${WIDTH}
)
```

将宏定义注入指定 target 的编译命令，等价于给编译器追加 `-D<宏名>=<值>`。

**执行阶段：**

```
set(SOFTMAX_VARIANT 8 CACHE ...)         ← 配置阶段（Reload）：把值存入 CMakeCache.txt
target_compile_definitions(...)          ← 配置阶段（Reload）：读取 CMake 变量，把 -DSOFTMAX_VARIANT=8 写入 build.ninja
nvcc 编译 kernels.cu                     ← 编译阶段（Build）预处理：读取 build.ninja 中的 -DSOFTMAX_VARIANT=8
```

`target_compile_definitions` 本身在配置阶段执行（写 `build.ninja`），对 `kernels.cu` 的实际影响发生在编译阶段的预处理器步骤。

**`PRIVATE` 作用域：**

| 作用域 | 说明 |
|--------|------|
| `PRIVATE` | 宏定义只作用于 `softmax_cuda` 自身的编译 |
| `PUBLIC` | 自身编译 + 传递给链接了此 target 的其他 target |
| `INTERFACE` | 只传递，自身不用（用于纯头文件库） |

本项目是叶子 target（无其他 target 链接它），`PRIVATE` / `PUBLIC` 效果相同，用 `PRIVATE` 更准确。

**各宏定义展开（本机默认值示例）：**

```
TORCH_EXTENSION_NAME=softmax_cuda  →  -DTORCH_EXTENSION_NAME=softmax_cuda
BLOCK_DIM_Y=${BLOCK_DIM_Y}         →  -DBLOCK_DIM_Y=1024
UNROLL_FACTOR=${UNROLL_FACTOR}     →  -DUNROLL_FACTOR=8
SOFTMAX_VARIANT=${SOFTMAX_VARIANT} →  -DSOFTMAX_VARIANT=8
WIDTH=${WIDTH}                     →  -DWIDTH=0
```

`TORCH_EXTENSION_NAME` 必须与库文件名一致，否则 `import` 时 Python 找不到 `PyInit_softmax_cuda` 入口，报 `ImportError`。

**`kernels.cu` 中 `#ifndef` 保护与 `-D` 标志的关系：**

```c
#ifndef BLOCK_DIM_Y
#define BLOCK_DIM_Y 1024   // 未传 -D 时用此默认值
#endif
// 编译时收到 -DBLOCK_DIM_Y=1024，#ifndef 条件为假，跳过此块，使用传入值
```

**与 `add_compile_options` 的区别：**

| | `add_compile_options` | `target_compile_definitions` |
|-|-----------------------|------------------------------|
| 作用范围 | 对所有 target 全局生效 | 只对指定 target 生效 |

### target_include_directories

```cmake
target_include_directories(softmax_cuda PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)
```

| 变量 | 内容 |
|------|------|
| `CUDAToolkit_INCLUDE_DIRS` | `/usr/local/cuda/include`（`cuda_runtime.h` 等） |
| `TORCH_INCLUDE_DIRS` | `torch/extension.h` / `torch/torch.h` / `pybind11` 头文件 |
| `Python3_INCLUDE_DIRS` | `Python.h` 所在目录，由 `find_package(Python3 ... Development)` 提供 |

`Python3_INCLUDE_DIRS` 必须显式加入的原因：`TORCH_INCLUDE_DIRS` 只包含 PyTorch 自身的头文件目录，不包含 Python 头文件目录；但 `torch/extension.h` 在其包含链中需要 `Python.h`，若不加入此目录，编译 `interface.cpp` 时会报 `Python.h: No such file or directory`。

### target_link_libraries

```cmake
target_link_libraries(softmax_cuda PRIVATE
    CUDA::cudart
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)
```

| 库 | 内容 |
|----|------|
| `CUDA::cudart` | CUDA 运行时（`cudaMalloc` / kernel 调度等） |
| `${TORCH_LIBRARIES}` | `libtorch.so` / `libtorch_cpu.so` / `libc10.so` 等 |
| `${TORCH_PYTHON_LIBRARY}` | `libtorch_python.so`，`PYBIND11_MODULE` 展开时需要 |

### 目标 C++/CUDA 标准

```cmake
set_target_properties(softmax_cuda PROPERTIES
    CXX_STANDARD  20
    CUDA_STANDARD 20
)
```

**`set_target_properties` 与全局 `set(CMAKE_CXX_STANDARD 20)` 的区别：**

| | 全局 `CMAKE_CXX_STANDARD` | `set_target_properties PROPERTIES` |
|-|---------------------------|-------------------------------------|
| 作用范围 | 所有 target 继承 | 只作用于指定 target，优先级更高 |

**`CUDA_STANDARD` 必须显式设置的原因：**

`CMAKE_CXX_STANDARD` 只影响 `.cpp` 的编译，CMake 没有全局 `CMAKE_CUDA_STANDARD` 自动同步机制。不显式设置 `CUDA_STANDARD` 时，nvcc 编译 `kernels.cu` 的设备代码退回默认标准（C++14），与主机代码标准不一致，模板实例化时出现难以定位的错误。

`CXX_STANDARD 20` 在本项目中是冗余的（全局已有 `CMAKE_CXX_STANDARD 20`，会自动继承），显式写出只是让意图更清晰；真正必要的只有 `CUDA_STANDARD 20`。

---

## 7. ELF 二进制类型

三者均为 ELF（Executable and Linkable Format）二进制，ELF 头中的 `type` 字段区分身份。

### .o（目标文件，ET_REL）

单个源文件编译后的中间产物：

- 内含机器码片段，但外部符号（如 `cudaMalloc` / `torch::xxx`）地址未填，只有占位符
- 不能独立运行
- 存在意义是增量编译：只有改动的源文件重新生成 `.o`，没改的复用旧 `.o`，最后统一链接

### .so（共享对象，ET_DYN）

多个 `.o` 链接后的完整产物：

- 所有符号地址已解析填入，是完整可加载的二进制
- 没有 `main()` 入口，不能独立运行，须由可执行文件在运行时通过 `dlopen()` 加载
- PTX 作为数据段嵌入 `.o` 后随链接进入 `.so`，驱动运行时从 `.so` 中读取并 JIT 编译

### 可执行文件（ET_EXEC / ET_DYN）

有 `main()` 入口，可独立运行。本项目中 `python3` 是可执行文件，`softmax_cuda.so` 是它加载的库：

```
python3 → import softmax_cuda → dlopen(softmax_cuda.so) → 函数可调用
```

`.so` 不会"变成"可执行文件，也不存在从 `.so` 到可执行文件的转换路径。

### 完整构建流程对比

**本项目 `add_library(softmax_cuda SHARED ...)`：**

```
src/interface.cpp  ──(g++)──▶  interface.cpp.o  ─┐
                                                 ├─(ld)──▶  softmax_cuda.so  ← 无入口，须由 python3 dlopen 加载
src/kernels.cu     ──(nvcc)─▶  kernels.cu.o     ─┘
  含 SASS（sm_86）和 PTX（compute_86）两段，均嵌入 .o，链接后进入 .so
```

**参考 GEMM 项目 `add_executable(gemm ...)`：**

```
gemm.cu      ──(nvcc)──▶  gemm.cu.o      ─┐
sgemm_v1.cu  ──(nvcc)──▶  sgemm_v1.cu.o  ─┤
sgemm_v2.cu  ──(nvcc)──▶  sgemm_v2.cu.o  ─┼─(ld)──▶  gemm  ← 有 main() 入口，./gemm 直接运行
utils.cpp    ──(g++)───▶  utils.cpp.o    ─┘
```

两者编译阶段完全相同，区别仅在链接阶段：

| 链接类型 | 产物 | 说明 |
|----------|------|------|
| `add_library SHARED` | 无入口的共享库（ET_DYN，无 main） | Python `dlopen` 加载 |
| `add_executable` | 有入口的可执行文件（ET_EXEC / ET_DYN，有 main） | `./gemm` 直接运行 |

---

## 8. CLion 操作与命令行对应

### Reload CMake Project

**含义：** 重新执行 `CMakeLists.txt`，更新构建规则文件（`build.ninja` / `Makefile`）。不编译任何源文件，不删除已有 `.o` 文件。

**触发时机：** 修改 `CMakeLists.txt` 后必须执行；修改 `SOFTMAX_VARIANT` / `BLOCK_DIM_Y` 等 `-D` 参数后同样需要。

**命令行：**

```bash
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=.venv/lib/python3.13/site-packages/torch/share/cmake \
  -DPython3_ROOT_DIR=/home/liam/python_linux/python_venv/.venv \
  -DSOFTMAX_VARIANT=8 \
  -DBLOCK_DIM_Y=1024 \
  -DUNROLL_FACTOR=8 \
  -DWIDTH=0 \
  -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
  -G Ninja \
  -S /home/liam/cpp_linux/Fast_Softmax \
  -B /home/liam/cpp_linux/Fast_Softmax/cmake-build-release
```

**选 Ninja 而非 Make 的原因：**

| | Ninja | Make |
|-|-------|------|
| 依赖图 | 配置阶段由 CMake 生成完毕，构建时直接查图 | 每次构建递归扫描完整依赖树再决定重建目标 |
| 启动开销 | 极低 | 相对较大 |
| 增量编译 | 并行调度更精确，只改 `interface.cpp` 时更快识别只重编该文件 | 较慢 |
| 最终产物 | 完全相同 | 完全相同 |

**CLion 中等价配置（Settings → Build, Execution, Deployment → CMake → CMake options）：**

```
-DCMAKE_PREFIX_PATH=/home/liam/python_linux/python_venv/.venv/lib/python3.13/site-packages/torch/share/cmake
-DPython3_ROOT_DIR=/home/liam/python_linux/python_venv/.venv
```

### Clean

**含义：** 删除所有编译产物（`.o` 文件、`.so` 共享库），保留 CMake 配置（`build.ninja`）。下次 Build 时所有源文件强制重新编译。

**命令行：**

```bash
cmake --build cmake-build-release --target clean
```

### Build（增量编译）

**含义：** 只重新编译有变化的源文件，其余复用已有 `.o`，最终链接生成 `softmax_cuda.cpython-3XX-...so`。Ninja 通过文件时间戳 / 内容哈希判断哪些文件需要重编译。

**注意：** CMakeLists.txt 变了但未 Reload，Build 不会感知编译选项的变化。修改了 `SOFTMAX_VARIANT` 等 `-D` 参数后，必须先 Reload 再 Build。

**命令行：**

```bash
cmake --build cmake-build-release --target softmax_cuda -- -j 18
#     │        │                   │       │              │   └─ 并行编译进程数
#     │        │                   │       │              └─ "--" 之后的参数透传给底层构建工具
#     │        │                   │       └─ 只构建名为 softmax_cuda 的 target
#     │        │                   └─ 指定要构建的目标名称
#     │        └─ 构建目录（build.ninja 所在位置）
#     └─ cmake 进入构建模式
```

**`cmake --build` 与 `ninja` / `make` 的关系：**

`cmake --build` 是对底层构建工具的封装，本项目生成器为 Ninja，实际执行：

```bash
ninja -C cmake-build-release softmax_cuda -j 18
```

三种等价写法：

```bash
cd cmake-build-release && ninja softmax_cuda -j 18    # cd 进去再执行
ninja -C cmake-build-release softmax_cuda -j 18        # -C 指定目录，无需 cd
cmake --build cmake-build-release ...                  # cmake 内部处理 cd（生成器无关）
```

### Rebuild

**含义：** 等价于 Clean + Build，先删除所有产物，再完整重新编译。用于解决增量编译状态不一致、头文件依赖混乱等问题。

**命令行：**

```bash
cmake --build cmake-build-release --target clean
cmake --build cmake-build-release --target softmax_cuda -- -j 18
```

### 自动触发说明

CLion 不会自动执行 Reload 或 Build，两者均需手动触发：

| 操作 | 触发方式 |
|------|----------|
| 修改 `CMakeLists.txt` 后 | CLion 顶部弹出提示栏，点击 "Reload CMake Project"（Settings → CMake → "Reload CMake project on changes" 可开启自动 Reload，但默认关闭——因为 Reload 会执行 `execute_process`，如 `nvidia-smi` / `python`，有耗时） |
| 修改 `.cu` / `.cpp` 后 | 保存文件，手动点击 Build 或按快捷键（CLion 不在保存时自动编译，CUDA 单文件编译耗时长） |

### 常见工作流

| 操作 | 步骤 |
|------|------|
| 修改 `CMakeLists.txt` | 手动 Reload → 手动 Build |
| 修改 `SOFTMAX_VARIANT` / `BLOCK_DIM_Y` | 手动 Reload → 手动 Build（宏值变化必须重新配置） |
| 只修改 `kernels.cu` / `interface.cpp` | 手动 Build（增量编译，只重编改动的文件） |
| 编译状态混乱 / 链接报奇怪错误 | 手动 Rebuild |
| 彻底清除重来 | 删除 `cmake-build-release` 目录 → 手动 Reload → 手动 Build |

### 构建产物与 Python 使用

产物路径：

```
cmake-build-release/softmax_cuda.cpython-3XX-x86_64-linux-gnu.so
```

Python 中加载：

```python
import sys
sys.path.insert(0, '/home/liam/cpp_linux/Fast_Softmax/cmake-build-release')
import softmax_cuda
y = softmax_cuda.softmax_cuda(x)
```
