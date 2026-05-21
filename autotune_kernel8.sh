#!/usr/bin/env bash

set -u
# 管道退出码取所有段中最坏的退出码（默认是最后一段的退出码）
#   不加 pipefail：cmake ... | tee 的退出码 = tee 的退出码（几乎恒为 0）
#                  if ! cmake ... | tee → ! 0 = 真 → 无论 cmake 是否失败都进 then 块，条件判断失效
#   加 pipefail  ：cmake 失败(非零) | tee(0) → 管道退出码 = cmake 的非零退出码
#                  if ! cmake ... | tee → ! 非零 = 真 → 正确检测到编译失败
set -o pipefail

# 切换到脚本文件所在目录，使所有相对路径以项目根为基准
#
#   $0          脚本自身的调用路径，由调用方式决定：
#                 ./autotune_kernel8.sh              → $0 = ./autotune_kernel8.sh
#                 /home/liam/.../autotune_kernel8.sh → $0 = /home/liam/.../autotune_kernel8.sh
#
#   dirname "$0"  取 $0 的目录部分（脚本文件所在目录），不是执行命令时的当前工作目录：
#                 $0 = ./cpp_linux/Fast_Softmax/autotune_kernel8.sh → dirname = ./cpp_linux/Fast_Softmax
#                 pwd = /home/liam（调用时所在目录，dirname 与此无关）
#
#   cd + pwd    将相对路径转为绝对路径，保证无论从哪个目录调用脚本，
#               后续 KERNEL / OUTPUT / EXEC 等相对路径始终以项目根为基准
cd "$(dirname "$0")"

KERNEL="src/kernels/common.cuh"
OUTPUT="autotune/autotune_results.txt"
EXEC="./validation"

# URF 候选值（循环展开因子）
URF_VALUES=(1 2 4 8)

# 记录原始行，脚本退出时（含中断）恢复源文件
#
#   $( )              命令替换：运行括号内的命令，将其 stdout 捕获为字符串赋给变量
#
#   grep "constexpr int URF {" "$KERNEL"
#     grep            在文件中搜索匹配模式的行，将匹配行输出到 stdout
#     第一个参数      搜索模式（固定字符串），匹配含 "constexpr int URF {" 的行
#     "$KERNEL"       被搜索的文件路径（双引号防止路径中的空格被 shell 拆分为多个参数）
#     输出            整行内容，如：constexpr int URF {UNROLL_FACTOR};
#
#   ORIGINAL_LINE     保存匹配行的完整文本，供 restore() 函数在脚本退出时写回文件，
#                     恢复 sed 修改前的原始内容
ORIGINAL_LINE=$(grep "constexpr int URF {" "$KERNEL")

# restore：将 $KERNEL 中被 sed 修改的 URF 行恢复为原始内容
restore() {
    echo ""
    echo "Restoring: $ORIGINAL_LINE"

    # sed -i "s/pattern/replacement/" file
    #   -i          原地修改文件（in-place），直接覆盖磁盘上的文件
    #   s/pat/rep/  替换命令：将匹配 pat 的内容替换为 rep
    #
    # pattern：constexpr int URF {.*
    #   constexpr int URF {   固定前缀，定位目标行
    #   .*                    匹配行尾的任意内容（含当前被修改的数字）
    #
    # replacement：$(echo "$ORIGINAL_LINE" | sed 's/[\/&]/\\&/g')
    #   外层 $( )             命令替换，在执行 sed -i 之前由 shell 展开为字符串
    #   echo "$ORIGINAL_LINE" 将保存的原始行输出到 stdout
    #   | sed 's/[\/&]/\\&/g' 对原始行做转义处理：
    #     's/[\/&]/\\&/g'     将 / 和 & 替换为 \/ 和 \&
    #     原因：sed 替换命令用 / 作分隔符，若原始行中含 / 会提前截断 replacement；
    #           & 在 sed replacement 中表示"整个匹配串"，需转义为字面 &；
    #           加 \ 前缀后这两个字符变为普通字面量，不再有特殊含义
    #   展开结果示例：
    #     ORIGINAL_LINE = "constexpr int URF {UNROLL_FACTOR};"
    #     转义后        = "constexpr int URF {UNROLL_FACTOR};"（本例无需转义，结果不变）
    sed -i "s/constexpr int URF {.*/$(echo "$ORIGINAL_LINE" | sed 's/[\/&]/\\&/g')/" "$KERNEL"
}

# trap：注册信号处理函数，在 shell 收到指定信号时自动调用
#   trap restore EXIT
#     restore   收到信号时执行的命令（此处为函数名）
#     EXIT      伪信号，shell 在任意退出路径触发：
#                 正常退出（脚本执行完毕）  → 触发 EXIT
#                 Ctrl-C（SIGINT）         → shell 响应中断 → 退出 → 触发 EXIT
#                 set -e 触发的命令失败退出 → 触发 EXIT
#               无论何种退出原因，restore 都会被调用，保证源文件不残留修改后的值
trap restore EXIT

mkdir -p "$(dirname "$OUTPUT")"
# 清空结果文件，避免追加到上次残留数据
echo "" > "$OUTPUT"

TOTAL=${#URF_VALUES[@]}
CONFIG_NUM=0

for urf in "${URF_VALUES[@]}"; do
    CONFIG_NUM=$(( CONFIG_NUM + 1 ))
    echo ""
    echo "($CONFIG_NUM/$TOTAL): URF=$urf"

    # ── sed 原地修改 URF 初始化值 ────────────────────────────────────────────
    # 匹配 "constexpr int URF {任意内容};"，整行替换为新值
    # cmake 的 -DUNROLL_FACTOR 宏通过 #ifndef 保护注入，与此 constexpr 变量独立：
    #   cmake 注入 -DUNROLL_FACTOR=X → 宏展开为数字 → 传给 constexpr URF
    #   sed 直接替换 constexpr 行，绕过宏，使 URF 持有字面值
    sed -i "s/constexpr int URF {.*/constexpr int URF {$urf};/" "$KERNEL"

    # ── 重新编译 validation target ───────────────────────────────────────────
    # cmake --build 读取 build.ninja（配置阶段已生成），无需重新 cmake configure
    # 只重编译 validation 及其依赖（不重编 softmax_cuda .so），节省时间
    # 编译失败（如 URF 超出寄存器限制）属预期内，跳过该值继续搜索
    if ! cmake --build cmake-build-release --target validation -- -j "$(nproc)" 2>&1 | tee -a "$OUTPUT"; then
        echo "COMPILE FAILED: URF=$urf" | tee -a "$OUTPUT"
        echo "-------------------" | tee -a "$OUTPUT"
        continue
    fi

    echo "URF=$urf" | tee -a "$OUTPUT"

    # ── 运行 validation，只测 kernel 8 ──────────────────────────────────────
    # validation 接受 kernel 编号作为 argv[1]
    # 2>&1 合并 stderr（divergence 报错）进管道，tee 同时写终端和结果文件
    # timeout 防止极慢配置（寄存器溢出）挂死脚本
    if ! timeout 60 "$EXEC" 8 2>&1 | tee -a "$OUTPUT"; then
        echo "RUNTIME FAILED or TIMEOUT: URF=$urf" | tee -a "$OUTPUT"
    fi

    echo "-------------------" | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
done

echo "Done. Results in $OUTPUT"
