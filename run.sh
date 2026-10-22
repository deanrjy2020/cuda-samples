#!/usr/bin/env bash
# 用法: ./run.sh. 依次调用所有 subfolder 中的 run.sh, build 并运行对应的 sample.
set -euo pipefail

# 从脚本所在的仓库根目录查找, 允许从任意目录调用本脚本.
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$root_dir"

# 排除根脚本自身及 Git 元数据; 使用 NUL 分隔, 兼容路径中的空格.
script_list="$(mktemp)"
trap 'rm -f -- "$script_list"' EXIT
find . -type d -name .git -prune -o -type f -name run.sh ! -path ./run.sh -print0 | sort -z > "$script_list"

count=0
failed=()
while IFS= read -r -d '' script; do
    count=$((count + 1))
    printf '\n>>> Running %s\n' "$script"
    # 子脚本使用相对路径构建, 因此必须在各自目录中运行.
    if (cd -- "$(dirname -- "$script")" && bash ./run.sh); then
        printf '>>> PASS: %s\n' "$script"
    else
        status=$?
        printf '>>> FAIL (%d): %s\n' "$status" "$script" >&2
        failed+=("$script")
    fi
done < "$script_list"

# 即使某个示例失败也继续运行其余脚本, 最后统一返回失败状态.
printf '\nCompleted: %d scripts, %d failed.\n' "$count" "${#failed[@]}"
if ((${#failed[@]})); then
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
