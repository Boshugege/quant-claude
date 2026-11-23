#!/usr/bin/env bash
# 并发跑分：bash 因子实验/par.sh <并发数> <名称>=<文件> [<名称>=<文件> ...]
# 每个任务日志写到 因子实验/结果/score_<名称>.log
# 提交走 _submit.sh：在评测机本机用 curl，在其它机器走 ssh。
P="$1"; shift
cd "$(dirname "$0")/.." || exit 1
mkdir -p 因子实验/结果
one() {
  local item="$1" name="${1%%=*}" file="${1#*=}"
  bash 因子实验/_submit.sh "$file" -- start=2023-01 end=2024-12 "name=$name" \
    > "因子实验/结果/score_$name.log" 2>&1
  tail -20 "因子实验/结果/score_$name.log" | grep -E "^任务 |综合|paired" || tail -3 "因子实验/结果/score_$name.log"
}
export -f one
printf '%s\n' "$@" | xargs -P "$P" -I{} bash -c 'one "{}"'
echo "全部完成：$*"
