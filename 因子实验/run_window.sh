#!/usr/bin/env bash
# 批量提交指定时间窗口的评分任务（并行 4 个）。用法：
#   bash 因子实验/run_window.sh 2020-01 2020-12 y20_1090=因子代码/1090_xxx.py [y20_1091=... ...]
# 说明：评测机在本机（127.0.0.1:46220）时 _submit.sh 直接走 curl；并行度由评测机按内存自适应。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
START="$1"; END="$2"; shift 2
one() {
  local item="$1" name="${item%%=*}" file="${item#*=}"
  bash 因子实验/_submit.sh "$file" -- start="$START" end="$END" "name=$name" > "因子实验/结果/score_$name.log" 2>&1
  echo "done $name"
}
export -f one; export START END
printf '%s\n' "$@" | xargs -P 4 -I{} bash -c 'one "{}"'
echo ALLDONE
