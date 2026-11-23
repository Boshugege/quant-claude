#!/usr/bin/env bash
# 顺序跑分队列：把候选文件一个个提交到评分机（单进程内串行）。
# 用法：bash 因子实验/runqueue.sh <名称>=<文件> [<名称>=<文件> ...]
# 提交走 _submit.sh：在评测机本机用 curl，在其它机器走 ssh。
cd "$(dirname "$0")/.." || exit 1
for item in "$@"; do
  name="${item%%=*}"
  file="${item#*=}"
  echo "=== $(date +%H:%M:%S) 开始 $name ($file) ==="
  bash 因子实验/_submit.sh "$file" -- start=2023-01 end=2024-12 "name=$name" 2>&1 | tail -6 || echo "!! $name 失败"
  echo "=== $(date +%H:%M:%S) 结束 $name ==="
done
echo "队列完成：$*"
