#!/bin/bash
# 远端顺序任务队列：依次运行 jobs 文件中的每行 train_local 参数，同时只跑 MAX 个（默认 1）。
# 所有任务的输出追加到 e2e/runs/live.log（在本机 Terminal 用 tail -F 实时查看），每个任务另有 runs/<name>/log.txt。
# 用法（远端 quant 目录）：nohup bash e2e/scripts/queue.sh e2e/runs/queue.txt 1 > /dev/null 2>&1 &
JOBS="$1"; MAX="${2:-1}"
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
running() { pgrep -f "^$PY e2e/scripts/train_local.py" | wc -l | tr -d ' '; }
while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue;; esac
    while [ "$(running)" -ge "$MAX" ]; do sleep 20; done
    name=$(echo "$line" | sed -E 's/.*--run ([^ ]+).*/\1/')
    echo "" >> $LIVE
    echo "==================== $(date '+%m-%d %T') 开始 $name : $line" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/train_local.py $line 2>&1 | grep --line-buffered -v -E "Warning|warn\(|^ +" >> $LIVE
    echo "==================== $(date '+%m-%d %T') 结束 $name (exit ${PIPESTATUS[0]})" >> $LIVE
done < "$JOBS"
echo "==================== $(date '+%m-%d %T') 队列完成" >> $LIVE
