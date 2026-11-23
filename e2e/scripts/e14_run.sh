#!/bin/bash
# E14：E13 配方 + 权重滑动平均 0.995 + 当前时段 token，四个成员（种子 42–45），2020–2024 全量，串行训练。
LIVE=e2e/runs/live.log
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') E14 训练开始（四成员串行）" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_E14.txt 1
echo "==================== $(date '+%m-%d %T') E14 训练完成" >> $LIVE
