#!/bin/bash
# S18：近期数据加权（fold-B：训练 2020–23 或 2021–23，验证 2024）。等前一个训练结束后串行跑四组，最后只比 2024 年。
# 用法（远端 quant 目录）：nohup bash e2e/scripts/s18_run.sh > /dev/null 2>&1 &
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
Q=e2e/runs/queue_S18.txt
while pgrep -f "e2e/scripts/train_local.py" >/dev/null || pgrep -f "e2e/scripts/queue.sh e2e/runs/queue_F13" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S18 开始（近期数据加权，fold-B 验证 2024）" >> $LIVE
bash e2e/scripts/queue.sh "$Q" 1
R=$(grep -v '^#' "$Q" | sed -E 's/.*--run ([^ ]+).*/\1/')
ARGS=""
for r in $R; do ARGS="$ARGS $r=e2e/runs/$r"; done
echo "==================== $(date '+%m-%d %T') 对比 S18（2024）" >> $LIVE
PAIRS=2024:val_pred.parquet TAG=S18 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ARGS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S18 完成" >> $LIVE
