#!/bin/bash
# S19：训练时加入中间时刻截面（fold-A，种子 42）。等 F14/F15 结束后串行跑，预测 2024，与 A2（P08 两个种子）对比。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
Q=e2e/runs/queue_S19.txt
while pgrep -f "e2e/scripts/train_local.py" >/dev/null || pgrep -f "queue_F1415" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S19 开始（训练加中间时刻截面）" >> $LIVE
bash e2e/scripts/queue.sh "$Q" 1
R=$(grep -v '^#' "$Q" | sed -E 's/.*--run ([^ ]+).*/\1/')
for r in $R; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
done
ARGS="A2s42=e2e/runs/S12_A2_fuse A2s43=e2e/runs/S17z_A2_seed43"
for r in $R; do ARGS="$ARGS $r=e2e/runs/$r"; done
echo "==================== $(date '+%m-%d %T') 两折对比 S19" >> $LIVE
TAG=S19 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ARGS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S19 完成" >> $LIVE
