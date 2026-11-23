#!/bin/bash
# S9：辅助目标挖深（底座 = S8c：H4 + 盘口 + 辅助目标 0.5、N=2，折 A）。等 F09（E09）训练结束后再开始，串行。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
while pgrep -f "queue_F09.txt" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S9 开始" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_S9.txt 1
R=$(grep -v '^#' e2e/runs/queue_S9.txt | sed -E 's/.*--run ([^ ]+).*/\1/')
for r in $R; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
done
echo "==================== $(date '+%m-%d %T') 两折对比" >> $LIVE
ARGS="S7b_base=e2e/runs/S7b_book S8c_aux05=e2e/runs/S8c_aux05"; for r in $R; do ARGS="$ARGS $r=e2e/runs/$r"; done
TAG=S9 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ARGS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S9 完成（S10 不自动开始，等用户看完因子赛道结果再定）" >> $LIVE
