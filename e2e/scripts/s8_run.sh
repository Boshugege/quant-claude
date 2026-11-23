#!/bin/bash
# S8：多方向探索（底座 = H4 + 盘口窗口通道 = S7b）。第一段不需要新代码；第二段（queue_S8_2.txt）等代码写好后再放上来。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
bash e2e/scripts/queue.sh e2e/runs/queue_S8_1.txt 1
while [ ! -f e2e/runs/queue_S8_2.txt ]; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_S8_2.txt 1
R=$(cat e2e/runs/queue_S8_1.txt e2e/runs/queue_S8_2.txt | grep -v '^#' | sed -E 's/.*--run ([^ ]+).*/\1/')
for r in $R; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
done
echo "==================== $(date '+%m-%d %T') 两折对比" >> $LIVE
ARGS="H4=e2e/runs/H4_winfeat7 S7b_base=e2e/runs/S7b_book"; for r in $R; do ARGS="$ARGS $r=e2e/runs/$r"; done
TAG=S8 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ARGS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S8 完成" >> $LIVE
