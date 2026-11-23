#!/bin/bash
# S7：H4 底座上加新信息（盘口窗口通道 / 股票状态 / 市场状态 / 1 分钟三档全字段）。fold-A 训练 → 预测 2024 → 两折对比（H4 为基准）。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
bash e2e/scripts/queue.sh e2e/runs/queue_S7.txt 1
R="S7b_book S7c_state S7d_state_mkt S7a_full"
for r in $R; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
done
echo "==================== $(date '+%m-%d %T') 两折对比" >> $LIVE
ARGS="H4_winfeat7=e2e/runs/H4_winfeat7"; for r in $R; do ARGS="$ARGS $r=e2e/runs/$r"; done
TAG=S7 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ARGS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S7 完成" >> $LIVE
