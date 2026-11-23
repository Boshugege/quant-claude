#!/bin/bash
# R11（10-19）：IC 稳定性项（批内截面 IC 标准差）对 R7a 的 2023 IC_IR；训完补 2024 预测，两年对比 S21b、R7a。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
bash e2e/scripts/queue.sh e2e/runs/queue_R11.txt 1
for r in R11a_R7a_icir05 R11b_R7a_icir10; do
  [ -f e2e/runs/$r/pred_2024.parquet ] || { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R11" >> $LIVE
TAG=R11 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py base=e2e/runs/S21b_tod R7a=e2e/runs/R7a_rank_ep4_cls_slotw_aux06 R11a=e2e/runs/R11a_R7a_icir05 R11b=e2e/runs/R11b_R7a_icir10 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R11 完成" >> $LIVE
