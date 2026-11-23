#!/bin/bash
# R11（10-19）：IC 稳定性项（批内截面 IC 标准差）对 R7a 的 2023 IC_IR；训完补 2024 预测，两年对比 S21b、R7a。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
bash e2e/scripts/queue.sh e2e/runs/queue_R12.txt 1
for r in R12a_rank_ep4_s44 R12b_R7a_s44; do
  [ -f e2e/runs/$r/pred_2024.parquet ] || { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R12" >> $LIVE
TAG=R12 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py base=e2e/runs/S21b_tod R7a=e2e/runs/R7a_rank_ep4_cls_slotw_aux06 R3f=e2e/runs/R3f_rank_ep4 R8b=e2e/runs/R8b_rank_ep4_s43 R12a=e2e/runs/R12a_rank_ep4_s44 R9a=e2e/runs/R9a_rank_ep4_cls_slotw_aux06_s43 R12b=e2e/runs/R12b_R7a_s44 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R12 完成" >> $LIVE
