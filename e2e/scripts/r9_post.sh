#!/bin/bash
# 第九／十轮后处理（10-19）：等 R10a 训完，串行补 2024 预测，再与 S21b、R7a 两年对比。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "结束 R10a" $LIVE; do sleep 60; done
for r in R9a_rank_ep4_cls_slotw_aux06_s43 R10a_rank_ep5_cls_slotw_aux06; do
  [ -f e2e/runs/$r/pred_2024.parquet ] || { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R9／R10" >> $LIVE
TAG=R10 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py base=e2e/runs/S21b_tod R7a=e2e/runs/R7a_rank_ep4_cls_slotw_aux06 R9a=e2e/runs/R9a_rank_ep4_cls_slotw_aux06_s43 R10a=e2e/runs/R10a_rank_ep5_cls_slotw_aux06 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R10 完成" >> $LIVE
