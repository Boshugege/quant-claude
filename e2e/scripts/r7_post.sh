#!/bin/bash
# 第七轮后处理（10-18 夜）：等 R6 队列（含 R7a）全部完成、2024 预测齐了，再算近门槛臂的两两配对集成（只推理），对照底座两种子集成 S21_ens_tod2。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "R6 完成" $LIVE; do sleep 60; done
RUNS="R3f_rank_ep4 R6a_rank_ep4_slotw R6b_rank_ep4_cls_slotw R6c_rank_ep4_aux06_slotw R7a_rank_ep4_cls_slotw_aux06"
A="tod2=e2e/runs/S21_ens_tod2"
set -- $RUNS
for a in $RUNS; do
  shift
  for b in "$@"; do
    [ -f e2e/runs/$a/pred_2024.parquet ] && [ -f e2e/runs/$b/pred_2024.parquet ] || continue
    d=e2e/runs/PAIR_${a%%_*}_${b%%_*}
    [ -f $d/pred_2024.parquet ] || $PY e2e/scripts/ens_pred.py $d e2e/runs/$a e2e/runs/$b > /dev/null 2>&1
    A="$A ${a%%_*}+${b%%_*}=$d"
  done
done
echo "==================== $(date "+%m-%d %T") 第七轮配对集成（对照 S21_ens_tod2）" >> $LIVE
TAG=PAIR7 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") 第七轮完成" >> $LIVE
