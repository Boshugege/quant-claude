#!/bin/bash
# 今天的秩目标系列两两配对集成（只推理），对照底座两种子集成（S21_ens_tod2），平台口径两年五项写进 live.log 与 analysis/cmp_PAIR.csv。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
RUNS="R2a_rank R3a_rank_s43 R3b_rank_cls R3c_rank_aux075 R3d_rank_slotw R3e_rank_cls_aux075 R3g_rank_cls_slotw R3h_rank_cls_aux075_slotw R3f_rank_ep4"
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
TAG=PAIR PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " > e2e/runs/analysis/pair_search.txt
echo done >> e2e/runs/analysis/pair_search.txt
