#!/bin/bash
# R13（10-19 夜）：在 P13 配方（公榜 0.86209）上每次只改一处，找下一个结构性提升；等 E15 训完再开。训完补 2024 预测，两年对比 S21b、R3f（= P13）。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "E15 训练完成" $LIVE; do sleep 60; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R13.txt 1
RUNS="R13a_P13_auxn3 R13b_P13_daily40 R13c_P13_ema997 R13d_P13_drop15 R13e_P13_ep5"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] || continue
  [ -f e2e/runs/$r/pred_2024.parquet ] || { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R13（P13 底座）" >> $LIVE
A="base=e2e/runs/S21b_tod P13=e2e/runs/R3f_rank_ep4"; for r in $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R13 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R13 完成" >> $LIVE
