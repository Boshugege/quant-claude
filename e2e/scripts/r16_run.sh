#!/bin/bash
# R16（10-20）：按交易日抽样集成（每个成员去掉 20% 交易日，种子 42／43）对比两种子全量集成（R14c + R15a）；每批 8 个截面。两年对比。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "R15 完成" $LIVE; do sleep 30; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R16.txt 1
RUNS="R16a_ep6_bag20_s42 R16b_ep6_bag20_s43 R16c_ep6_spb8"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] || continue
  [ -f e2e/runs/$r/pred_2024.parquet ] || { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
d=e2e/runs/ENS_R16_bag2; [ -f $d/pred_2024.parquet ] || $PY e2e/scripts/ens_pred.py $d e2e/runs/R16a_ep6_bag20_s42 e2e/runs/R16b_ep6_bag20_s43 > /dev/null 2>&1
d=e2e/runs/ENS_R16_seed2; [ -f $d/pred_2024.parquet ] || $PY e2e/scripts/ens_pred.py $d e2e/runs/R14c_P13_ep6 e2e/runs/R15a_P13_ep6_s43 > /dev/null 2>&1
echo "==================== $(date "+%m-%d %T") 两折对比 R16" >> $LIVE
A="ep6_s42=e2e/runs/R14c_P13_ep6 ep6_s43=e2e/runs/R15a_P13_ep6_s43 ep6_s44=e2e/runs/R15b_P13_ep6_s44 seed2=e2e/runs/ENS_R16_seed2 bag2=e2e/runs/ENS_R16_bag2"; for r in $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R16 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/seed_corr.py R14c_P13_ep6 R15a_P13_ep6_s43 R16a_ep6_bag20_s42 R16b_ep6_bag20_s43 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R16 完成" >> $LIVE
