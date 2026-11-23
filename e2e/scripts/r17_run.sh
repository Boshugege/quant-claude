#!/bin/bash
# R17（10-20）：第三个抽样成员（种子 44），比三成员抽样集成 vs 三种子全量集成（R14c、R15a、R15b）。两年对比。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "R16 完成" $LIVE; do sleep 30; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R17.txt 1
r=R17a_ep6_bag20_s44
[ -f e2e/runs/$r/transformer_model.json ] && [ ! -f e2e/runs/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
d=e2e/runs/ENS_R17_bag3; [ -f $d/pred_2024.parquet ] || $PY e2e/scripts/ens_pred.py $d e2e/runs/R16a_ep6_bag20_s42 e2e/runs/R16b_ep6_bag20_s43 e2e/runs/$r > /dev/null 2>&1
d=e2e/runs/ENS_R17_seed3; [ -f $d/pred_2024.parquet ] || $PY e2e/scripts/ens_pred.py $d e2e/runs/R14c_P13_ep6 e2e/runs/R15a_P13_ep6_s43 e2e/runs/R15b_P13_ep6_s44 > /dev/null 2>&1
echo "==================== $(date "+%m-%d %T") 两折对比 R17" >> $LIVE
TAG=R17 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py seed2=e2e/runs/ENS_R16_seed2 bag2=e2e/runs/ENS_R16_bag2 seed3=e2e/runs/ENS_R17_seed3 bag3=e2e/runs/ENS_R17_bag3 R17a=e2e/runs/$r 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/seed_corr.py R15b_P13_ep6_s44 R15a_P13_ep6_s43 $r R16b_ep6_bag20_s43 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R17 完成" >> $LIVE
