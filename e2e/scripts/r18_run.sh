#!/bin/bash
# R18（10-20，排在 R19 之后）：第四个全量种子与第四个抽样成员（种子 45），比四种子全量集成 vs 四成员抽样集成（交付规模）。两年对比。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "R19 完成" $LIVE; do sleep 30; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R18.txt 1
for r in R18a_ep6_s45 R18b_ep6_bag20_s45; do
  [ -f e2e/runs/$r/transformer_model.json ] && [ ! -f e2e/runs/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
d=e2e/runs/ENS_R18_seed4; [ -f $d/pred_2024.parquet ] || $PY e2e/scripts/ens_pred.py $d e2e/runs/R14c_P13_ep6 e2e/runs/R15a_P13_ep6_s43 e2e/runs/R15b_P13_ep6_s44 e2e/runs/R18a_ep6_s45 > /dev/null 2>&1
d=e2e/runs/ENS_R18_bag4; [ -f $d/pred_2024.parquet ] || $PY e2e/scripts/ens_pred.py $d e2e/runs/R16a_ep6_bag20_s42 e2e/runs/R16b_ep6_bag20_s43 e2e/runs/R17a_ep6_bag20_s44 e2e/runs/R18b_ep6_bag20_s45 > /dev/null 2>&1
echo "==================== $(date "+%m-%d %T") 两折对比 R18" >> $LIVE
TAG=R18 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py seed3=e2e/runs/ENS_R17_seed3 bag3=e2e/runs/ENS_R17_bag3 seed4=e2e/runs/ENS_R18_seed4 bag4=e2e/runs/ENS_R18_bag4 R18a=e2e/runs/R18a_ep6_s45 R18b=e2e/runs/R18b_ep6_bag20_s45 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R18 完成" >> $LIVE
