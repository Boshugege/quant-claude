#!/bin/bash
# R36（10-25，用户："全力在端到端寻找切实的提分点"）：腿换手损失——评分机的换手只数前 10%／后 10% 腿的成分变化（按风格残差排名），
# 原来的换手惩罚（R31）作用在全部股票上、伤 IC；这里只在腿边界附近惩罚相邻截面"软腿成员"的变化（因子赛道最小抬升的训练时版本）。
#   a 权重 1.0，b 权重 4.0，c 只用相邻截面配对的训练单元（对照）。种子 42，对照 R22a。
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log; R=e2e/runs
while pgrep -f "e2e/scripts/train_local.py|gbdt_member.py|e17_add_gbdt.py" >/dev/null; do sleep 30; done
echo "==================== $(date "+%m-%d %T") R36 开始（腿换手损失）" >> $LIVE
bash e2e/scripts/queue.sh $R/queue_R36.txt 1
RUNS="R36a_leg1 R36b_leg4 R36c_pair0"
for r in $RUNS; do
  [ -f $R/$r/transformer_model.json ] && [ ! -f $R/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py $R/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R36" >> $LIVE
A="R22a=$R/R22a_ep6_mtl05 R24a=$R/R24a_mtl05_s44 R24c=$R/R24c_mtl05_s43"
for r in $RUNS; do [ -f $R/$r/pred_2024.parquet ] && A="$A ${r%%_*}=$R/$r"; done
TAG=R36 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R36 完成" >> $LIVE
