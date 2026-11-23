#!/bin/bash
# R35（10-24 傍晚，用户："要有实质性的改进"）：1 分钟输入的归一化 rel2 让每个量类字段各自除以自己的窗口均值，
# 买一量与卖一量、买卖委托笔数之间的相对水平（不平衡、每笔挂单规模）在 1 分钟层面看不到。rel3：买卖同类字段共用一个参照。
#   a rel3 + base 字段；b rel3 + 一档委托笔数（ob1）；c rel3 + 三档全字段（full）。种子 42，对照 R22a（E16 配方）。
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
R=e2e/runs
until grep -q "R34 完成" $LIVE; do sleep 60; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
cp e2e/submission/Transformer_modelsave_train.py e2e/submission_next/Transformer_modelsave_train.prev_r35.py
cp e2e/submission_next/Transformer_modelsave_train.py e2e/submission/Transformer_modelsave_train.py
cp e2e/submission_next/train_local.py e2e/scripts/train_local.py
echo "==================== $(date "+%m-%d %T") R35 开始（rel3：买卖同类字段共用参照；ob1／full 字段），换上新版提交脚本" >> $LIVE
bash e2e/scripts/queue.sh $R/queue_R35.txt 1
RUNS="R35a_rel3_base R35b_rel3_ob1 R35c_rel3_full"
for r in $RUNS; do
  [ -f $R/$r/transformer_model.json ] && [ ! -f $R/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py $R/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R35" >> $LIVE
A="R22a=$R/R22a_ep6_mtl05 R24a=$R/R24a_mtl05_s44 R24c=$R/R24c_mtl05_s43"
for r in $RUNS; do [ -f $R/$r/pred_2024.parquet ] && A="$A ${r%%_*}=$R/$r"; done
TAG=R35 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/stress_halves.py $A 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R35 完成" >> $LIVE
