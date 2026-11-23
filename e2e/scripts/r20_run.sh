#!/bin/bash
# R20（10-20，用户："去探索真的让网络训练变强大的结构"）：专家表示的池化与跨尺度交互。6 轮底座、种子 42，两年对比（含高、低离散半）。
#   a 注意力池化（可学习查询，起点等价于平均）；b 平均 + 最近一个 token；c 三个尺度之间的跨尺度注意力
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "R17 完成" $LIVE; do sleep 30; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R20.txt 1
RUNS="R20a_ep6_poolattn R20b_ep6_poolmix R20c_ep6_xscale"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] && [ ! -f e2e/runs/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R20" >> $LIVE
A="ep6_s42=e2e/runs/R14c_P13_ep6 ep6_s43=e2e/runs/R15a_P13_ep6_s43 ep6_s44=e2e/runs/R15b_P13_ep6_s44"; for r in $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R20 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/stress_halves.py $A 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R20 完成" >> $LIVE
