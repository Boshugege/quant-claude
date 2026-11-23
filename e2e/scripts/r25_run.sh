#!/bin/bash
# R25（10-21，用户："看看模型结构能不能优化，简化的同时提高性能和分数"）：在 R22a（6 轮 + 多任务头 0.5，种子 42）上逐项做减法。
#   a 三个专家头不计损失（只训练融合头）  b 同伴注意力 → 减去截面平均表示  c 不做同伴对比  d 去掉日线专家  e 去掉日内专家
#   f 1 分钟专家 1 层  g 宽度 64 → 48  h 去掉辅助目标（之后 2 个截面）  i 去掉 shape 通道  j 1 分钟回看 64 → 32
#   对照：R22a 与 R24 的多任务头种子 43／44、6 轮底座三种子。每组两年对比（含高、低离散半）。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "R24 完成" $LIVE; do sleep 30; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R25.txt 1
RUNS="R25a_nohead R25b_peermean R25d_nodaily R25e_nointra R25f_short1 R25h_noaux R25g_d48 R25c_peernone R25i_noshape R25j_seq32"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] && [ ! -f e2e/runs/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R25" >> $LIVE
A="ep6_s42=e2e/runs/R14c_P13_ep6 ep6_s43=e2e/runs/R15a_P13_ep6_s43 ep6_s44=e2e/runs/R15b_P13_ep6_s44 R22a=e2e/runs/R22a_ep6_mtl05"
for r in R24a_mtl05_s44 R24c_mtl05_s43 $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R25 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/stress_halves.py $A 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R25 完成" >> $LIVE
