#!/bin/bash
# R26（10-21 傍晚，用户："结构简化消融先不做，继续考虑如何提升分数"）：多任务监督扩展。在 R22a（6 轮 + 多任务头 0.5，目标 rv,lv）上加目标：
#   a 之后 60 分钟收益  c 之后 30 分钟最高／最低收盘  e 七个全上  b 之后 10 分钟收益  d 签名成交量  f 只用收益类（r60, r10）
#   对照：R22a／R24c／R24a（多任务头三种子）与 6 轮底座三种子。每组两年对比（含高、低离散半）。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R26.txt 1
RUNS="R26a_mtl_r60 R26c_mtl_hilo R26e_mtl_all7 R26b_mtl_r10 R26d_mtl_sv R26f_mtl_ret"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] && [ ! -f e2e/runs/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R26" >> $LIVE
A="ep6_s42=e2e/runs/R14c_P13_ep6 ep6_s43=e2e/runs/R15a_P13_ep6_s43 ep6_s44=e2e/runs/R15b_P13_ep6_s44 R22a=e2e/runs/R22a_ep6_mtl05 R24c=e2e/runs/R24c_mtl05_s43 R24a=e2e/runs/R24a_mtl05_s44"
for r in $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R26 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/stress_halves.py $A 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R26 完成" >> $LIVE
