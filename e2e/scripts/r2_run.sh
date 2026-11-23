#!/bin/bash
# 端到端第二轮（10-18）：等第一轮（r1_run.sh）收尾后开始。R2a 秩 → 正态分位数目标（与评测秩 IC 同口径）、R2b 列表式损失小权重 0.05；
# 叠加臂按第一轮结果追加到 e2e/runs/queue_R2.txt（queue.sh 逐行读取，运行中追加有效）。对照 S21b 与噪声带 S21e。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
cmp() { TAG=$1 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py "${@:2}" 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE; }
pred() { echo "==================== $(date '+%m-%d %T') 预测 2024：$1" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$1 2024 2>&1 | grep -v -i warn >> $LIVE; }
while pgrep -f "r1_run.sh" >/dev/null || pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') R2 开始（第二轮）" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_R2.txt 1
ARMS=$(sed -E 's/.*--run ([^ ]+).*/\1/' e2e/runs/queue_R2.txt | grep -v "^#" | tr "\n" " ")
for r in $ARMS; do [ -f e2e/runs/$r/transformer_model.json ] && pred $r; done
echo "==================== $(date '+%m-%d %T') 两折对比 R2（对照 S21b 与噪声带 S21e）" >> $LIVE
A="base=e2e/runs/S21b_tod seed43=e2e/runs/S21e_tod_s43"; for r in $ARMS; do A="$A $r=e2e/runs/$r"; done
cmp R2 $A
echo "==================== $(date '+%m-%d %T') R2 完成" >> $LIVE
