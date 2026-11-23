#!/bin/bash
# 端到端第四／五轮（10-18 晚）：R4c，然后 R5d 两阶段残差提升（第一阶段 = R2a），R5a／R5b 加宽（96／128），R5c EMA 0.998，R5e 每批 8 个截面。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
cmp() { TAG=$1 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py "${@:2}" 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE; }
pred() { [ -f e2e/runs/$1/pred_2024.parquet ] && return; echo "==================== $(date '+%m-%d %T') 预测 2024：$1" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$1 2024 2>&1 | grep -v -i warn >> $LIVE; }
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
pred R3f_rank_ep4 & pred R4b_rank_aux06_slotw & wait
bash e2e/scripts/queue.sh e2e/runs/queue_R5.txt 1
ARMS="R3f_rank_ep4 R4b_rank_aux06_slotw $(sed -E 's/.*--run ([^ ]+).*/\1/' e2e/runs/queue_R5.txt | tr '\n' ' ')"
for r in $ARMS; do [ -f e2e/runs/$r/transformer_model.json ] && pred $r; done
echo "==================== $(date '+%m-%d %T') 两折对比 R4／R5（对照 S21b、R2a）" >> $LIVE
A="base=e2e/runs/S21b_tod R2a_rank=e2e/runs/R2a_rank"; for r in $ARMS; do A="$A $r=e2e/runs/$r"; done
cmp R5 $A
echo "==================== $(date '+%m-%d %T') R5 完成" >> $LIVE
