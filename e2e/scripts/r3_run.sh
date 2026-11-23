#!/bin/bash
# 端到端第三轮（10-18 下午）：R2a 秩 → 正态分位数目标两年都明显提升（IC +0.004、IC_IR +0.09、多空 +2.5、压力 +0.04～+0.10，换手 +0.02），
# 以它为新底座叠加：换种子（稳健性）、尾盘段通道、辅助目标 0.75（降换手）、二者组合、14:30 降权、4 轮。对照 R2a 与 S21b。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
cmp() { TAG=$1 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py "${@:2}" 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE; }
pred() { echo "==================== $(date '+%m-%d %T') 预测 2024：$1" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$1 2024 2>&1 | grep -v -i warn >> $LIVE; }
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
[ -f e2e/runs/R2b_listnet005/transformer_model.json ] && pred R2b_listnet005
echo "==================== $(date '+%m-%d %T') R3 开始（第三轮：秩目标为新底座）" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_R3.txt 1
ARMS=$(sed -E 's/.*--run ([^ ]+).*/\1/' e2e/runs/queue_R3.txt | grep -v "^#" | tr "\n" " ")
for r in $ARMS; do [ -f e2e/runs/$r/transformer_model.json ] && pred $r; done
echo "==================== $(date '+%m-%d %T') 两折对比 R3（对照 S21b、R2a）" >> $LIVE
A="base=e2e/runs/S21b_tod seed43=e2e/runs/S21e_tod_s43 R2a_rank=e2e/runs/R2a_rank R2b_listnet005=e2e/runs/R2b_listnet005"; for r in $ARMS; do A="$A $r=e2e/runs/$r"; done
cmp R3 $A
echo "==================== $(date '+%m-%d %T') R3 完成" >> $LIVE
