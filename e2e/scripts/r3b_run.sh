#!/bin/bash
# 端到端第三轮（重排后）：R3a、R3b、R3c（已在跑）之后，按优先级跑 R3e（cls + 辅助 0.75）、R3g（cls + 14:30 降权）、R3h（三者）、R3d、R3f。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
cmp() { TAG=$1 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py "${@:2}" 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE; }
pred() { [ -f e2e/runs/$1/pred_2024.parquet ] && return; echo "==================== $(date '+%m-%d %T') 预测 2024：$1" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$1 2024 2>&1 | grep -v -i warn >> $LIVE; }
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
pred R3c_rank_aux075 &
bash e2e/scripts/queue.sh e2e/runs/queue_R3b.txt 1
wait
ARMS="R3a_rank_s43 R3b_rank_cls R3c_rank_aux075 $(sed -E 's/.*--run ([^ ]+).*/\1/' e2e/runs/queue_R3b.txt | tr '\n' ' ')"
for r in $ARMS; do [ -f e2e/runs/$r/transformer_model.json ] && pred $r; done
echo "==================== $(date '+%m-%d %T') 两折对比 R3（对照 S21b、R2a）" >> $LIVE
A="base=e2e/runs/S21b_tod seed43=e2e/runs/S21e_tod_s43 R2a_rank=e2e/runs/R2a_rank"; for r in $ARMS; do A="$A $r=e2e/runs/$r"; done
cmp R3 $A
echo "==================== $(date '+%m-%d %T') R3 完成" >> $LIVE
