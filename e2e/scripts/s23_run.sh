#!/bin/bash
# S23（只推理）：融合头 + w·日线专家头（日内几乎不变的锚），看能否抵消时段带来的换手。底座 S20b、时段 token 式 S21b、调制式 S22b。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S23 开始（只推理：日线专家头作日内不变锚）" >> $LIVE
for r in S21b_tod S22b_tod_film S20b_ema995; do
    PYTHONWARNINGS=ignore $PY e2e/scripts/head_mix.py e2e/runs/$r 0.1,0.2,0.35 daily 2>&1 | grep -v -i warn | grep -E "相关|done" >> $LIVE
done
A=""
for r in S21b_tod S22b_tod_film S20b_ema995; do
    A="$A $r=e2e/runs/$r"
    for w in 0.1 0.2 0.35; do A="$A ${r}_d$w=e2e/runs/$r/dmix_w$w"; done
done
echo "==================== $(date '+%m-%d %T') 两折对比 S23" >> $LIVE
TAG=S23 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S23 完成" >> $LIVE
