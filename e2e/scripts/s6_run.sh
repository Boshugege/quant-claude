#!/bin/bash
# S6：H4 / S5a 的 epoch 用法对比（ep2、ep3、ep2+3 权重平均、ep2+3 预测平均），只推理。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
echo "==================== $(date '+%m-%d %T') S6 开始：epoch 断点推理" >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/s6_epoch.py e2e/runs/H4_winfeat7 e2e/runs/S5a_wf7_icir 2>&1 | grep -v -i warn >> $LIVE
ARGS=""
for r in H4_winfeat7 S5a_wf7_icir; do
    ARGS="$ARGS ${r}_ep3=e2e/runs/$r"
    for v in ep2 swa23 ens23; do ARGS="$ARGS ${r}_$v=e2e/runs/S6/${r}_$v"; done
done
echo "==================== $(date '+%m-%d %T') 两折对比" >> $LIVE
TAG=S6 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ARGS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S6 完成" >> $LIVE
