#!/bin/bash
# E08 前的 S5：以 H4（窗口扩展 7 通道）为底，测 IC 稳定性项叠加、09:30 是否参与训练、两档小换手惩罚。fold-A 训练 → 预测 2024 → 两折对比（H4 为基准）。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
bash e2e/scripts/queue.sh e2e/runs/queue_S5.txt 1
R="S5a_wf7_icir S5b_wf7_keep930 S5c_wf7_icir_turn05 S5d_wf7_icir_turn02"
for r in $R; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
done
echo "==================== $(date '+%m-%d %T') 两折对比" >> $LIVE
ARGS="G3_base=e2e/runs/G3_e07_proxyind H1_skip930=e2e/runs/H1_skip930 H4_winfeat7=e2e/runs/H4_winfeat7"; for r in $R; do ARGS="$ARGS $r=e2e/runs/$r"; done
TAG=S5 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ARGS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S5 完成" >> $LIVE
