#!/bin/bash
# E07 全量重训（F07）→ S3/S4 fold-A 六组 → 各自预测 2024 → 平台口径两折对比（G3 为基准）。输出追加到 e2e/runs/live.log。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
bash e2e/scripts/queue.sh e2e/runs/queue_E07_F_S34.txt 1
R="H1_skip930 H2a_daybatch H2_turn H3_icir H4_winfeat7 H5_peer240"
for r in $R; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
done
echo "==================== $(date '+%m-%d %T') 两折对比" >> $LIVE
ARGS="G3_base=e2e/runs/G3_e07_proxyind"; for r in $R; do ARGS="$ARGS $r=e2e/runs/$r"; done
TAG=S34 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ARGS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S34 完成" >> $LIVE
