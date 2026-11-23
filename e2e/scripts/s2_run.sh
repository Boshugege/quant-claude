#!/bin/bash
# E07 S2：三组训练（顺序执行）→ 各自预测 2024 → 平台口径两折对比（含 E06A 参照）。输出都追加到 e2e/runs/live.log。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
bash e2e/scripts/queue.sh e2e/runs/queue_E07_S2.txt 1
for r in G1_e07_none G2_e07_proxy G3_e07_proxyind; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
done
echo "==================== $(date '+%m-%d %T') 两折对比" >> $LIVE
TAG=S2 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py E06A_official=e2e/runs/E06A_experts_peer_barra_yoff G1_none=e2e/runs/G1_e07_none \
    G2_proxy=e2e/runs/G2_e07_proxy G3_proxyind=e2e/runs/G3_e07_proxyind 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S2 完成" >> $LIVE
