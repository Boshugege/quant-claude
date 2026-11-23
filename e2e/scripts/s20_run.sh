#!/bin/bash
# S20：降低单模型的种子噪声（权重滑动平均、余弦学习率），fold-A。串行训练、预测 2024、与 A2 两个种子及其集成对比，并量两个种子的预测相关。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
Q=e2e/runs/queue_S20.txt
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S20 开始（降低种子噪声：权重滑动平均 / 余弦学习率）" >> $LIVE
bash e2e/scripts/queue.sh "$Q" 1
R=$(grep -v '^#' "$Q" | sed -E 's/.*--run ([^ ]+).*/\1/')
for r in $R; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
done
ARGS="A2s42=e2e/runs/S12_A2_fuse A2s43=e2e/runs/S17z_A2_seed43 ens2=e2e/runs/S17_ens_A2x2"
for r in $R; do ARGS="$ARGS $r=e2e/runs/$r"; done
echo "==================== $(date '+%m-%d %T') 两折对比 S20" >> $LIVE
TAG=S20 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ARGS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') 种子相关（基线 0.878 / 0.871）" >> $LIVE
$PY e2e/scripts/seed_corr.py S12_A2_fuse S17z_A2_seed43 S20b_ema995 S20c_ema995_s43 >> $LIVE 2>&1
echo "==================== $(date '+%m-%d %T') S20 完成" >> $LIVE
