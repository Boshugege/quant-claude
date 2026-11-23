#!/bin/bash
# S15：在 E10 配方（C1）上各加一样（时段 / 同时段 token / 辅助 1.5）。用法：bash e2e/scripts/s11_run.sh <队列文件> <标签>；串行，跑完预测 2024 并与底座各种子对比，然后停下。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
Q="$1"; TAG="$2"
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S11 $TAG 开始" >> $LIVE
bash e2e/scripts/queue.sh "$Q" 1
R=$(grep -v '^#' "$Q" | sed -E 's/.*--run ([^ ]+).*/\1/')
for r in $R; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
done
ARGS="base42=e2e/runs/S8c_aux05 base43=e2e/runs/S10z_seed43 base44=e2e/runs/S11z_seed44 C1=e2e/runs/S13_C1_fuse_patch_aux10"
for r in $R; do ARGS="$ARGS $r=e2e/runs/$r"; done
echo "==================== $(date '+%m-%d %T') 两折对比 $TAG" >> $LIVE
TAG=S11_$TAG PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ARGS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S11 $TAG 完成" >> $LIVE
