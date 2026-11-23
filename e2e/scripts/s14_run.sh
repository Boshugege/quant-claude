#!/bin/bash
# S14：C2 = C1 + 同伴走势通道（fold-A）→ 预测 2024 并对比 → 紧接着全区间重训 F11（E11 候选）。串行。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S14 C2 开始" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_S14_C2.txt 1
echo "==================== $(date '+%m-%d %T') 预测 2024：S14_C2_C1_peer" >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/S14_C2_C1_peer 2024 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date '+%m-%d %T') 两折对比 S14" >> $LIVE
TAG=S14 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py base42=e2e/runs/S8c_aux05 base43=e2e/runs/S10z_seed43 base44=e2e/runs/S11z_seed44 \
    C1=e2e/runs/S13_C1_fuse_patch_aux10 C2=e2e/runs/S14_C2_C1_peer 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S14 C2 完成；F11 全区间重训开始" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_F11.txt 1
echo "==================== $(date '+%m-%d %T') F11 完成（等打包判断）" >> $LIVE
