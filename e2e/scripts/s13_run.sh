#!/bin/bash
# S13 精简交付路线：C1（融合头 + 全天 patch + 辅助目标 1.0，fold-A）→ 预测 2024 并与底座三种子对比 → 紧接着全区间重训 F10（E10 候选）。串行。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S13 C1 开始" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_S13_C1.txt 1
echo "==================== $(date '+%m-%d %T') 预测 2024：S13_C1_fuse_patch_aux10" >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/S13_C1_fuse_patch_aux10 2024 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date '+%m-%d %T') 两折对比 S13" >> $LIVE
TAG=S13 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py base42=e2e/runs/S8c_aux05 base43=e2e/runs/S10z_seed43 base44=e2e/runs/S11z_seed44 \
    A2_fuse=e2e/runs/S12_A2_fuse T2_patch=e2e/runs/S11_T2_patch42 C1=e2e/runs/S13_C1_fuse_patch_aux10 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S13 C1 完成；F10 全区间重训开始" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_F10.txt 1
echo "==================== $(date '+%m-%d %T') F10 完成（等打包判断）" >> $LIVE
