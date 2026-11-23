#!/bin/bash
# S17：融合头加强（E12 候选）。先做推理期头部混合（不训练），再串行训练 S17a/b/c 与 A2 第二个种子，最后两折对比。
# 用法（远端 quant 目录）：nohup bash e2e/scripts/s17_run.sh > /dev/null 2>&1 &
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
Q=e2e/runs/queue_S17.txt
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S17 开始：头部混合（只推理）" >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/head_mix.py e2e/runs/S12_A2_fuse 0.25,0.5 2>&1 | grep -v -i warn >> $LIVE
TAG=S17_mix PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py A2=e2e/runs/S12_A2_fuse mix025=e2e/runs/S12_A2_fuse/mix_w0.25 mix05=e2e/runs/S12_A2_fuse/mix_w0.5 base42=e2e/runs/S8c_aux05 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
bash e2e/scripts/queue.sh "$Q" 1
R=$(grep -v '^#' "$Q" | sed -E 's/.*--run ([^ ]+).*/\1/')
for r in $R; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
done
$PY e2e/scripts/ens_pred.py e2e/runs/S17_ens_A2x2 e2e/runs/S12_A2_fuse e2e/runs/S17z_A2_seed43 >> $LIVE 2>&1
ARGS="A2=e2e/runs/S12_A2_fuse mix025=e2e/runs/S12_A2_fuse/mix_w0.25 mix05=e2e/runs/S12_A2_fuse/mix_w0.5 ens2=e2e/runs/S17_ens_A2x2"
for r in $R; do ARGS="$ARGS $r=e2e/runs/$r"; done
echo "==================== $(date '+%m-%d %T') 两折对比 S17" >> $LIVE
TAG=S17 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ARGS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S17 完成" >> $LIVE
