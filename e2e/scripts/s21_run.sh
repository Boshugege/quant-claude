#!/bin/bash
# S21：在 A2 + 权重滑动平均（S20b）上重测四个结构改动（时段、同时段 token、截面 Transformer、加宽），fold-A，种子 42。
# 串行训练、预测 2024、与 S20b 及 EMA 双种子集成对比；再做只推理的分析：各臂与 S20b 的预测相关、"S20b + 结构变体"两两集成 vs EMA 两种子集成。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
Q=e2e/runs/queue_S21.txt
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S21 开始（A2 + EMA 上重测结构改动 + 结构多样性集成）" >> $LIVE
bash e2e/scripts/queue.sh "$Q" 1
R=$(grep -v '^#' "$Q" | sed -E 's/.*--run ([^ ]+).*/\1/')
for r in $R; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
done
ENS="ens_ema2=e2e/runs/S20_ens_ema2"
for r in $R; do
    [ -f e2e/runs/$r/pred_2024.parquet ] || continue
    $PY e2e/scripts/ens_pred.py e2e/runs/S21_ens_$r e2e/runs/S20b_ema995 e2e/runs/$r >> $LIVE 2>&1
    ENS="$ENS ens_$r=e2e/runs/S21_ens_$r"
done
ARGS="ema42=e2e/runs/S20b_ema995 ema43=e2e/runs/S20c_ema995_s43"
for r in $R; do ARGS="$ARGS $r=e2e/runs/$r"; done
echo "==================== $(date '+%m-%d %T') 两折对比 S21（单臂）" >> $LIVE
TAG=S21 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ARGS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') 两折对比 S21（集成：S20b + 各臂 vs EMA 两种子）" >> $LIVE
TAG=S21e PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $ENS 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') 与 S20b 的预测相关（EMA 两种子为 0.941 / 0.934）" >> $LIVE
P=""; for r in $R; do [ -f e2e/runs/$r/pred_2024.parquet ] && P="$P S20b_ema995 $r"; done
$PY e2e/scripts/seed_corr.py $P >> $LIVE 2>&1
echo "==================== $(date '+%m-%d %T') S21 完成" >> $LIVE
