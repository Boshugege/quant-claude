#!/bin/bash
# S22：时段只进融合头（拼输入 / 逐通道缩放平移），目标是保住 S21b（时段进每个 token）的 IC 增益、去掉它带来的 +0.02 换手。fold-A，种子 42，EMA。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
Q=e2e/runs/queue_S22.txt
cmp() { TAG=$1 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py "${@:2}" 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE; }
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S22 开始（时段只进融合头：fuse / film）" >> $LIVE
bash e2e/scripts/queue.sh "$Q" 1
for r in S22a_tod_fuse S22b_tod_film; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
    $PY e2e/scripts/ens_pred.py e2e/runs/S22_ens_$r e2e/runs/S20b_ema995 e2e/runs/$r >> $LIVE 2>&1
done
echo "==================== $(date '+%m-%d %T') 两折对比 S22（单臂）" >> $LIVE
cmp S22 ema42=e2e/runs/S20b_ema995 tod_token=e2e/runs/S21b_tod S22a_fuse=e2e/runs/S22a_tod_fuse S22b_film=e2e/runs/S22b_tod_film
echo "==================== $(date '+%m-%d %T') 两折对比 S22（集成：S20b + 各臂）" >> $LIVE
cmp S22e ens_ema2=e2e/runs/S20_ens_ema2 mix_token=e2e/runs/S21_ens_S21b_tod mix_fuse=e2e/runs/S22_ens_S22a_tod_fuse mix_film=e2e/runs/S22_ens_S22b_tod_film
echo "==================== $(date '+%m-%d %T') 预测相关（与 S20b、与 S21b）" >> $LIVE
$PY e2e/scripts/seed_corr.py S20b_ema995 S22a_tod_fuse S20b_ema995 S22b_tod_film S21b_tod S22a_tod_fuse S21b_tod S22b_tod_film >> $LIVE 2>&1
echo "==================== $(date '+%m-%d %T') S22 完成" >> $LIVE
