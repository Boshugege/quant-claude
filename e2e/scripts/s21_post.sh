#!/bin/bash
# S21 后半段（10-17 14:30 按 E13 公榜小分调整，取代 s21_run.sh 的收尾）：
#   取消 S21a 加宽（最慢、此前为负、加大方差与公榜偏好相反）；S21c 同时段 token 2023 五项全差，不再预测 2024。
#   第一段：S21d 训完后立即对 S21b（时段）、S21d（截面 Transformer）预测 2024，出单臂对比、"S20b + 各臂"集成对比、与 S20b 的预测相关。
#   第二段：加训 S21e（时段 + EMA，种子 43），比较四种集成：底座对（S20b+S20c）、时段对（S21b+S21e）、混合对（S20b+S21b）、四个全上。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
cmp() { TAG=$1 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py "${@:2}" 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE; }
pred() { echo "==================== $(date '+%m-%d %T') 预测 2024：$1" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$1 2024 2>&1 | grep -v -i warn >> $LIVE; }
while pgrep -f "queue.sh e2e/runs/queue_S21.txt" >/dev/null || pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S21 分析开始（调整后：S21a 取消；S21c 2023 五项全差，不再预测）" >> $LIVE
for r in S21b_tod S21d_xs2; do pred $r; $PY e2e/scripts/ens_pred.py e2e/runs/S21_ens_$r e2e/runs/S20b_ema995 e2e/runs/$r >> $LIVE 2>&1; done
echo "==================== $(date '+%m-%d %T') 两折对比 S21（单臂）" >> $LIVE
cmp S21 ema42=e2e/runs/S20b_ema995 ema43=e2e/runs/S20c_ema995_s43 S21b_tod=e2e/runs/S21b_tod S21d_xs2=e2e/runs/S21d_xs2
echo "==================== $(date '+%m-%d %T') 两折对比 S21（集成：S20b + 各臂 vs EMA 两种子）" >> $LIVE
cmp S21e ens_ema2=e2e/runs/S20_ens_ema2 ens_tod=e2e/runs/S21_ens_S21b_tod ens_xs2=e2e/runs/S21_ens_S21d_xs2
echo "==================== $(date '+%m-%d %T') 与 S20b 的预测相关（EMA 两种子为 0.941 / 0.934）" >> $LIVE
$PY e2e/scripts/seed_corr.py S20b_ema995 S21b_tod S20b_ema995 S21d_xs2 >> $LIVE 2>&1
echo "==================== $(date '+%m-%d %T') S21 分析完成" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_S21e.txt 1
pred S21e_tod_s43
$PY e2e/scripts/ens_pred.py e2e/runs/S21_ens_tod2 e2e/runs/S21b_tod e2e/runs/S21e_tod_s43 >> $LIVE 2>&1
$PY e2e/scripts/ens_pred.py e2e/runs/S21_ens_mix4 e2e/runs/S20b_ema995 e2e/runs/S20c_ema995_s43 e2e/runs/S21b_tod e2e/runs/S21e_tod_s43 >> $LIVE 2>&1
$PY e2e/scripts/ens_pred.py e2e/runs/S21_ens_tod2_base1 e2e/runs/S21b_tod e2e/runs/S21e_tod_s43 e2e/runs/S20b_ema995 >> $LIVE 2>&1
echo "==================== $(date '+%m-%d %T') 两折对比 S21e（四种集成：底座对 / 时段对 / 混合对 / 四个全上）" >> $LIVE
cmp S21f tod43=e2e/runs/S21e_tod_s43 base2=e2e/runs/S20_ens_ema2 tod2=e2e/runs/S21_ens_tod2 mix2=e2e/runs/S21_ens_S21b_tod tod2b1=e2e/runs/S21_ens_tod2_base1 all4=e2e/runs/S21_ens_mix4
echo "==================== $(date '+%m-%d %T') 时段两种子相关" >> $LIVE
$PY e2e/scripts/seed_corr.py S21b_tod S21e_tod_s43 >> $LIVE 2>&1
echo "==================== $(date '+%m-%d %T') S21e 完成" >> $LIVE
