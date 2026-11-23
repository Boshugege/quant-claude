#!/bin/bash
# 端到端第一轮（10-18，目标"探索到本质提升"）：在 S21b（融合头 + 辅助 0.5 + EMA + 时段 token，fold-A，种子 42）上逐个加一处改动。
#   R1a 列表式排序损失 0.5、R1b 尾部加权相关 0.5、R1c 尾盘段通道 cls、R1d 星期几 one-hot、R1e dropout 0.2 + weight decay 0.01、
#   R1f 不同输入视角（去盘口通道 + 日线 60 天，只作集成搭档评估）。噪声带 = S21e（种子 43）。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
cmp() { TAG=$1 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py "${@:2}" 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE; }
pred() { echo "==================== $(date '+%m-%d %T') 预测 2024：$1" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$1 2024 2>&1 | grep -v -i warn >> $LIVE; }
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') R1 开始（第一轮：损失、通道、正则、输入视角）" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_R1.txt 1
ARMS="R1a_listnet05 R1b_tail05 R1c_cls R1d_dow R1e_reg R1f_view"
for r in $ARMS; do [ -f e2e/runs/$r/transformer_model.json ] && pred $r; done
$PY e2e/scripts/ens_pred.py e2e/runs/R1_ens_view e2e/runs/S21b_tod e2e/runs/R1f_view >> $LIVE 2>&1
echo "==================== $(date '+%m-%d %T') 两折对比 R1（单臂，对照 S21b 与噪声带 S21e）" >> $LIVE
A="base=e2e/runs/S21b_tod seed43=e2e/runs/S21e_tod_s43"; for r in $ARMS; do A="$A $r=e2e/runs/$r"; done
cmp R1 $A
echo "==================== $(date '+%m-%d %T') 两折对比 R1（集成搭档：时段两种子 vs S21b + R1f）" >> $LIVE
cmp R1e tod2=e2e/runs/S21_ens_tod2 base_view=e2e/runs/R1_ens_view
echo "==================== $(date '+%m-%d %T') 与 S21b 的预测相关（时段两种子 0.938 / 0.933）" >> $LIVE
$PY e2e/scripts/seed_corr.py S21b_tod R1f_view >> $LIVE 2>&1
echo "==================== $(date '+%m-%d %T') R1 完成" >> $LIVE
