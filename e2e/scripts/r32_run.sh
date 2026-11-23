#!/bin/bash
# R32（10-23，用户："你都可以试试"）：端到端加收盘集合竞价通道（前几天的竞价收益、竞价成交占比；win group auc），种子 42。
#   a 对照 R22a（E16 配方）；b 再加 7 个中间时刻，对照 R27a。
# 等 E16 训完再换上新版提交脚本（submission_next：auc 通道 + 混合集成 E04_EXTRA_MEMBERS），避免改动正在训练的成员。
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "E16 训练完成" $LIVE; do sleep 60; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
cp e2e/submission/Transformer_modelsave_train.py e2e/submission_next/Transformer_modelsave_train.prev.py
cp e2e/submission_next/Transformer_modelsave_train.py e2e/submission/Transformer_modelsave_train.py
cp e2e/src_next/feather_data.py e2e/src/feather_data.py
echo "==================== $(date "+%m-%d %T") 换上新版提交脚本（auc 通道、混合集成）" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_R32.txt 1
RUNS="R32a_auc R32b_auc_x7"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] && [ ! -f e2e/runs/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R32" >> $LIVE
A="R22a=e2e/runs/R22a_ep6_mtl05 R27a=e2e/runs/R27a_extra7 R24a=e2e/runs/R24a_mtl05_s44 R24c=e2e/runs/R24c_mtl05_s43"
for r in $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R32 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/stress_halves.py $A 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R32 完成" >> $LIVE
