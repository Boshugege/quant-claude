#!/bin/bash
# E15（10-19 用户确认）：P13 配方（秩目标 + 4 轮 + 融合头 + 辅助 0.5 + EMA 0.995 + 当前时段 token）全量 2020–24，四个成员（种子 42–45）串行。
# 先等 R12a 训完、补它的 2024 预测并与 R3f／R8b 对比（P13 配方三个种子），再开 E15。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 20; done
echo "==================== $(date "+%m-%d %T") 预测 2024：R12a_rank_ep4_s44" >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/R12a_rank_ep4_s44 2024 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") 两折对比 R12（P13 配方三个种子）" >> $LIVE
TAG=R12 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py base=e2e/runs/S21b_tod R3f_s42=e2e/runs/R3f_rank_ep4 R8b_s43=e2e/runs/R8b_rank_ep4_s43 R12a_s44=e2e/runs/R12a_rank_ep4_s44 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R12 完成" >> $LIVE
echo "==================== $(date "+%m-%d %T") E15 训练开始（四成员串行）" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_E15.txt 1
echo "==================== $(date "+%m-%d %T") E15 训练完成" >> $LIVE
