#!/bin/bash
# R38（10-25 晚，批判性审视）：端到端预测约 21% 的截面方差是官方风格＋行业，评分机剔除后丢掉（IC 0.127→0.112）；训练用的行情代理只覆盖
# 规模 0.50、BTOP 0.52、行业 0.32。上界实验：训练剔除直接用官方暴露（只本地，不能交付），量"完全对齐"能多多少剔除后 IC。种子 42，对照 R22a。
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log; R=e2e/runs
while pgrep -f "e2e/scripts/train_local.py|gbdt_member.py" >/dev/null; do sleep 30; done
echo "==================== $(date "+%m-%d %T") R38 开始（训练剔除用官方暴露，上界）" >> $LIVE
bash e2e/scripts/queue.sh $R/queue_R38.txt 1
r=R38a_official
[ -f $R/$r/transformer_model.json ] && [ ! -f $R/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py $R/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
echo "==================== $(date "+%m-%d %T") 两折对比 R38" >> $LIVE
TAG=R38 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py R22a=$R/R22a_ep6_mtl05 R24a=$R/R24a_mtl05_s44 R24c=$R/R24c_mtl05_s43 R38a=$R/$r 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R38 完成" >> $LIVE
