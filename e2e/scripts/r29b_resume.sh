#!/bin/bash
# R29 续跑（10-22 下午）：macstudio 14:38 重启，R29b 第 6 轮训练到一半中断，r29_run.sh 也没了。
#   1）R29a 补 2024 预测；2）R29b 从第 5 轮存档续训（--resume）；3）R29b 预测 2024；4）两折对比 R29 与压力两半
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
echo "==================== $(date "+%m-%d %T") R29 续跑（重启后）" >> $LIVE
[ -f e2e/runs/R29a_extra7_s44/pred_2024.parquet ] || { echo "==================== $(date "+%m-%d %T") 预测 2024：R29a_extra7_s44" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/R29a_extra7_s44 2024 2>&1 | grep -v -i warn >> $LIVE; }
L=$(grep -- "--run R29b_extra14 " e2e/runs/queue_R29.txt)
echo "==================== $(date "+%m-%d %T") 开始 R29b_extra14（续训）: $L --resume" >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/train_local.py $L --resume 2>&1 | grep --line-buffered -v -E "Warning|warn\(|^ +" >> $LIVE
echo "==================== $(date "+%m-%d %T") 结束 R29b_extra14 (exit ${PIPESTATUS[0]})" >> $LIVE
[ -f e2e/runs/R29b_extra14/transformer_model.json ] && [ ! -f e2e/runs/R29b_extra14/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：R29b_extra14" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/R29b_extra14 2024 2>&1 | grep -v -i warn >> $LIVE; }
echo "==================== $(date "+%m-%d %T") 两折对比 R29" >> $LIVE
A="R22a=e2e/runs/R22a_ep6_mtl05 R24c=e2e/runs/R24c_mtl05_s43 R24a=e2e/runs/R24a_mtl05_s44 R27a=e2e/runs/R27a_extra7 R28a=e2e/runs/R28a_extra7_s43"
for r in R29a_extra7_s44 R29b_extra14; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R29 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/stress_halves.py $A 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R29 完成" >> $LIVE
