#!/bin/bash
# G8／G9（10-25）：树模型成员再加端到端的其余通道（收盘段季节性 cls、截面状态 mkt、领先滞后 lead、星期 dow、收盘竞价 auc），更深的树。
# 等 R36 跑完（macstudio 训练串行）。评估：四成员 Transformer 集成 + 树（五分之一权重）。
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
until grep -q "R36 完成" e2e/runs/live.log; do sleep 120; done
W=shape,book,book2,book3,tod,pos,state,cls,mkt,lead,dow,auc
GB_RICH=1 $PY e2e/scripts/gbdt_member.py G8_win12 --fields full --norm rel3 --tgt e16 --win $W --trees 800 --depth 8 --lr 0.03 --rows 0.5 --cols 0.7 --min-leaf 1000 > e2e/runs/G8.out 2>&1
bash e2e/scripts/g1_eval.sh G8_win12 > e2e/runs/G8_eval.txt 2>&1
GB_RICH=1 $PY e2e/scripts/gbdt_member.py G9_win12_d10 --fields full --norm rel3 --tgt e16 --win $W --trees 1000 --depth 10 --lr 0.02 --rows 0.5 --cols 0.5 --min-leaf 500 > e2e/runs/G9.out 2>&1
bash e2e/scripts/g1_eval.sh G9_win12_d10 > e2e/runs/G9_eval.txt 2>&1
