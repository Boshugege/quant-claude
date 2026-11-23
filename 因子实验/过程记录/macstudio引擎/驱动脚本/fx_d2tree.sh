#!/bin/bash
# 可解释的树（10-19）：深度 2（每棵树只问两个问题 = 单因子形状 + 两两交互），1617 的特征与多期目标，多配置 × 4 种子平均，与 1617 比。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 --depth 2 "$@"; }
for s in 7 11 23 42; do run --trees 400 --aux-w 0.25 --seed $s --tag dd2_t400_s$s --save-json $L/h2_dd2_t400_s$s.json > $L/dd2_t400_s$s.log 2>&1 & run --trees 300 --aux-w 0.40 --seed $s --tag dd2_t300a4_s$s --save-json $L/h2_dd2_t300a4_s$s.json > $L/dd2_t300a4_s$s.log 2>&1 & done; wait
A="dd2_t400_s7+dd2_t400_s11+dd2_t400_s23+dd2_t400_s42"; B="dd2_t300a4_s7+dd2_t300a4_s11+dd2_t300a4_s23+dd2_t300a4_s42"
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $A $B $A+$B > $L/c24y.log 2>&1
echo done > $L/fx_d2tree.done
