#!/bin/bash
# F10（10-19）：N_dir 在平台上是活的（用户自查 10-18），树特征里把它放回（只去 Y_sd5、X_n5d）；1617 的 16 个配置与种子原样重拟合，与 1617 直接比。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
L=/Users/user/Desktop/work/quant/fx/root/logs
for s in 7 11 23 42; do run --depth 4 --trees 150 --aux-w 0.25 --seed $s --tag nd_d4t150_s$s --save-json $L/h2_nd_d4t150_s$s.json > $L/nd_d4t150_s$s.log 2>&1 & done; wait
for s in 7 11 23 42; do run --depth 4 --trees 130 --aux-w 0.25 --seed $s --tag nd_d4t130_s$s --save-json $L/h2_nd_d4t130_s$s.json > $L/nd_d4t130_s$s.log 2>&1 & done; wait
for s in 7 11 23 42; do run --depth 5 --aux-w 0.40 --seed $s --tag nd_d5_s$s --save-json $L/h2_nd_d5_s$s.json > $L/nd_d5_s$s.log 2>&1 & done; wait
for s in 3 5 13 17; do run --depth 5 --aux-w 0.40 --seed $s --tag nd_d5_s$s --save-json $L/h2_nd_d5_s$s.json > $L/nd_d5_s$s.log 2>&1 & done; wait
N4="nd_d4t150_s7+nd_d4t150_s11+nd_d4t150_s23+nd_d4t150_s42+nd_d4t130_s7+nd_d4t130_s11+nd_d4t130_s23+nd_d4t130_s42"
N5="nd_d5_s7+nd_d5_s11+nd_d5_s23+nd_d5_s42+nd_d5_s3+nd_d5_s5+nd_d5_s13+nd_d5_s17"
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $N4+$N5 $N4 $N5 > $L/c24l.log 2>&1
echo done > $L/fx_gbdt12.done
