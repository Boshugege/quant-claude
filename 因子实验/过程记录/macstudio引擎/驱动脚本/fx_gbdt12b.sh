#!/bin/bash
# F10b（10-19）：深度 6（85 棵、多期 0.50）也放回 N_dir，8 个种子；等 F10 完成后再一起出组合对照。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 --depth 6 --trees 85 --aux-w 0.50 "$@"; }
L=/Users/user/Desktop/work/quant/fx/root/logs
for s in 7 11 23 42; do run --seed $s --tag nd_d6_s$s --save-json $L/h2_nd_d6_s$s.json > $L/nd_d6_s$s.log 2>&1 & done; wait
for s in 3 5 13 17; do run --seed $s --tag nd_d6_s$s --save-json $L/h2_nd_d6_s$s.json > $L/nd_d6_s$s.log 2>&1 & done; wait
until [ -f $L/fx_gbdt12.done ]; do sleep 30; done
N4="nd_d4t150_s7+nd_d4t150_s11+nd_d4t150_s23+nd_d4t150_s42+nd_d4t130_s7+nd_d4t130_s11+nd_d4t130_s23+nd_d4t130_s42"
N5="nd_d5_s7+nd_d5_s11+nd_d5_s23+nd_d5_s42+nd_d5_s3+nd_d5_s5+nd_d5_s13+nd_d5_s17"
N6="nd_d6_s7+nd_d6_s11+nd_d6_s23+nd_d6_s42+nd_d6_s3+nd_d6_s5+nd_d6_s13+nd_d6_s17"
D6A="d6t85a050s+d6t85a050s_s11+d6t85a050s_s23+d6t85a050s_s42+d6t85a050s_s3+d6t85a050s_s5+d6t85a050s_s13+d6t85a050s_s17"
REFS=1618=$D6A PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $N6 $N4+$N5+$N6 > $L/c24m.log 2>&1
echo done > $L/fx_gbdt12b.done
