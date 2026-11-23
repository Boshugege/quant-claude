#!/bin/bash
# F7（10-18 夜）：深度 6 预先定配置（85 棵、多期 0.50）补 7 个种子、凑满 8 个不挑；再与深度 4／5 混合。种子名单固定：基础, 11, 23, 42 | 3, 5, 13, 17。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --depth 6 --trees 85 --aux-w 0.50 --aux-n 2 "$@"; }
L=/Users/user/Desktop/work/quant/fx/root/logs
for s in 11 23 42 3; do run --seed $s --tag d6t85a050s_s$s --save-json $L/h2_d6t85a050s_s$s.json > $L/d6t85a050s_s$s.log 2>&1 & done; wait
for s in 5 13 17; do run --seed $s --tag d6t85a050s_s$s --save-json $L/h2_d6t85a050s_s$s.json > $L/d6t85a050s_s$s.log 2>&1 & done; wait
D4="d4t150a025s+t150s11+t150s23+t150s42+d4t130a025s+d4t130a025s_s11+d4t130a025s_s23+t130s42"
D5="d5a040s+d5a040s_s11+d5a040s_s23+d5a040s_s42+d5a040s_s3+d5a040s_s5+d5a040s_s13+d5a040s_s17"
D6="d6t85a050s+d6t85a050s_s11+d6t85a050s_s23+d6t85a050s_s42+d6t85a050s_s3+d6t85a050s_s5+d6t85a050s_s13+d6t85a050s_s17"
Q4="d4t150a025s+t150s11+t150s23+t150s42"; Q5="d5a040s+d5a040s_s11+d5a040s_s23+d5a040s_s42"; Q6="d6t85a050s+d6t85a050s_s11+d6t85a050s_s23+d6t85a050s_s42"
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24_tree_avg.py $D6 $D4+$D5 $D4+$D5+$D6 $D5+$D6 $Q4+$Q5+$Q6 > $L/c24i.log 2>&1
echo done > $L/fx_gbdt9.done
