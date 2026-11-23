#!/bin/bash
# F8（10-18 夜）：深度 6 第二配置（70 棵、多期 0.40）补 6 个种子凑满 8 个不挑；深度 7 探路 4 个配置；组合同时对照 1600 与 1617；1603 混树检验。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees "$@"; }
L=/Users/user/Desktop/work/quant/fx/root/logs
B="--depth 6 --trees 70 --aux-w 0.40 --aux-n 2"
for s in 23 42 3 5; do run $B --seed $s --tag d6t70a040s_s$s --save-json $L/h2_d6t70a040s_s$s.json > $L/d6t70a040s_s$s.log 2>&1 & done; wait
for s in 13 17; do run $B --seed $s --tag d6t70a040s_s$s --save-json $L/h2_d6t70a040s_s$s.json > $L/d6t70a040s_s$s.log 2>&1 & done
run --depth 7 --trees 60 --aux-w 0.50 --aux-n 2 --tag d7t60a050s --save-json $L/h2_d7t60a050s.json > $L/d7t60a050s.log 2>&1 &
run --depth 7 --trees 60 --aux-w 0.60 --aux-n 2 --tag d7t60a060s --save-json $L/h2_d7t60a060s.json > $L/d7t60a060s.log 2>&1 &
wait
run --depth 7 --trees 45 --aux-w 0.50 --aux-n 2 --tag d7t45a050s --save-json $L/h2_d7t45a050s.json > $L/d7t45a050s.log 2>&1 &
run --depth 7 --trees 45 --aux-w 0.40 --aux-n 2 --tag d7t45a040s --save-json $L/h2_d7t45a040s.json > $L/d7t45a040s.log 2>&1 &
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c33_1603_tree.py > $L/c33.log 2>&1 &
wait
D6A="d6t85a050s+d6t85a050s_s11+d6t85a050s_s23+d6t85a050s_s42+d6t85a050s_s3+d6t85a050s_s5+d6t85a050s_s13+d6t85a050s_s17"
D6B="d6t70a040s+d6t70a040s_s11+d6t70a040s_s23+d6t70a040s_s42+d6t70a040s_s3+d6t70a040s_s5+d6t70a040s_s13+d6t70a040s_s17"
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $D6A $D6B $D6A+$D6B D4+$D6A D4+$D6A+$D6B D4+D5+$D6A+$D6B d7t60a050s d7t60a060s d7t45a050s d7t45a040s > $L/c24j.log 2>&1
echo done > $L/fx_gbdt10.done
