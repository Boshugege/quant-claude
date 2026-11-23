#!/bin/bash
# F15b（10-19）：1617 的 16 个配置与种子，全部拟合到 2023（2020–23），16 模型平均，与 1617（拟合到 2022）比 2024 年。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 --fit 2020-2023 "$@"; }
for s in 7 11 23 42; do run --depth 4 --trees 150 --aux-w 0.25 --seed $s --tag g23_d4t150_s$s > $L/g23_d4t150_s$s.log 2>&1 & run --depth 4 --trees 130 --aux-w 0.25 --seed $s --tag g23_d4t130_s$s > $L/g23_d4t130_s$s.log 2>&1 & done; wait
for s in 7 11 23 42 3 5 13 17; do run --depth 5 --aux-w 0.40 --seed $s --tag g23_d5_s$s > $L/g23_d5_s$s.log 2>&1 & done; wait
G="g23_d4t150_s7+g23_d4t150_s11+g23_d4t150_s23+g23_d4t150_s42+g23_d4t130_s7+g23_d4t130_s11+g23_d4t130_s23+g23_d4t130_s42+g23_d5_s7+g23_d5_s11+g23_d5_s23+g23_d5_s42+g23_d5_s3+g23_d5_s5+g23_d5_s13+g23_d5_s17"
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $G > $L/c24v.log 2>&1
echo done > $L/fx_gbdt21.done
