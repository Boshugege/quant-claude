#!/bin/bash
# T6（10-20）：1617 的配置比例再训 16 个新种子（深度 4 的两种各 4 个、深度 5 的 8 个），看 32 模型平均相对 1617、新 16 模型单独相对 1617（种子运气检验）。2023／2024 判断。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
for s in 3 5 13 17; do
  run --depth 4 --trees 150 --aux-w 0.25 --seed $s --tag x_d4t150_s$s > $L/x_d4t150_s$s.log 2>&1 &
  run --depth 4 --trees 130 --aux-w 0.25 --seed $s --tag x_d4t130_s$s > $L/x_d4t130_s$s.log 2>&1 &
done; wait
for s in 19 29 31 37 41 43 47 53; do run --depth 5 --aux-w 0.40 --seed $s --tag x_d5_s$s > $L/x_d5_s$s.log 2>&1 & done; wait
N="x_d4t150_s3+x_d4t150_s5+x_d4t150_s13+x_d4t150_s17+x_d4t130_s3+x_d4t130_s5+x_d4t130_s13+x_d4t130_s17+x_d5_s19+x_d5_s29+x_d5_s31+x_d5_s37+x_d5_s41+x_d5_s43+x_d5_s47+x_d5_s53"
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $N D4+D5+$N > $L/c24x.log 2>&1
echo done > $L/fx_t6.done
