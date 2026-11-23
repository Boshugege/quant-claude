#!/bin/bash
# T9（10-20）：1617 的 16 个配置与种子 + 委托笔数特征（快 3 + 慢 3），拟合 2020–22，16 模型平均与 1617 比 2023／2024。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20,OQ_NIMB,OQ_SIZE,OQ_DSIZE,OQ_NIMB_20,OQ_SIZE_20,OQ_DSIZE_20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
for s in 11 23 42; do
  run --depth 4 --trees 150 --aux-w 0.25 --seed $s --tag oq_d4t150_s$s > $L/oq_d4t150_s$s.log 2>&1 &
  run --depth 4 --trees 130 --aux-w 0.25 --seed $s --tag oq_d4t130_s$s > $L/oq_d4t130_s$s.log 2>&1 &
done; wait
for s in 11 23 42 3 5 13 17; do run --depth 5 --aux-w 0.40 --seed $s --tag oq_d5_s$s > $L/oq_d5_s$s.log 2>&1 & done; wait
G="oq_d4t150+oq_d4t150_s11+oq_d4t150_s23+oq_d4t150_s42+oq_d4t130+oq_d4t130_s11+oq_d4t130_s23+oq_d4t130_s42+oq_d5+oq_d5_s11+oq_d5_s23+oq_d5_s42+oq_d5_s3+oq_d5_s5+oq_d5_s13+oq_d5_s17"
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $G > $L/c24t9.log 2>&1
echo done > $L/fx_t9.done
