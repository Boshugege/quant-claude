#!/bin/bash
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --depth 4 --aux-w 0.25 --aux-n 2 "$@"; }
L=/Users/user/Desktop/work/quant/fx/root/logs
run --trees 150 --seed 11 --tag t150s11 --save-json $L/h2_t150s11.json > $L/t150s11.log 2>&1 &
run --trees 150 --seed 23 --tag t150s23 --save-json $L/h2_t150s23.json > $L/t150s23.log 2>&1 &
run --trees 150 --seed 42 --tag t150s42 --save-json $L/h2_t150s42.json > $L/t150s42.log 2>&1 &
run --trees 130 --seed 42 --tag t130s42 --save-json $L/h2_t130s42.json > $L/t130s42.log 2>&1 &
wait
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24_tree_avg.py d4t150a025s+t150s11 t150s23+t150s42 d4t130a025s_s23+t130s42 d4t150a025s+t150s11+t150s23+t150s42 d4t130a025s+d4t130a025s_s11+d4t130a025s_s23+t130s42 d4t150a025s+t150s11+t150s23+t150s42+d4t130a025s+d4t130a025s_s11+d4t130a025s_s23+t130s42 t150s11 t150s23 t150s42 t130s42 > $L/c24b.log 2>&1
echo done > $L/fx_gbdt3.done
