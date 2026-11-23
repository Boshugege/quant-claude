#!/bin/bash
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees "$@"; }
L=/Users/user/Desktop/work/quant/fx/root/logs
run --depth 5 --aux-w 0.30 --aux-n 2 --tag d5a030s --save-json $L/h2_d5a030s.json > $L/d5a030s.log 2>&1 &
run --depth 5 --aux-w 0.40 --aux-n 2 --tag d5a040s --save-json $L/h2_d5a040s.json > $L/d5a040s.log 2>&1 &
run --depth 5 --trees 80 --aux-w 0.30 --aux-n 2 --tag d5t80a030s --save-json $L/h2_d5t80a030s.json > $L/d5t80a030s.log 2>&1 &
run --depth 4 --trees 200 --aux-w 0.35 --aux-n 2 --tag d4t200a035s --save-json $L/h2_d4t200a035s.json > $L/d4t200a035s.log 2>&1 &
wait
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24_tree_avg.py d5a030s d5a040s d5t80a030s d4t200a035s d4t150a025s+t150s11+t150s23+t150s42+d4t130a025s+d4t130a025s_s11+d4t130a025s_s23+t130s42 > $L/c24c.log 2>&1
echo done > $L/fx_gbdt4.done
