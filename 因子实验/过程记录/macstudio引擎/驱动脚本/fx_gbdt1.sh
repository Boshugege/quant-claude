#!/bin/bash
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees "$@"; }
L=/Users/user/Desktop/work/quant/fx/root/logs
run --depth 4 --aux-w 0.20 --aux-n 2 --tag d4a020s --save-json $L/h2_d4a020s.json > $L/d4a020s.log 2>&1 &
run --depth 4 --aux-w 0.30 --aux-n 2 --tag d4a030s --save-json $L/h2_d4a030s.json > $L/d4a030s.log 2>&1 &
run --depth 4 --trees 130 --aux-w 0.25 --aux-n 2 --tag d4t130a025s --save-json $L/h2_d4t130a025s.json > $L/d4t130a025s.log 2>&1 &
run --depth 4 --aux-w 0.25 --aux-n 2 --seed 11 --tag d4a025s_s11 --save-json $L/h2_d4a025s_s11.json > $L/d4a025s_s11.log 2>&1 &
run --depth 4 --aux-w 0.25 --aux-n 2 --target-gauss --tag d4a025g --save-json $L/h2_d4a025g.json > $L/d4a025g.log 2>&1 &
wait
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c21_gbdt_platform.py d4a020s d4a030s d4t130a025s d4a025s_s11 d4a025g > $L/c21_fx1.log 2>&1
echo done > $L/fx_gbdt1.done
