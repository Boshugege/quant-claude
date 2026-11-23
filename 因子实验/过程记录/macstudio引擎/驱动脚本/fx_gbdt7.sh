#!/bin/bash
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees "$@"; }
L=/Users/user/Desktop/work/quant/fx/root/logs
run --depth 6 --aux-w 0.40 --aux-n 2 --tag d6a040s --save-json $L/h2_d6a040s.json > $L/d6a040s.log 2>&1 &
run --depth 6 --aux-w 0.50 --aux-n 2 --tag d6a050s --save-json $L/h2_d6a050s.json > $L/d6a050s.log 2>&1 &
run --depth 5 --aux-w 0.50 --aux-n 2 --tag d5a050s --save-json $L/h2_d5a050s.json > $L/d5a050s.log 2>&1 &
run --depth 6 --trees 70 --aux-w 0.40 --aux-n 2 --tag d6t70a040s --save-json $L/h2_d6t70a040s.json > $L/d6t70a040s.log 2>&1 &
wait
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24_tree_avg.py d6a040s d6a050s d5a050s d6t70a040s > $L/c24g.log 2>&1
echo done > $L/fx_gbdt7.done
