#!/bin/bash
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --depth 6 "$@"; }
L=/Users/user/Desktop/work/quant/fx/root/logs
run --aux-w 0.60 --aux-n 2 --tag d6a060s --save-json $L/h2_d6a060s.json > $L/d6a060s.log 2>&1 &
run --aux-w 0.70 --aux-n 2 --tag d6a070s --save-json $L/h2_d6a070s.json > $L/d6a070s.log 2>&1 &
run --trees 85 --aux-w 0.50 --aux-n 2 --tag d6t85a050s --save-json $L/h2_d6t85a050s.json > $L/d6t85a050s.log 2>&1 &
run --trees 70 --aux-w 0.40 --aux-n 2 --seed 11 --tag d6t70a040s_s11 --save-json $L/h2_d6t70a040s_s11.json > $L/d6t70a040s_s11.log 2>&1 &
wait
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24_tree_avg.py d6a060s d6a070s d6t85a050s d6t70a040s_s11 d6t70a040s+d6t70a040s_s11 > $L/c24h.log 2>&1
echo done > $L/fx_gbdt8.done
