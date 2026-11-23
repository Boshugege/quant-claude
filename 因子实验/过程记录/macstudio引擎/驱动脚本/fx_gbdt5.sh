#!/bin/bash
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --depth 5 "$@"; }
L=/Users/user/Desktop/work/quant/fx/root/logs
for s in 11 23 42; do
  run --aux-w 0.40 --aux-n 2 --seed $s --tag d5a040s_s$s --save-json $L/h2_d5a040s_s$s.json > $L/d5a040s_s$s.log 2>&1 &
  run --trees 80 --aux-w 0.30 --aux-n 2 --seed $s --tag d5t80a030s_s$s --save-json $L/h2_d5t80a030s_s$s.json > $L/d5t80a030s_s$s.log 2>&1 &
done
wait
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24_tree_avg.py d5a040s_s11 d5a040s_s23 d5a040s_s42 d5t80a030s_s11 d5t80a030s_s23 d5t80a030s_s42 d5a040s+d5a040s_s11+d5a040s_s23+d5a040s_s42 d5t80a030s+d5t80a030s_s11+d5t80a030s_s23+d5t80a030s_s42 d5a040s+d5a040s_s11+d5a040s_s23+d5a040s_s42+d5t80a030s+d5t80a030s_s11+d5t80a030s_s23+d5t80a030s_s42 > $L/c24d.log 2>&1
echo done > $L/fx_gbdt5.done
