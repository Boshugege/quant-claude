#!/bin/bash
# F9（10-18 夜）：多期目标的跨度轴。同配置、同种子（默认 7），把"之后 2 个截面"改成"之后 3 个"或"当日剩余全部"（--aux-rest），与跨度 2 的同胞直接比。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees "$@"; }
L=/Users/user/Desktop/work/quant/fx/root/logs
run --depth 6 --trees 85 --aux-w 0.50 --aux-n 3 --tag d6t85a050n3 --save-json $L/h2_d6t85a050n3.json > $L/d6t85a050n3.log 2>&1 &
run --depth 6 --trees 85 --aux-w 0.50 --aux-rest --tag d6t85a050r --save-json $L/h2_d6t85a050r.json > $L/d6t85a050r.log 2>&1 &
run --depth 4 --trees 150 --aux-w 0.25 --aux-n 3 --tag d4t150a025n3 --save-json $L/h2_d4t150a025n3.json > $L/d4t150a025n3.log 2>&1 &
run --depth 4 --trees 150 --aux-w 0.25 --aux-rest --tag d4t150a025r --save-json $L/h2_d4t150a025r.json > $L/d4t150a025r.log 2>&1 &
wait
REFS=sib=d6t85a050s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py d6t85a050n3 d6t85a050r > $L/c24k.log 2>&1
REFS=sib=d4t150a025s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py d4t150a025n3 d4t150a025r >> $L/c24k.log 2>&1
echo done > $L/fx_gbdt11.done
