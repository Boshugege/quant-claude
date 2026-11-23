#!/bin/bash
# 新方向 D2（10-19）：盘口深层树——只用盘口特征（快 + 慢）与时段哑变量，1617 的三个配置、种子 7；同时跑 D1 线性。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c49_book_dir.py > $L/c49.log 2>&1 &
X=BK_DIMB,BK_SLOPEI,BK_CONS,BK_RESIL,BK_AOS_DEV,BK_THIN,BK_DIMB_20,BK_SLOPEI_20,BK_CONS_20,BK_RESIL_20,BK_AOS_20,BK_THIN_20
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra-only --extra $X --save --quiet-trees --aux-n 2 "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag bko_d4t150 > $L/bko_d4t150.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag bko_d5 > $L/bko_d5.log 2>&1 &
run --depth 6 --trees 85 --aux-w 0.50 --tag bko_d6 > $L/bko_d6.log 2>&1 &
wait
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py bko_d4t150 bko_d5 bko_d6 bko_d4t150+bko_d5+bko_d6 > $L/c24w.log 2>&1
echo done > $L/fx_d2.done
