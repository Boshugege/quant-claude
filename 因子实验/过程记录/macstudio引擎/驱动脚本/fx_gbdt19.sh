#!/bin/bash
# F13c（10-19）：盘口快版本（加 IC、加换手）+ 慢版本（降换手）一起进树，三个配置、种子 7，与同胞比。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20,BK_DIMB,BK_CONS,BK_THIN,BK_DIMB_20,BK_SLOPEI_20,BK_AOS_20,BK_RESIL_20,BK_CONS_20,BK_THIN_20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag bb_d4t150 --save-json $L/h2_bb_d4t150.json > $L/bb_d4t150.log 2>&1 &
run --depth 4 --trees 130 --aux-w 0.25 --tag bb_d4t130 --save-json $L/h2_bb_d4t130.json > $L/bb_d4t130.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag bb_d5 --save-json $L/h2_bb_d5.json > $L/bb_d5.log 2>&1 &
wait
rm -f $L/c24t.log
for p in "bb_d4t150 d4t150a025s" "bb_d4t130 d4t130a025s" "bb_d5 d5a040s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24t.log 2>&1; done
echo done > $L/fx_gbdt19.done
