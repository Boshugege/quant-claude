#!/bin/bash
# F13（10-19）：盘口深层新根进树（BK_DIMB／BK_CONS／BK_THIN，排名），1617 的三个配置、种子 7，与同胞直接比。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20,BK_DIMB,BK_CONS,BK_THIN --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag bk_d4t150 --save-json $L/h2_bk_d4t150.json > $L/bk_d4t150.log 2>&1 &
run --depth 4 --trees 130 --aux-w 0.25 --tag bk_d4t130 --save-json $L/h2_bk_d4t130.json > $L/bk_d4t130.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag bk_d5 --save-json $L/h2_bk_d5.json > $L/bk_d5.log 2>&1 &
wait
rm -f $L/c24q.log
for p in "bk_d4t150 d4t150a025s" "bk_d4t130 d4t130a025s" "bk_d5 d5a040s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24q.log 2>&1; done
echo done > $L/fx_gbdt16.done
