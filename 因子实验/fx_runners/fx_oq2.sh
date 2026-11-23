#!/bin/bash
# 10-20：盘口委托笔数深挖。c67 细化叠加；T8 委托笔数特征进树（快 3 + 慢 3），三配置、种子 7，与同胞比 2023／2024。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c67_oq_combo.py > $L/c67.log 2>&1 &
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20,OQ_NIMB,OQ_SIZE,OQ_DSIZE,OQ_NIMB_20,OQ_SIZE_20,OQ_DSIZE_20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag oq_d4t150 > $L/oq_d4t150.log 2>&1 &
run --depth 4 --trees 130 --aux-w 0.25 --tag oq_d4t130 > $L/oq_d4t130.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag oq_d5 > $L/oq_d5.log 2>&1 &
wait
rm -f $L/c24oq.log
for p in "oq_d4t150 d4t150a025s" "oq_d4t130 d4t130a025s" "oq_d5 d5a040s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24oq.log 2>&1; done
REFS=sib3=d4t150a025s+d4t130a025s+d5a040s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py oq_d4t150+oq_d4t130+oq_d5 >> $L/c24oq.log 2>&1
echo done > $L/fx_oq2.done
