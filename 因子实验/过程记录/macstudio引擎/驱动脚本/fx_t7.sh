#!/bin/bash
# T7（10-20）：树的训练目标逐截面对风格 + 行业取残差（与评测口径一致）；1617 的三个配置、种子 7，与同胞比 2023／2024。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 --target-resid "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag tr_d4t150 > $L/tr_d4t150.log 2>&1 &
run --depth 4 --trees 130 --aux-w 0.25 --tag tr_d4t130 > $L/tr_d4t130.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag tr_d5 > $L/tr_d5.log 2>&1 &
wait
rm -f $L/c24tr.log
for p in "tr_d4t150 d4t150a025s" "tr_d4t130 d4t130a025s" "tr_d5 d5a040s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24tr.log 2>&1; done
REFS=sib3=d4t150a025s+d4t130a025s+d5a040s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py tr_d4t150+tr_d4t130+tr_d5 >> $L/c24tr.log 2>&1
echo done > $L/fx_t7.done
