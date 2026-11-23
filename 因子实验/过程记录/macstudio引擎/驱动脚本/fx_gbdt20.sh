#!/bin/bash
# F15（10-19）：树用更新的数据拟合是否往后看更好。同配置、同种子（7），拟合 2020–23 与 2020–22，只比 2024 年。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 --fit 2020-2023 "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag f23_d4t150 > $L/f23_d4t150.log 2>&1 &
run --depth 4 --trees 130 --aux-w 0.25 --tag f23_d4t130 > $L/f23_d4t130.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag f23_d5 > $L/f23_d5.log 2>&1 &
run --depth 6 --trees 85 --aux-w 0.50 --tag f23_d6 > $L/f23_d6.log 2>&1 &
wait
rm -f $L/c24u.log
for p in "f23_d4t150 d4t150a025s" "f23_d4t130 d4t130a025s" "f23_d5 d5a040s" "f23_d6 d6t85a050s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24u.log 2>&1; done
echo done > $L/fx_gbdt20.done
