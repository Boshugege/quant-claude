#!/bin/bash
# T3a（10-20）：第一批"日频过热"四个慢信号（K_PH250、K_VTS、K_VOLTS、K_CGO）作为特征进树；三配置、种子 7，拟合 2020–22，与同胞比 2023／2024。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20,K_PH250,K_VTS,K_VOLTS,K_CGO --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag k1_d4t150 > $L/k1_d4t150.log 2>&1 &
run --depth 4 --trees 130 --aux-w 0.25 --tag k1_d4t130 > $L/k1_d4t130.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag k1_d5 > $L/k1_d5.log 2>&1 &
wait
rm -f $L/c24k1.log
for p in "k1_d4t150 d4t150a025s" "k1_d4t130 d4t130a025s" "k1_d5 d5a040s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24k1.log 2>&1; done
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py k1_d4t150+k1_d4t130+k1_d5 >> $L/c24k1.log 2>&1
echo done > $L/fx_t3a.done
