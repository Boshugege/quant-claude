#!/bin/bash
# T4（10-20）：单调约束树（方向 = 拟合区间 2020–22 内各特征与目标相关的符号，时段哑变量不约束）；1617 的三个配置、种子 7，与同胞比 2023／2024。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 --mono "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag mono_d4t150 > $L/mono_d4t150.log 2>&1 &
run --depth 4 --trees 130 --aux-w 0.25 --tag mono_d4t130 > $L/mono_d4t130.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag mono_d5 > $L/mono_d5.log 2>&1 &
wait
rm -f $L/c24m.log
for p in "mono_d4t150 d4t150a025s" "mono_d4t130 d4t130a025s" "mono_d5 d5a040s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24m.log 2>&1; done
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py mono_d4t150+mono_d4t130+mono_d5 >> $L/c24m.log 2>&1
echo done > $L/fx_t4.done
