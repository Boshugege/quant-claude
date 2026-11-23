#!/bin/bash
# T4c（10-20）：头部不约束、各根与额外特征单调（mono_fh）补齐另外两个配置，三配置平均与同胞三配置平均比 2023／2024。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
H=s155,s139,s161,k1381,k1385,k1406,H1363,s152
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 --mono --mono-free $H "$@"; }
run --depth 4 --trees 130 --aux-w 0.25 --tag mono_fh_d4t130 > $L/mono_fh_d4t130.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag mono_fh_d5 > $L/mono_fh_d5.log 2>&1 &
wait
rm -f $L/c24mc.log
REFS=sib3=d4t150a025s+d4t130a025s+d5a040s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py mono_fh+mono_fh_d4t130+mono_fh_d5 >> $L/c24mc.log 2>&1
for p in "mono_fh_d4t130 d4t130a025s" "mono_fh_d5 d5a040s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24mc.log 2>&1; done
echo done > $L/fx_t4c.done
