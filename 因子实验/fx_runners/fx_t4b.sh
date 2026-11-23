#!/bin/bash
# T4b（10-20）：单调约束的分组消融（深度 4、150 棵、种子 7）：头部不约束／各根不约束／额外特征不约束，与同胞 d4t150a025s、全约束 mono_d4t150 比 2023／2024。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
H=s155,s139,s161,k1381,k1385,k1406,H1363,s152
RT=L_lab10,M_q_lin,W_vdir,W_zero,Z_mfr60,Z_mfr90
EX=L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra $EX --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 --depth 4 --trees 150 --aux-w 0.25 --mono "$@"; }
run --mono-free $H --tag mono_fh > $L/mono_fh.log 2>&1 &
run --mono-free $RT --tag mono_fr > $L/mono_fr.log 2>&1 &
run --mono-free $EX --tag mono_fe > $L/mono_fe.log 2>&1 &
wait
rm -f $L/c24mb.log
for t in mono_fh mono_fr mono_fe; do REFS=sib=d4t150a025s,mono=mono_d4t150 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $t >> $L/c24mb.log 2>&1; done
echo done > $L/fx_t4b.done
