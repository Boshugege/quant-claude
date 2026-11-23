#!/bin/bash
# T11f（10-22）：t11a（一档规模比 QS1L、QS1A 进树）按 1617 的 16 个成员重训（深度 4：150／130 棵 × 种子 7、11、23、42，多期 0.25；
# 深度 5：100 棵 × 种子 7、11、23、42、3、5、13、17，多期 0.40），两波各 8 个（与 e2e GPU 训练同机，控制内存）。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
X=L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20,QS1L,QS1A
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 --extra $X "$@"; }
for s in 11 23 42; do run --depth 4 --trees 150 --aux-w 0.25 --seed $s --tag t11f_d4t150_s$s > $L/t11f_d4t150_s$s.log 2>&1 & done
for s in 11 23 42; do run --depth 4 --trees 130 --aux-w 0.25 --seed $s --tag t11f_d4t130_s$s > $L/t11f_d4t130_s$s.log 2>&1 & done
for s in 11 23; do run --depth 5 --aux-w 0.40 --seed $s --tag t11f_d5_s$s > $L/t11f_d5_s$s.log 2>&1 & done
wait
for s in 42 3 5 13 17; do run --depth 5 --aux-w 0.40 --seed $s --tag t11f_d5_s$s > $L/t11f_d5_s$s.log 2>&1 & done
wait
T="t11a_d4t150+t11f_d4t150_s11+t11f_d4t150_s23+t11f_d4t150_s42+t11a_d4t130+t11f_d4t130_s11+t11f_d4t130_s23+t11f_d4t130_s42+t11a_d5+t11f_d5_s11+t11f_d5_s23+t11f_d5_s42+t11f_d5_s3+t11f_d5_s5+t11f_d5_s13+t11f_d5_s17"
rm -f $L/c24t11f.log
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $T >> $L/c24t11f.log 2>&1
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c108_export_trees.py t11f $T >> $L/c24t11f.log 2>&1
echo done > $L/fx_t11f.done
