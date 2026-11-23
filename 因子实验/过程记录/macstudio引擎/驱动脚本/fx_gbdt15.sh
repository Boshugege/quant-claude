#!/bin/bash
# F12（10-19）：截面状态特征（MKT_*，加信息、加换手）+ 更强的多期目标（压换手），深度 4 两个配置 × 多期 0.35／0.45，与不加 MKT、多期 0.25 的同胞比。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20,MKT_SEG,MKT_DAY,MKT_DISP,MKT_PACE --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 --depth 4 "$@"; }
for t in 150 130; do for w in 035 045; do run --trees $t --aux-w 0.${w#0} --tag mk_d4t${t}a$w --save-json $L/h2_mk_d4t${t}a$w.json > $L/mk_d4t${t}a$w.log 2>&1 & done; done; wait
rm -f $L/c24p.log
for t in 150 130; do for w in 035 045; do REFS=sib=d4t${t}a025s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py mk_d4t${t}a$w >> $L/c24p.log 2>&1; done; done
echo done > $L/fx_gbdt15.done
