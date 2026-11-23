#!/bin/bash
# F11（10-19）：树特征加 4 个截面状态特征（MKT_SEG／MKT_DAY／MKT_DISP／MKT_PACE，原始值），与 1617 的 4 个配置同种子（7）同胞直接比。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
until ! pgrep -f "[m]1_market" > /dev/null; do sleep 15; done
grep -q "保存 MKT_PACE" $L/m1.log || { echo "m1 失败" > $L/fx_gbdt13.done; exit 1; }
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20,MKT_SEG,MKT_DAY,MKT_DISP,MKT_PACE --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag mk_d4t150 --save-json $L/h2_mk_d4t150.json > $L/mk_d4t150.log 2>&1 &
run --depth 4 --trees 130 --aux-w 0.25 --tag mk_d4t130 --save-json $L/h2_mk_d4t130.json > $L/mk_d4t130.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag mk_d5 --save-json $L/h2_mk_d5.json > $L/mk_d5.log 2>&1 &
run --depth 6 --trees 85 --aux-w 0.50 --tag mk_d6 --save-json $L/h2_mk_d6.json > $L/mk_d6.log 2>&1 &
wait
REFS=sib=d4t150a025s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py mk_d4t150 > $L/c24n.log 2>&1
REFS=sib=d4t130a025s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py mk_d4t130 >> $L/c24n.log 2>&1
REFS=sib=d5a040s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py mk_d5 >> $L/c24n.log 2>&1
REFS=sib=d6t85a050s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py mk_d6 >> $L/c24n.log 2>&1
echo done > $L/fx_gbdt13.done
