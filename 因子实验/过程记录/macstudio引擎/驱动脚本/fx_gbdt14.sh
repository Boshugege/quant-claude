#!/bin/bash
# F11b（10-19）：日内不变的市况特征（MKD_RET1／RET5／VOL20／DISP5，只用前几日），与同胞（同配置、种子 7）直接比。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY m1_market.py > $L/m1b.log 2>&1
grep -q "保存 MKD_DISP5" $L/m1b.log || { echo "m1b 失败" > $L/fx_gbdt14.done; exit 1; }
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20,MKD_RET1,MKD_RET5,MKD_VOL20,MKD_DISP5 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag md_d4t150 --save-json $L/h2_md_d4t150.json > $L/md_d4t150.log 2>&1 &
run --depth 4 --trees 130 --aux-w 0.25 --tag md_d4t130 --save-json $L/h2_md_d4t130.json > $L/md_d4t130.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag md_d5 --save-json $L/h2_md_d5.json > $L/md_d5.log 2>&1 &
run --depth 6 --trees 85 --aux-w 0.50 --tag md_d6 --save-json $L/h2_md_d6.json > $L/md_d6.log 2>&1 &
wait
for p in "md_d4t150 d4t150a025s" "md_d4t130 d4t130a025s" "md_d5 d5a040s" "md_d6 d6t85a050s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24o.log 2>&1; done
echo done > $L/fx_gbdt14.done
