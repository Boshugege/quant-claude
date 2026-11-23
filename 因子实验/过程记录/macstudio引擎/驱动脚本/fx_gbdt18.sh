#!/bin/bash
# F14（10-19）：树的完整口径（成交笔数三根都活：H1363 不扣、Y_sd5／X_n5d／N_dir 都作特征），三个配置、种子 7，与同胞比（需平台安全写法才可交付）。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --save --quiet-trees --aux-n 2 "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag fu_d4t150 --save-json $L/h2_fu_d4t150.json > $L/fu_d4t150.log 2>&1 &
run --depth 4 --trees 130 --aux-w 0.25 --tag fu_d4t130 --save-json $L/h2_fu_d4t130.json > $L/fu_d4t130.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag fu_d5 --save-json $L/h2_fu_d5.json > $L/fu_d5.log 2>&1 &
wait
rm -f $L/c24s.log
for p in "fu_d4t150 d4t150a025s" "fu_d4t130 d4t130a025s" "fu_d5 d5a040s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24s.log 2>&1; done
echo done > $L/fx_gbdt18.done
