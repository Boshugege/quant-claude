#!/bin/bash
# T10d（10-21 晚）：s17 发现冲刺的有效成分是"最近 30 分钟反转 + 1/8 当天顺势"，T10 用的"按已过段数"版本反而弱。
# 把原始版冲刺 SP_LSD 与两个原始成分 SP_RSEG（本段反转）、SP_RDAY（当天收益反向）交给树，加速度、段内收盘位置照旧。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
until [ -f $L/fx_t10.done ]; do sleep 30; done
PYTHONPATH=. PYTHONWARNINGS=ignore $PY m25_sprint_regime.py > $L/m25b.log 2>&1 || exit 1
BASEX=L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
X=$BASEX,SP_LSD,SP_RSEG,SP_RDAY,SP_ACC,SP_CLV
run --extra $X --depth 4 --trees 150 --aux-w 0.25 --tag t10d_d4t150 > $L/t10d_d4t150.log 2>&1 &
run --extra $X --depth 4 --trees 130 --aux-w 0.25 --tag t10d_d4t130 > $L/t10d_d4t130.log 2>&1 &
run --extra $X --depth 5 --aux-w 0.40 --tag t10d_d5 > $L/t10d_d5.log 2>&1 &
wait
rm -f $L/c24t10d.log
for p in "t10d_d4t150 d4t150a025s" "t10d_d4t130 d4t130a025s" "t10d_d5 d5a040s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24t10d.log 2>&1; done
REFS=sib3=d4t150a025s+d4t130a025s+d5a040s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py t10d_d4t150+t10d_d4t130+t10d_d5 >> $L/c24t10d.log 2>&1
echo done > $L/fx_t10d.done
