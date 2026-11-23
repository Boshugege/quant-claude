#!/bin/bash
# T11（10-22，用户："充分利用一下两边的资源"）：公榜上换手最值钱，重缓冲（β 1.5～2.0）下多空腿基本在 10:00 就定了，之后几个截面都靠它。
#   a  一档规模比（QS1L 截面前 10 分钟、QS1A 整段）进 1617 的树（线性附加时它是唯一不推高换手的增量）
#   r1 目标改成"当前标签 + w·同日之后全部截面标签之和"（--aux-rest），权重同 1617（0.25／0.40）：学持续性
#   r2 同 r1，权重加倍（0.5／0.8）
# 1617 的三种配置、种子 7，与不加改动的同胞比；导出 2023–24 平均值，交评分机加规模比 + 缓冲带。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY m26_queue_size.py > $L/m26.log 2>&1 || { echo m26 失败 >> $L/m26.log; exit 1; }
BASEX=L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
run --extra $BASEX,QS1L,QS1A --depth 4 --trees 150 --aux-w 0.25 --tag t11a_d4t150 > $L/t11a_d4t150.log 2>&1 &
run --extra $BASEX,QS1L,QS1A --depth 4 --trees 130 --aux-w 0.25 --tag t11a_d4t130 > $L/t11a_d4t130.log 2>&1 &
run --extra $BASEX,QS1L,QS1A --depth 5 --aux-w 0.40 --tag t11a_d5 > $L/t11a_d5.log 2>&1 &
run --extra $BASEX --aux-rest --depth 4 --trees 150 --aux-w 0.25 --tag t11r1_d4t150 > $L/t11r1_d4t150.log 2>&1 &
run --extra $BASEX --aux-rest --depth 4 --trees 130 --aux-w 0.25 --tag t11r1_d4t130 > $L/t11r1_d4t130.log 2>&1 &
run --extra $BASEX --aux-rest --depth 5 --aux-w 0.40 --tag t11r1_d5 > $L/t11r1_d5.log 2>&1 &
run --extra $BASEX --aux-rest --depth 4 --trees 150 --aux-w 0.50 --tag t11r2_d4t150 > $L/t11r2_d4t150.log 2>&1 &
run --extra $BASEX --aux-rest --depth 4 --trees 130 --aux-w 0.50 --tag t11r2_d4t130 > $L/t11r2_d4t130.log 2>&1 &
run --extra $BASEX --aux-rest --depth 5 --aux-w 0.80 --tag t11r2_d5 > $L/t11r2_d5.log 2>&1 &
wait
rm -f $L/c24t11.log
for t in t11a t11r1 t11r2; do
  REFS=sib3=d4t150a025s+d4t130a025s+d5a040s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py ${t}_d4t150+${t}_d4t130+${t}_d5 >> $L/c24t11.log 2>&1
  PYTHONPATH=. PYTHONWARNINGS=ignore $PY c108_export_trees.py $t ${t}_d4t150+${t}_d4t130+${t}_d5 >> $L/c24t11.log 2>&1
done
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c108_export_trees.py sib3 d4t150a025s+d4t130a025s+d5a040s >> $L/c24t11.log 2>&1
echo done > $L/fx_t11.done
