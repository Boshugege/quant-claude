#!/bin/bash
# T12（10-23 晚，用户："不要拘泥于小的微调"）：把新信息（订单流、收盘位置、收盘竞价惯性、规模比）作为特征交给 1617 的树，
# 让树自己学交互（14:30 方向相反等；1617 的特征里已有时段哑变量）。1617 的三种配置、种子 7，对照同胞 sib3。
#   a BASEX + QS1L、QS1A + OF10、CLV10、AUC5、AUC20     b BASEX + OF10、AUC5、AUC20（不含规模比）
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY m27_flow_auc.py > $L/m27.log 2>&1 || { echo m27 失败 >> $L/m27.log; exit 1; }
BASEX=L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
XA=$BASEX,QS1L,QS1A,OF10,CLV10,AUC5,AUC20
XB=$BASEX,OF10,AUC5,AUC20
run --extra $XA --depth 4 --trees 150 --aux-w 0.25 --tag t12a_d4t150 > $L/t12a_d4t150.log 2>&1 &
run --extra $XA --depth 4 --trees 130 --aux-w 0.25 --tag t12a_d4t130 > $L/t12a_d4t130.log 2>&1 &
run --extra $XA --depth 5 --aux-w 0.40 --tag t12a_d5 > $L/t12a_d5.log 2>&1 &
run --extra $XB --depth 4 --trees 150 --aux-w 0.25 --tag t12b_d4t150 > $L/t12b_d4t150.log 2>&1 &
run --extra $XB --depth 4 --trees 130 --aux-w 0.25 --tag t12b_d4t130 > $L/t12b_d4t130.log 2>&1 &
run --extra $XB --depth 5 --aux-w 0.40 --tag t12b_d5 > $L/t12b_d5.log 2>&1 &
wait
rm -f $L/c24t12.log
for t in t12a t12b; do
  REFS=sib3=d4t150a025s+d4t130a025s+d5a040s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py ${t}_d4t150+${t}_d4t130+${t}_d5 >> $L/c24t12.log 2>&1
  PYTHONPATH=. PYTHONWARNINGS=ignore $PY c108_export_trees.py $t ${t}_d4t150+${t}_d4t130+${t}_d5 >> $L/c24t12.log 2>&1
done
echo done > $L/fx_t12.done
