#!/bin/bash
# 10-27 T-B／T-C：1669 的 14 个附加项 + q 作为树的输入（树学它们的条件效应，而不是线性叠加）；T-C 再加委托笔数六列（T9）。
# 1617 的三种配置、种子 7，拟合 2020–22；导出 2023–24（2023 验证、2024 测试）交评分机走 1669 的后处理。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs; mkdir -p $L
BASEX=L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20
AX=A_s35_of_l10_rel,A_s20_ql_end,A_s16_l10_clv,A_s37_auc_m20,A_s37_auc_m5,A_s40_q14_s1,A_s40_q14_end,A_s40_of14,A_s5_c_abnvol,A_s18_abnamt_day,A_s25_ts_sv_l10,A_s16_l10_ret,A_s6_amt_5d20d,A_s10_am_drift,A_s20_ql_s1only
OQ=OQ_SIZE,OQ_SIZE_20,OQ_NIMB,OQ_NIMB_20,OQ_DSIZE,OQ_DSIZE_20
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
for t in tB:$BASEX,$AX tC:$BASEX,$AX,$OQ; do
  n=${t%%:*}; x=${t#*:}
  run --extra $x --depth 4 --trees 150 --aux-w 0.25 --tag ${n}_d4t150 > $L/${n}_d4t150.log 2>&1 &
  run --extra $x --depth 4 --trees 130 --aux-w 0.25 --tag ${n}_d4t130 > $L/${n}_d4t130.log 2>&1 &
  run --extra $x --depth 5 --aux-w 0.40 --tag ${n}_d5 > $L/${n}_d5.log 2>&1 &
done
wait
for n in tB tC; do PYTHONPATH=. PYTHONWARNINGS=ignore $PY c108_export_trees.py $n ${n}_d4t150+${n}_d4t130+${n}_d5 >> $L/fx_tb.log 2>&1; done
echo done > $L/fx_tb.done
