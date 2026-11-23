#!/bin/bash
# 新方向 D3（10-19）：新信息树——盘口深层（快 + 慢）+ 同时段季节性 + 截面状态 + 日内不变市况 + 价格同步性 + 时段哑变量；不用任何根与头部。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
X=BK_DIMB,BK_SLOPEI,BK_CONS,BK_RESIL,BK_AOS_DEV,BK_THIN,BK_DIMB_20,BK_SLOPEI_20,BK_CONS_20,BK_RESIL_20,BK_AOS_20,BK_THIN_20,L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20,MKT_SEG,MKT_DAY,MKT_DISP,MKT_PACE,MKD_RET5,MKD_VOL20,SYNC20
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra-only --extra $X --save --quiet-trees --aux-n 2 "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag nwi_d4t150 > $L/nwi_d4t150.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag nwi_d5 > $L/nwi_d5.log 2>&1 &
run --depth 6 --trees 85 --aux-w 0.50 --tag nwi_d6 > $L/nwi_d6.log 2>&1 &
wait
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py nwi_d4t150+nwi_d5+nwi_d6 > $L/c24x.log 2>&1
echo done > $L/fx_d3.done
