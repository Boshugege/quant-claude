#!/bin/bash
# 新信号源第三批（10-20 晚，用户："继续找新信号源，然后比较好的信号源集中起来想办法优化"）：N1 成交单笔大小拆分、N3 盘口挂单变化、N2 波动标准化反转 + N4 订单流季节性。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
export PYTHONPATH=. PYTHONWARNINGS=ignore
( $PY m22_tradesize.py > $L/m22.log 2>&1; SIGS=TS_BIG,TS_SMALL,TS_MID $PY m19_peer_flow.py > $L/m19c.log 2>&1; $PY c41_book_eval.py TS_BIG,TS_SMALL,TS_MID,TS_BIGSH,PD_TS_BIG,PD_TS_SMALL,PD_TS_MID > $L/c89.log 2>&1; echo done > $L/n1.done ) &
( $PY m23_queue_dyn.py > $L/m23.log 2>&1; $PY c41_book_eval.py QB_NET,QB_CNET,QB_CHURN > $L/c90.log 2>&1; echo done > $L/n3.done ) &
( $PY c87_volnorm_season.py > $L/c87.log 2>&1; echo done > $L/n24.done ) &
wait
