#!/bin/bash
# 纯方向深挖第二批（10-20，用户："多探索探索，把性能利用起来"）：J5 日内锚定、J6 平滑扫描、J7 同伴订单流，并行。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
export PYTHONPATH=. PYTHONWARNINGS=ignore
( $PY m18_anchor.py > $L/m18.log 2>&1; $PY c41_book_eval.py AN_VWAP,AN_RPOS,AN_MICRO,AN_SPRD,AN_CONC,AN_VWAP_20,AN_RPOS_20,AN_MICRO_20,AN_SPRD_20,AN_CONC_20 > $L/c77.log 2>&1; echo done > $L/j5.done ) &
( $PY c76_smooth_sweep.py > $L/c76.log 2>&1; echo done > $L/j6.done ) &
( $PY m19_peer_flow.py > $L/m19.log 2>&1; $PY c41_book_eval.py PF_QL_NIMB23,PF_OQ_SIZE,PF_MS_OFI,PF_CF_OFI3,PD_QL_NIMB23,PD_OQ_SIZE,PD_MS_OFI,PD_CF_OFI3 > $L/c78.log 2>&1; echo done > $L/j7.done ) &
wait
