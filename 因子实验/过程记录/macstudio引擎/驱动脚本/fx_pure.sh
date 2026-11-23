#!/bin/bash
# 纯方向深挖（10-20，用户："不要就这样交公榜，再发掘发掘"）：四条线并行。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
export PYTHONPATH=. PYTHONWARNINGS=ignore
( $PY m16_ofi.py > $L/m16.log 2>&1; $PY c41_book_eval.py CF_OFI1,CF_OFI3,CF_DIV,CF_TICK,CF_OFI1_20,CF_OFI3_20,CF_TICK_20 > $L/c71.log 2>&1; echo done > $L/j1.done ) &
( $PY m17_queue2.py > $L/m17.log 2>&1; $PY c41_book_eval.py QL_NIMB23,QL_NIMBA,QL_ARR,QL_RTS,QL_LATE,QL_SLATE,QL_NIMBA_20,QL_ARR_20,QL_RTS_20 > $L/c72.log 2>&1; echo done > $L/j2.done ) &
( $PY c70_pure_smooth.py > $L/c70.log 2>&1; echo done > $L/j3.done ) &
( for kl in "10 60" "50 60" "20 20" "20 120"; do set -- $kl; K=$1 LB=$2 $PY m11c_peer_var.py > $L/m11c_k$1l$2.log 2>&1; done; $PY c73_peer_var_eval.py > $L/c73.log 2>&1; echo done > $L/j4.done ) &
wait
