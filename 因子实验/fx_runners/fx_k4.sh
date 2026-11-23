#!/bin/bash
# 可解释信号库第四批（10-20）：m11 同伴领先—滞后 4 个信号；c41 评估单独五窗口 + 叠到 1610／1603。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY m11_peer.py > $L/m11.log 2>&1
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c41_book_eval.py P_SEG,P_LAG,P_CUM,P_DAY > $L/c60.log 2>&1
echo done > $L/fx_k4.done
