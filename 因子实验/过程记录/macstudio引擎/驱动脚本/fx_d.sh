#!/bin/bash
# 方向 D（10-20）：涨跌停机制。m13 生成信号与事件研究；c41 评估单独五窗口 + 叠到 1610／1603。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY m13_limit.py > $L/m13.log 2>&1
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c41_book_eval.py LU_AT,LU_BROKE,LU_DIST,LD_AT,LD_BROKE,LD_DIST,LU_Y,LD_Y > $L/c64.log 2>&1
echo done > $L/fx_d.done
