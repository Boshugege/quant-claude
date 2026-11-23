#!/bin/bash
# 新信息源（10-20）：开盘集合竞价。m14 生成 6 个信号；c41 评估单独五窗口 + 叠到 1610／1603。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY m14_auction.py > $L/m14.log 2>&1
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c41_book_eval.py AU_IMB,AU_LATE,AU_EARLY,AU_MGROW,AU_VOLR,AU_SIZE > $L/c65.log 2>&1
echo done > $L/fx_au.done
