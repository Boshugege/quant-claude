#!/bin/bash
# 新信息源（10-20）：盘口委托笔数。m15 生成信号；c41 评估单独五窗口 + 叠到 1610／1603。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY m15_queue.py > $L/m15.log 2>&1
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c41_book_eval.py OQ_NIMB,OQ_SIZE,OQ_DSIZE,OQ_AVG,OQ_NIMB_20,OQ_SIZE_20,OQ_DSIZE_20 > $L/c66.log 2>&1
echo done > $L/fx_oq.done
