#!/bin/bash
# 可解释信号库第二批（10-20）：m9 生成 10 个信号，c41 评估单独五窗口 + 叠到 1610／1603。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY m9_knowledge2.py > $L/m9.log 2>&1
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c41_book_eval.py K_MAX20,K_IVOL20,K_ILLIQ20,K_OVN20,K_INTRA20,K_REV20,K_PVC20,K_SKEW20,I_GAP,I_INTRA > $L/c57.log 2>&1
echo done > $L/fx_k2.done
