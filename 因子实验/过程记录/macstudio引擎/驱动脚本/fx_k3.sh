#!/bin/bash
# 可解释信号库第三批（10-20）：m10 1 分钟微观结构 6 个信号 + 20 日慢版本；c41 评估单独五窗口 + 叠到 1610／1603。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY m10_micro.py > $L/m10.log 2>&1
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c41_book_eval.py MS_VR,MS_AC1,MS_JUMP,MS_AMI,MS_SMART,MS_OFI,MS_VR_20,MS_AC1_20,MS_JUMP_20,MS_AMI_20,MS_SMART_20,MS_OFI_20 > $L/c58.log 2>&1
echo done > $L/fx_k3.done
