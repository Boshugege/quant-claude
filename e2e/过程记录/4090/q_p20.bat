@echo off
cd /d D:\quant-claude
set QUANT_ROOT=D:\quant-claude
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
echo %date% %time% start P20_tree >> e2e\runs\q4090.log
env\python.exe e2e\scripts\e17_add_gbdt.py e2e/runs/R37b_leg03 e2e/runs/R37b_leg03_g8 e2e/configs/g8_w025.json > e2e\runs\P20_tree.out 2>&1
echo %date% %time% end P20_tree >> e2e\runs\q4090.log
