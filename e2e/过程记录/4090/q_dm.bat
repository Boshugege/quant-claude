@echo off
cd /d D:\quant-claude
set QUANT_ROOT=D:\quant-claude
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
echo %date% %time% start DM >> e2e\runs\q4090.log
env\python.exe e2e\scripts\head_mix.py e2e/runs/R41o_leg01 0.1,0.2,0.3 daily > e2e\runs\DM_R41o_leg01.out 2>&1
env\python.exe e2e\scripts\head_mix.py e2e/runs/R43a_leg01_s43 0.1,0.2,0.3 daily > e2e\runs\DM_R43a_leg01_s43.out 2>&1
set TAG=DM& env\python.exe e2e\scripts\cmp_runs.py L01=e2e/runs/R41o_leg01 L01d1=e2e/runs/R41o_leg01/mix_w0.1 L01d2=e2e/runs/R41o_leg01/mix_w0.2 L01d3=e2e/runs/R41o_leg01/mix_w0.3 S43=e2e/runs/R43a_leg01_s43 S43d1=e2e/runs/R43a_leg01_s43/mix_w0.1 S43d2=e2e/runs/R43a_leg01_s43/mix_w0.2 S43d3=e2e/runs/R43a_leg01_s43/mix_w0.3 > e2e\runs\DM_cmp.out 2>&1
echo %date% %time% end DM >> e2e\runs\q4090.log
