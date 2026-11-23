@echo off
cd /d D:\quant-claude
set QUANT_ROOT=D:\quant-claude
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
echo %date% %time% start R47a_gru_leg01_s42 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R47a_gru_leg01_s42 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --leg-turn 0.1 --seq-enc gru --seed 42 > e2e\runs\R47a_gru_leg01_s42.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R47a_gru_leg01_s42 2024 >> e2e\runs\R47a_gru_leg01_s42.out 2>&1
echo %date% %time% end R47a_gru_leg01_s42 >> e2e\runs\q4090.log
echo %date% %time% start R47b_gru_leg01_s43 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R47b_gru_leg01_s43 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --leg-turn 0.1 --seq-enc gru --seed 43 > e2e\runs\R47b_gru_leg01_s43.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R47b_gru_leg01_s43 2024 >> e2e\runs\R47b_gru_leg01_s43.out 2>&1
echo %date% %time% end R47b_gru_leg01_s43 >> e2e\runs\q4090.log
echo %date% %time% R47 queue done >> e2e\runs\q4090.log
