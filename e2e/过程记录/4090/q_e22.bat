@echo off
cd /d D:\quant-claude
set QUANT_ROOT=D:\quant-claude
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
echo %date% %time% start F72_2020-24_E22_gru_leg01_s48 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run F72_2020-24_E22_gru_leg01_s48 --arch e04 --norm rel2 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --win-groups shape,book,tod --fuse --aux-w 0.5 --aux-n 2 --epochs 6 --n-models 1 --train 2020-01-01 2024-12-31 --val 2024-10-01 2024-12-31 --secs-per-batch 4 --no-analyze --weight-ema 0.995 --target cs_rank --mtl-w 0.5 --pair-units --leg-turn 0.1 --seq-enc gru --seed 48 > e2e\runs\F72_2020-24_E22_gru_leg01_s48.out 2>&1
echo %date% %time% end F72_2020-24_E22_gru_leg01_s48 >> e2e\runs\q4090.log
echo %date% %time% start F73_2020-24_E22_gru_leg01_s49 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run F73_2020-24_E22_gru_leg01_s49 --arch e04 --norm rel2 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --win-groups shape,book,tod --fuse --aux-w 0.5 --aux-n 2 --epochs 6 --n-models 1 --train 2020-01-01 2024-12-31 --val 2024-10-01 2024-12-31 --secs-per-batch 4 --no-analyze --weight-ema 0.995 --target cs_rank --mtl-w 0.5 --pair-units --leg-turn 0.1 --seq-enc gru --seed 49 > e2e\runs\F73_2020-24_E22_gru_leg01_s49.out 2>&1
echo %date% %time% end F73_2020-24_E22_gru_leg01_s49 >> e2e\runs\q4090.log
echo %date% %time% E22 gru done >> e2e\runs\q4090.log
