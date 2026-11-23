@echo off
cd /d D:\quant-claude
set QUANT_ROOT=D:\quant-claude
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
echo %date% %time% start R41b_leg02 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R41b_leg02 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --seed 42 --epochs 6 --pair-units --leg-turn 0.2 > e2e\runs\R41b_leg02.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R41b_leg02 2024 >> e2e\runs\R41b_leg02.out 2>&1
echo %date% %time% end R41b_leg02 >> e2e\runs\q4090.log
echo %date% %time% start R41a_leg015 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R41a_leg015 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --seed 42 --epochs 6 --pair-units --leg-turn 0.15 > e2e\runs\R41a_leg015.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R41a_leg015 2024 >> e2e\runs\R41a_leg015.out 2>&1
echo %date% %time% end R41a_leg015 >> e2e\runs\q4090.log
echo %date% %time% start R41o_leg01 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R41o_leg01 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --seed 42 --epochs 6 --pair-units --leg-turn 0.1 > e2e\runs\R41o_leg01.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R41o_leg01 2024 >> e2e\runs\R41o_leg01.out 2>&1
echo %date% %time% end R41o_leg01 >> e2e\runs\q4090.log
