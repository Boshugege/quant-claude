@echo off
cd /d D:\quant-claude
set QUANT_ROOT=D:\quant-claude
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
echo %date% %time% start R46a_tau008_s43 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R46a_tau008_s43 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --seed 43 --leg-turn 0.1 --leg-tau 0.08 > e2e\runs\R46a_tau008_s43.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R46a_tau008_s43 2024 >> e2e\runs\R46a_tau008_s43.out 2>&1
echo %date% %time% end R46a_tau008_s43 >> e2e\runs\q4090.log
echo %date% %time% start R46b_tau008_s44 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R46b_tau008_s44 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --seed 44 --leg-turn 0.1 --leg-tau 0.08 > e2e\runs\R46b_tau008_s44.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R46b_tau008_s44 2024 >> e2e\runs\R46b_tau008_s44.out 2>&1
echo %date% %time% end R46b_tau008_s44 >> e2e\runs\q4090.log
echo %date% %time% start R46c_leg01_s45 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R46c_leg01_s45 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --seed 45 --leg-turn 0.1 > e2e\runs\R46c_leg01_s45.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R46c_leg01_s45 2024 >> e2e\runs\R46c_leg01_s45.out 2>&1
echo %date% %time% end R46c_leg01_s45 >> e2e\runs\q4090.log
echo %date% %time% start R46d_leg015_s45 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R46d_leg015_s45 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --seed 45 --leg-turn 0.15 > e2e\runs\R46d_leg015_s45.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R46d_leg015_s45 2024 >> e2e\runs\R46d_leg015_s45.out 2>&1
echo %date% %time% end R46d_leg015_s45 >> e2e\runs\q4090.log
echo %date% %time% start R46e_tau008_s45 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R46e_tau008_s45 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --seed 45 --leg-turn 0.1 --leg-tau 0.08 > e2e\runs\R46e_tau008_s45.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R46e_tau008_s45 2024 >> e2e\runs\R46e_tau008_s45.out 2>&1
echo %date% %time% end R46e_tau008_s45 >> e2e\runs\q4090.log
echo %date% %time% start R46f_tau008_leg015_s42 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R46f_tau008_leg015_s42 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --seed 42 --leg-turn 0.15 --leg-tau 0.08 > e2e\runs\R46f_tau008_leg015_s42.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R46f_tau008_leg015_s42 2024 >> e2e\runs\R46f_tau008_leg015_s42.out 2>&1
echo %date% %time% end R46f_tau008_leg015_s42 >> e2e\runs\q4090.log
echo %date% %time% R46 queue done >> e2e\runs\q4090.log
