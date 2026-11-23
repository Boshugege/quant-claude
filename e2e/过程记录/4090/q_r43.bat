@echo off
cd /d D:\quant-claude
set QUANT_ROOT=D:\quant-claude
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
echo %date% %time% start R43a_leg01_s43 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R43a_leg01_s43 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --seed 43 --leg-turn 0.1 > e2e\runs\R43a_leg01_s43.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R43a_leg01_s43 2024 >> e2e\runs\R43a_leg01_s43.out 2>&1
echo %date% %time% end R43a_leg01_s43 >> e2e\runs\q4090.log
echo %date% %time% start R43b_leg015_s43 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R43b_leg015_s43 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --seed 43 --leg-turn 0.15 > e2e\runs\R43b_leg015_s43.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R43b_leg015_s43 2024 >> e2e\runs\R43b_leg015_s43.out 2>&1
echo %date% %time% end R43b_leg015_s43 >> e2e\runs\q4090.log
echo %date% %time% start R43c_leg01_s44 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R43c_leg01_s44 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --seed 44 --leg-turn 0.1 > e2e\runs\R43c_leg01_s44.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R43c_leg01_s44 2024 >> e2e\runs\R43c_leg01_s44.out 2>&1
echo %date% %time% end R43c_leg01_s44 >> e2e\runs\q4090.log
echo %date% %time% start R43d_leg015_s44 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R43d_leg015_s44 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --seed 44 --leg-turn 0.15 > e2e\runs\R43d_leg015_s44.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R43d_leg015_s44 2024 >> e2e\runs\R43d_leg015_s44.out 2>&1
echo %date% %time% end R43d_leg015_s44 >> e2e\runs\q4090.log
echo %date% %time% start R44a_leg01_tau008 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R44a_leg01_tau008 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --seed 42 --leg-turn 0.1 --leg-tau 0.08 > e2e\runs\R44a_leg01_tau008.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R44a_leg01_tau008 2024 >> e2e\runs\R44a_leg01_tau008.out 2>&1
echo %date% %time% end R44a_leg01_tau008 >> e2e\runs\q4090.log
echo %date% %time% start R44b_leg01_tau03 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R44b_leg01_tau03 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --epochs 6 --pair-units --seed 42 --leg-turn 0.1 --leg-tau 0.3 > e2e\runs\R44b_leg01_tau03.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R44b_leg01_tau03 2024 >> e2e\runs\R44b_leg01_tau03.out 2>&1
echo %date% %time% end R44b_leg01_tau03 >> e2e\runs\q4090.log
