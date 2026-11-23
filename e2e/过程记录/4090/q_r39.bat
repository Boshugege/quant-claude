@echo off
cd /d D:\quant-claude
set QUANT_ROOT=D:\quant-claude
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
echo %date% %time% start R39a_proxy >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R39a_proxy --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --seed 42 --epochs 6 --train-neutral proxy > e2e\runs\R39a_proxy.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R39a_proxy 2024 >> e2e\runs\R39a_proxy.out 2>&1
echo %date% %time% end R39a_proxy >> e2e\runs\q4090.log
echo %date% %time% start R39b_none >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R39b_none --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --seed 42 --epochs 6 --train-neutral none > e2e\runs\R39b_none.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R39b_none 2024 >> e2e\runs\R39b_none.out 2>&1
echo %date% %time% end R39b_none >> e2e\runs\q4090.log
echo %date% %time% start R39c_lr05 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R39c_lr05 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --seed 42 --epochs 6 --train-neutral proxy_ind --lr 0.0005 > e2e\runs\R39c_lr05.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R39c_lr05 2024 >> e2e\runs\R39c_lr05.out 2>&1
echo %date% %time% end R39c_lr05 >> e2e\runs\q4090.log
echo %date% %time% start R39d_lr2 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R39d_lr2 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --seed 42 --epochs 6 --train-neutral proxy_ind --lr 0.002 > e2e\runs\R39d_lr2.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R39d_lr2 2024 >> e2e\runs\R39d_lr2.out 2>&1
echo %date% %time% end R39d_lr2 >> e2e\runs\q4090.log
echo %date% %time% start R39e_b8 >> e2e\runs\q4090.log
env\python.exe e2e\scripts\train_local.py --run R39e_b8 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --seed 42 --epochs 6 --train-neutral proxy_ind --secs-per-batch 8 > e2e\runs\R39e_b8.out 2>&1
env\python.exe e2e\scripts\pred_year.py e2e/runs/R39e_b8 2024 >> e2e\runs\R39e_b8.out 2>&1
echo %date% %time% end R39e_b8 >> e2e\runs\q4090.log
echo QUEUE DONE >> e2e\runs\q4090.log
env\python.exe e2e\scripts\pred_year.py e2e/runs/R22a_4090 2024 >> e2e\runs\R22a_4090.out 2>&1
echo R22a_4090 pred done >> e2e\runs\q4090.log
echo %date% %time% start M1_mlp >> e2e\runs\q4090.log
set GB_RICH=1
env\python.exe e2e\scripts\mlp_member.py M1_mlp --trees 600 --fields full --norm rel3 --tgt e16 --win shape,book,book2,book3,tod,pos,state,cls,mkt,lead,dow,auc > e2e\runs\M1_mlp.out 2>&1
echo %date% %time% end M1_mlp >> e2e\runs\q4090.log
