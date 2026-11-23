#!/bin/bash
# 10-20：c59 快信号 + 慢锚成对叠加；T5 微观结构特征（快 6 + 慢 4）进树，三配置、种子 7，与同胞比 2023／2024。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c59_micro_pair.py > $L/c59.log 2>&1 &
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20,MS_OFI,MS_SMART,MS_AC1,MS_JUMP,MS_AMI,MS_VR,MS_AC1_20,MS_JUMP_20,MS_VR_20,MS_AMI_20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
run --depth 4 --trees 150 --aux-w 0.25 --tag ms_d4t150 > $L/ms_d4t150.log 2>&1 &
run --depth 4 --trees 130 --aux-w 0.25 --tag ms_d4t130 > $L/ms_d4t130.log 2>&1 &
run --depth 5 --aux-w 0.40 --tag ms_d5 > $L/ms_d5.log 2>&1 &
wait
rm -f $L/c24ms.log
for p in "ms_d4t150 d4t150a025s" "ms_d4t130 d4t130a025s" "ms_d5 d5a040s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24ms.log 2>&1; done
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py ms_d4t150+ms_d4t130+ms_d5 >> $L/c24ms.log 2>&1
echo done > $L/fx_k3b.done
