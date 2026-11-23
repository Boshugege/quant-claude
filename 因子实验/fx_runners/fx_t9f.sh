#!/bin/bash
# T9f（10-20）：委托笔数版 16 模型（1617 的 16 个配置与种子 + OQ 六个特征），全量拟合 2020–2024，导出 JSON（交付 1627 用）。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20,OQ_NIMB,OQ_SIZE,OQ_DSIZE,OQ_NIMB_20,OQ_SIZE_20,OQ_DSIZE_20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 --fit 2020-2024 "$@"; }
for s in 7 11 23 42; do
  run --depth 4 --trees 150 --aux-w 0.25 --seed $s --tag oqf24_d4t150_s$s --save-json $L/h2_oqf24_d4t150_s$s.json > $L/oqf24_d4t150_s$s.log 2>&1 &
  run --depth 4 --trees 130 --aux-w 0.25 --seed $s --tag oqf24_d4t130_s$s --save-json $L/h2_oqf24_d4t130_s$s.json > $L/oqf24_d4t130_s$s.log 2>&1 &
done; wait
for s in 7 11 23 42 3 5 13 17; do run --depth 5 --aux-w 0.40 --seed $s --tag oqf24_d5_s$s --save-json $L/h2_oqf24_d5_s$s.json > $L/oqf24_d5_s$s.log 2>&1 & done; wait
echo done > $L/fx_t9f.done
