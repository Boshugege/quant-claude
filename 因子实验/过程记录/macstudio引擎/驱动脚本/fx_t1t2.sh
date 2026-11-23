#!/bin/bash
# T1（10-19 夜）：1617 的 16 个配置与种子，拟合 2020–2024（交付用全量；公榜区间不与本地重叠）。T2：去掉 1363 合成值的树（三配置、种子 7），与同胞比 2023／2024。
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --extra L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20 --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
# T2 先开（3 个），与 T1 第一批并行
run --no-h1363 --depth 4 --trees 150 --aux-w 0.25 --tag noh_d4t150 > $L/noh_d4t150.log 2>&1 &
run --no-h1363 --depth 4 --trees 130 --aux-w 0.25 --tag noh_d4t130 > $L/noh_d4t130.log 2>&1 &
run --no-h1363 --depth 5 --aux-w 0.40 --tag noh_d5 > $L/noh_d5.log 2>&1 &
for s in 7 11 23 42; do
  t=$([ $s = 7 ] && echo d4t150a025s || echo t150s$s); run --fit 2020-2024 --depth 4 --trees 150 --aux-w 0.25 --seed $s --tag f24_$t --save-json $L/h2_f24_$t.json > $L/f24_$t.log 2>&1 &
  t=$([ $s = 7 ] && echo d4t130a025s || ([ $s = 42 ] && echo t130s42 || echo d4t130a025s_s$s)); run --fit 2020-2024 --depth 4 --trees 130 --aux-w 0.25 --seed $s --tag f24_$t --save-json $L/h2_f24_$t.json > $L/f24_$t.log 2>&1 &
done; wait
for s in 7 11 23 42 3 5 13 17; do t=$([ $s = 7 ] && echo d5a040s || echo d5a040s_s$s); run --fit 2020-2024 --depth 5 --aux-w 0.40 --seed $s --tag f24_$t --save-json $L/h2_f24_$t.json > $L/f24_$t.log 2>&1 & done; wait
rm -f $L/c24z.log
for p in "noh_d4t150 d4t150a025s" "noh_d4t130 d4t130a025s" "noh_d5 d5a040s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24z.log 2>&1; done
PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py noh_d4t150+noh_d4t130+noh_d5 >> $L/c24z.log 2>&1
echo done > $L/fx_t1t2.done
