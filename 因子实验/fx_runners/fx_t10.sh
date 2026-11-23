#!/bin/bash
# T10（10-21 晚，用户："1617 的逻辑你能进一步思考和批判吗？冲刺确实不错，再看看有没有更好的优化方式，不要定论"）：
# 1617 是 27 个现成信号的排名堆叠，看不到超短线信息与市况。把冲刺类特征（及时间锚点、市况）直接放进树里重训，
# 1617 的三种配置（深度 4 ×150／×130 棵多期 0.25、深度 5 多期 0.40），种子 7，与不加新特征的同胞比 2023／2024：
#   a 冲刺类：SP_LSDP（节奏按已过段数）、SP_LSDV（VWAP 版）、SP_ACC（加速度）、SP_CLV（段内收盘位置）
#   b a + 时间锚点 SP_A1000      c a + 市况（MKT_SEG、MKT_DISP、MKT_DDISP，原值不排名；多期目标不变）
cd /Users/user/Desktop/work/quant/fx/root/home/user/quant-claude/因子实验
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
L=/Users/user/Desktop/work/quant/fx/root/logs
PYTHONPATH=. PYTHONWARNINGS=ignore $PY m25_sprint_regime.py > $L/m25.log 2>&1 || { echo m25 失败 >> $L/m25.log; exit 1; }
BASEX=L_5,VSH20,PACE20,DOW10,CLS20,CLOSEPOS20
run() { PYTHONPATH=. PYTHONWARNINGS=ignore $PY h2_robust.py --drop N_dir,Y_sd5,X_n5d --h1363-pf --h1363-sub Y_sd5,X_n5d --save --quiet-trees --aux-n 2 "$@"; }
for arm in "a SP_LSDP,SP_LSDV,SP_ACC,SP_CLV" "b SP_LSDP,SP_LSDV,SP_ACC,SP_CLV,SP_A1000" "c SP_LSDP,SP_LSDV,SP_ACC,SP_CLV,MKT_SEG,MKT_DISP,MKT_DDISP"; do
  set -- $arm; t=$1; X=$BASEX,$2
  run --extra $X --depth 4 --trees 150 --aux-w 0.25 --tag t10${t}_d4t150 > $L/t10${t}_d4t150.log 2>&1 &
  run --extra $X --depth 4 --trees 130 --aux-w 0.25 --tag t10${t}_d4t130 > $L/t10${t}_d4t130.log 2>&1 &
  run --extra $X --depth 5 --aux-w 0.40 --tag t10${t}_d5 > $L/t10${t}_d5.log 2>&1 &
  wait
  rm -f $L/c24t10$t.log
  for p in "t10${t}_d4t150 d4t150a025s" "t10${t}_d4t130 d4t130a025s" "t10${t}_d5 d5a040s"; do set -- $p; REFS=sib=$2 PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py $1 >> $L/c24t10$t.log 2>&1; done
  REFS=sib3=d4t150a025s+d4t130a025s+d5a040s PYTHONPATH=. PYTHONWARNINGS=ignore $PY c24b_tree_ref.py t10${t}_d4t150+t10${t}_d4t130+t10${t}_d5 >> $L/c24t10$t.log 2>&1
done
echo done > $L/fx_t10.done
