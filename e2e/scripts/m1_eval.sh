#!/bin/bash
# macstudio：取回 4090 的 MLP 成员 M1，评估 四成员 Transformer + 树 G8 + MLP 的各种组合（各成员截面排名后等权，五分之一／六分之一权重）
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
R=e2e/runs
mkdir -p $R/M1_mlp
scp -q "4090-lan:D:/quant-claude/e2e/runs/M1_mlp/val_pred.parquet" "4090-lan:D:/quant-claude/e2e/runs/M1_mlp/pred_2024.parquet" $R/M1_mlp/ 2>/dev/null
T4="$R/R22a_ep6_mtl05 $R/R24c_mtl05_s43 $R/R27a_extra7 $R/R28a_extra7_s43"
PYTHONWARNINGS=ignore $PY e2e/scripts/ens_w.py $R/ens_t4 $T4 > /dev/null
PYTHONWARNINGS=ignore $PY e2e/scripts/ens_w.py $R/ens_t4m $T4 $R/M1_mlp:1 > /dev/null
PYTHONWARNINGS=ignore $PY e2e/scripts/ens_w.py $R/ens_t4g $T4 $R/G8_win12:1 > /dev/null
PYTHONWARNINGS=ignore $PY e2e/scripts/ens_w.py $R/ens_t4gm $T4 $R/G8_win12:1 $R/M1_mlp:1 > /dev/null
TAG=M1 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py M1=$R/M1_mlp G8=$R/G8_win12 t4=$R/ens_t4 t4m=$R/ens_t4m t4g=$R/ens_t4g t4gm=$R/ens_t4gm 2>&1 | grep -v -i warn | grep -E "^ *[0-9]+ +202[34] "
