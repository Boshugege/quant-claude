#!/bin/bash
# 10-24：树模型成员 G1 单独、以及加进四成员 Transformer 集成（mx22a：R22a、R24c、R27a、R28a）的效果
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
R=e2e/runs; G=$R/${1:-G1_d6_t600}
T4="$R/R22a_ep6_mtl05 $R/R24c_mtl05_s43 $R/R27a_extra7 $R/R28a_extra7_s43"
PYTHONWARNINGS=ignore $PY e2e/scripts/ens_w.py $R/ens_t4 $T4 > /dev/null
PYTHONWARNINGS=ignore $PY e2e/scripts/ens_w.py $R/ens_t4g1 $T4 $G:1 > /dev/null
PYTHONWARNINGS=ignore $PY e2e/scripts/ens_w.py $R/ens_t4g2 $T4 $G:2 > /dev/null
PYTHONWARNINGS=ignore $PY e2e/scripts/ens_w.py $R/ens_t1g1 $R/R22a_ep6_mtl05 $G:1 > /dev/null
TAG=G1 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py G1=$G R22a=$R/R22a_ep6_mtl05 t1g1=$R/ens_t1g1 t4=$R/ens_t4 t4g1=$R/ens_t4g1 t4g2=$R/ens_t4g2 2>&1 | grep -v -i warn | grep -E "^ *[0-9]+ +202[34] "
