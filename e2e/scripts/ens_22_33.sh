#!/bin/bash
# 10-24：混合集成成员数（fold-A，CPU）：2 + 2（三种组合）对 3 + 3，决定 E16 之后是否值得再训两个全量成员
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
R=e2e/runs
M="R22a=$R/R22a_ep6_mtl05 R24c=$R/R24c_mtl05_s43 R24a=$R/R24a_mtl05_s44"
X="R27a=$R/R27a_extra7 R28a=$R/R28a_extra7_s43 R29a=$R/R29a_extra7_s44"
p() { for kv in $M $X; do [ "${kv%%=*}" = "$1" ] && echo "${kv#*=}"; done; }
mk() { out=$R/ens_$1; shift; [ -f $out/pred_2024.parquet ] || PYTHONWARNINGS=ignore $PY e2e/scripts/ens_pred.py $out $(for n in "$@"; do p $n; done) > /dev/null; }
mk mx22a R22a R24c R27a R28a
mk mx22b R22a R24a R27a R29a
mk mx22c R24c R24a R28a R29a
mk mx33 R22a R24c R24a R27a R28a R29a
TAG=ENS PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py mx22a=$R/ens_mx22a mx22b=$R/ens_mx22b mx22c=$R/ens_mx22c mx33=$R/ens_mx33 2>&1 | grep -v -i warn | grep -E "^ *[0-9]+ +202[34] "
