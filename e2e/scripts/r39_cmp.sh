#!/bin/bash
# macstudio 上运行：取回 4090 的 R39 结果，与 4090 上复跑的 R22a（X22a）及 macstudio 的 R22a 一起评
cd ~/Desktop/work/quant
R=e2e/runs
for r in R22a_4090 R39a_proxy R39b_none R39c_lr05 R39d_lr2 R39e_b8; do
  mkdir -p $R/$r
  scp -q "4090-lan:D:/quant-claude/e2e/runs/$r/val_pred.parquet" "4090-lan:D:/quant-claude/e2e/runs/$r/pred_2024.parquet" $R/$r/ 2>/dev/null
done
A="R22a=$R/R22a_ep6_mtl05 X22a=$R/R22a_4090"
for r in R39a_proxy R39b_none R39c_lr05 R39d_lr2 R39e_b8; do [ -f $R/$r/pred_2024.parquet ] && A="$A ${r%%_*}=$R/$r"; done
TAG=R39 PYTHONWARNINGS=ignore /opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -E "^ *[0-9]+ +202[34] "
