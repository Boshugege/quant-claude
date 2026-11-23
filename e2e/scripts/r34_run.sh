#!/bin/bash
# R34（10-24 下午，用户："端到端追一下"）：集成多样性的新来源，种子 43，按"同种子替换一个成员"比较：
#   a 中间时刻网格平移 5 分钟（09:50…14:20），替换 R28a（原网格 09:45…，种子 43）；b 目标改回 cs_z（截面 z 分数），替换 R24c（cs_rank，种子 43）
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
R=e2e/runs
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date "+%m-%d %T") R34 开始（集成多样性：平移网格、cs_z 目标）" >> $LIVE
bash e2e/scripts/queue.sh $R/queue_R34.txt 1
for r in R34a_x7s_s43 R34b_csz_s43; do
  [ -f $R/$r/transformer_model.json ] && [ ! -f $R/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py $R/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
mk() { out=$R/ens_$1; shift; PYTHONWARNINGS=ignore $PY e2e/scripts/ens_pred.py $out "$@" > /dev/null; }
mk mx22a $R/R22a_ep6_mtl05 $R/R24c_mtl05_s43 $R/R27a_extra7 $R/R28a_extra7_s43
mk r34a $R/R22a_ep6_mtl05 $R/R24c_mtl05_s43 $R/R27a_extra7 $R/R34a_x7s_s43
mk r34b $R/R22a_ep6_mtl05 $R/R34b_csz_s43 $R/R27a_extra7 $R/R28a_extra7_s43
mk r34ab $R/R22a_ep6_mtl05 $R/R34b_csz_s43 $R/R27a_extra7 $R/R34a_x7s_s43
echo "==================== $(date "+%m-%d %T") 对比 R34（单模型 + 同种子替换成员的四成员集成）" >> $LIVE
TAG=R34 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py R28a=$R/R28a_extra7_s43 R34a=$R/R34a_x7s_s43 R24c=$R/R24c_mtl05_s43 R34b=$R/R34b_csz_s43 mx22a=$R/ens_mx22a r34a=$R/ens_r34a r34b=$R/ens_r34b r34ab=$R/ens_r34ab 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R34 完成" >> $LIVE
