#!/bin/bash
# R37（10-25）：R36a 的腿换手损失 1.0 把换手从 0.60 压到 0.34～0.38，但 IC −0.017、IC_IR −0.43，太强；试 0.1／0.3。之后跑 G8／G9（树成员全部通道）。
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log; R=e2e/runs
until grep -q "R36 完成" $LIVE; do sleep 60; done
while pgrep -f "e2e/scripts/train_local.py|gbdt_member.py" >/dev/null; do sleep 30; done
echo "==================== $(date "+%m-%d %T") R37 开始（腿换手损失 0.1／0.3）" >> $LIVE
bash e2e/scripts/queue.sh $R/queue_R37.txt 1
for r in R37a_leg01 R37b_leg03; do
  [ -f $R/$r/transformer_model.json ] && [ ! -f $R/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py $R/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R37" >> $LIVE
A="R22a=$R/R22a_ep6_mtl05 R24a=$R/R24a_mtl05_s44 R24c=$R/R24c_mtl05_s43"
for r in R36a_leg1 R36c_pair0 R37a_leg01 R37b_leg03; do [ -f $R/$r/pred_2024.parquet ] && A="$A ${r%%_*}=$R/$r"; done
TAG=R37 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R37 完成" >> $LIVE
W=shape,book,book2,book3,tod,pos,state,cls,mkt,lead,dow,auc
GB_RICH=1 $PY e2e/scripts/gbdt_member.py G8_win12 --fields full --norm rel3 --tgt e16 --win $W --trees 800 --depth 8 --lr 0.03 --rows 0.5 --cols 0.7 --min-leaf 1000 > $R/G8.out 2>&1
bash e2e/scripts/g1_eval.sh G8_win12 > $R/G8_eval.txt 2>&1
GB_RICH=1 $PY e2e/scripts/gbdt_member.py G9_win12_d10 --fields full --norm rel3 --tgt e16 --win $W --trees 1000 --depth 10 --lr 0.02 --rows 0.5 --cols 0.5 --min-leaf 500 > $R/G9.out 2>&1
bash e2e/scripts/g1_eval.sh G9_win12_d10 > $R/G9_eval.txt 2>&1
