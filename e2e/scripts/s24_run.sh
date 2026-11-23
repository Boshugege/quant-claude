#!/bin/bash
# S24（学习式日线锚）与 S25（尾盘段通道），fold-A，种子 42，EMA。串行训练、预测 2024、对比：
#   S24 对照 S23 的推理式锚（S20b + 0.1、S22b + 0.1）；S25 对照 S20b、S22b。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
cmp() { TAG=$1 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py "${@:2}" 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE; }
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S24/S25 开始（学习式日线锚、尾盘段通道）" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_S24.txt 1
for r in S24a_base_anch01 S24b_film_anch01 S25a_cls S25b_film_cls; do
    echo "==================== $(date '+%m-%d %T') 预测 2024：$r" >> $LIVE
    PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE
done
echo "==================== $(date '+%m-%d %T') 两折对比 S24（学习式锚 vs 推理式锚）" >> $LIVE
cmp S24 base=e2e/runs/S20b_ema995 base_infer01=e2e/runs/S20b_ema995/dmix_w0.1 S24a_learn01=e2e/runs/S24a_base_anch01 film=e2e/runs/S22b_tod_film film_infer01=e2e/runs/S22b_tod_film/dmix_w0.1 S24b_learn01=e2e/runs/S24b_film_anch01
echo "==================== $(date '+%m-%d %T') 两折对比 S25（尾盘段通道）" >> $LIVE
cmp S25 base=e2e/runs/S20b_ema995 S25a_cls=e2e/runs/S25a_cls film=e2e/runs/S22b_tod_film S25b_film_cls=e2e/runs/S25b_film_cls
echo "==================== $(date '+%m-%d %T') S24/S25 完成" >> $LIVE
