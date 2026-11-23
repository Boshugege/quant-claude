#!/bin/bash
# S24／S25 补充（只推理）：S25 尾盘段通道 + 推理式日线锚；S24b 学习式锚之上再加推理式锚。对照 P12 配方（S22b + 推理锚 0.1）。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date '+%m-%d %T') S25 补充开始（只推理）" >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/head_mix.py e2e/runs/S25b_film_cls 0.1,0.2 daily 2>&1 | grep -v -i warn | grep -E "相关|done" >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/head_mix.py e2e/runs/S25a_cls 0.1 daily 2>&1 | grep -v -i warn | grep -E "相关|done" >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/head_mix.py e2e/runs/S24b_film_anch01 0.1,0.2,0.3 daily 2>&1 | grep -v -i warn | grep -E "相关|done" >> $LIVE
echo "==================== $(date '+%m-%d %T') 两折对比 S25 补充" >> $LIVE
TAG=S25p PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py base_infer01=e2e/runs/S20b_ema995/dmix_w0.1 S25a_infer01=e2e/runs/S25a_cls/dmix_w0.1 \
    film_infer01=e2e/runs/S22b_tod_film/dmix_w0.1 S25b=e2e/runs/S25b_film_cls S25b_infer01=e2e/runs/S25b_film_cls/dmix_w0.1 S25b_infer02=e2e/runs/S25b_film_cls/dmix_w0.2 \
    S24b_w01=e2e/runs/S24b_film_anch01/dmix_w0.1 S24b_w02=e2e/runs/S24b_film_anch01/dmix_w0.2 S24b_w03=e2e/runs/S24b_film_anch01/dmix_w0.3 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date '+%m-%d %T') S25 补充完成" >> $LIVE
