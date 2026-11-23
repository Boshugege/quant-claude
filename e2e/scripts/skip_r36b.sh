#!/bin/bash
# R36b（腿换手损失 4.0）在 R36a 1.0 已明显过强后没有意义：开始后自动跳过
cd ~/Desktop/work/quant
until grep -q "开始 R36b" e2e/runs/live.log; do sleep 20; done
sleep 30
pkill -f "run R36b_leg4"
echo "==================== $(date "+%m-%d %T") R36b 手动跳过（权重 4 明显过强，R36a 1.0 已经 IC −0.017）" >> e2e/runs/live.log
