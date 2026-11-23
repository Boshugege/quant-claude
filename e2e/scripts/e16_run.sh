#!/bin/bash
# E16（10-23，用户："你都可以试试"）：E15 配方 + 6 轮 + 多任务头 0.5，2020–24 全量，4 个成员 = 2 个普通（种子 42、43）+ 2 个训练加 7 个中间时刻（种子 44、45）。
# 依据：fold-A 集成里混入中间时刻成员两年 IC +0.002、IC_IR +0.02～+0.03、压力 +0.02（reports/15 "输出形式与集成多样性"）。逐个训练，之后合并打包。
cd ~/Desktop/work/quant
LIVE=e2e/runs/live.log
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_E16.txt 1
echo "==================== $(date "+%m-%d %T") E16 训练完成" >> $LIVE
