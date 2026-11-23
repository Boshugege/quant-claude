#!/bin/bash
# 下载并解压比赛公开的 30 分钟 bar 数据（约 2020-01 ~ 2024-12，按月 feather 文件）。
# 用法：bash download_data.sh [目标目录，默认 ./data]
# 解压后目录结构：<目标目录>/cpt_jyc_2026_e2e_bar30m/YYYYMM.0.feather
# 若放在其他位置，运行前设置环境变量：export JYC_BAR30M_DIR=<目标目录>/cpt_jyc_2026_e2e_bar30m
set -e
DEST="${1:-$(cd "$(dirname "$0")" && pwd)/data}"
URL="https://bigquant-public-download.tos-cn-shanghai.volces.com/jyc2026/cpt_jyc_2026_e2e_bar30m.tar.gz"
mkdir -p "$DEST"
cd "$DEST"
curl -L --fail -o bar30m.tar.gz "$URL"
tar -xzf bar30m.tar.gz && rm bar30m.tar.gz
chmod 755 cpt_jyc_2026_e2e_bar30m && chmod 644 cpt_jyc_2026_e2e_bar30m/*   # 压缩包内文件权限不规范，统一修正
echo "完成：$(ls cpt_jyc_2026_e2e_bar30m | wc -l | tr -d ' ') 个文件 -> $DEST/cpt_jyc_2026_e2e_bar30m"
