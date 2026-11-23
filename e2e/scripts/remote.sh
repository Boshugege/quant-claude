#!/bin/bash
# 在 macstudio 上运行 e2e 脚本：同步 e2e/{src,scripts,configs,submission} -> 远端 quant/e2e/，用 Conda quant-e2e 运行，取回 runs/ 与 reports/ 的轻量文件。
# 用法（任意目录）：bash e2e/scripts/remote.sh e2e/scripts/<脚本>.py [参数...]
# 远端只使用 ~/Desktop/work/quant；大文件（权重 json、缓存）不回传。
set -e
REMOTE=macstudio
RDIR=Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
E2E="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$E2E/.." && pwd)"
SCRIPT="$1"; shift
REL="${SCRIPT#*e2e/}"
rsync -a --exclude __pycache__ --exclude "*.json" "$E2E/src" "$E2E/scripts" "$E2E/configs" "$E2E/submission" "$REMOTE:$RDIR/e2e/"
rsync -a --exclude __pycache__ --exclude 结果 "$ROOT/工具/本地评分/" "$REMOTE:$RDIR/工具/本地评分/"
ssh -o BatchMode=yes "$REMOTE" "cd ~/$RDIR && YEAR=${YEAR:-2023} PYTHONUNBUFFERED=1 $PY e2e/$REL $*"
mkdir -p "$E2E/runs" "$E2E/reports"
rsync -a --exclude '*.json.big' --max-size=2m "$REMOTE:$RDIR/e2e/runs/" "$E2E/runs/" 2>/dev/null || true
rsync -a "$REMOTE:$RDIR/e2e/reports/" "$E2E/reports/" 2>/dev/null || true
