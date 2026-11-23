#!/bin/bash
# 在远端 Mac Studio 上运行本地评分/研究脚本。
# 历史脚本：这台远端机器已在比赛结束后停用，本脚本只作为记录保留。本机直接运行 python3 工具/本地评分/run.py 即可。
# 用法（项目根目录）：bash 工具/本地评分/remote.sh <本机脚本路径> [脚本参数...]
# 流程：上传 工具/本地评分/ 代码与该脚本 -> 远端用独立虚拟环境运行 -> 取回 结果/ 目录。
# 远端只使用 ~/Desktop/work/quant，不触碰其他目录；数据已在远端 数据/raw_bar30m/ 下。
set -e
REMOTE=macstudio
RDIR=Desktop/work/quant
SCRIPT="$1"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
rsync -a --exclude __pycache__ --exclude 结果 "$HERE/" "$REMOTE:$RDIR/工具/本地评分/"
rsync -a "$HERE/结果/缓存/" "$REMOTE:$RDIR/工具/本地评分/结果/缓存/" 2>/dev/null || true
mkdir -p "$HERE/结果"
NAME="$(basename "$SCRIPT")"
SCRIPT_ABS="$(cd "$(dirname "$SCRIPT")" && pwd)/$NAME"
if [ "$SCRIPT_ABS" = "$HERE/$NAME" ]; then
    TARGET="工具/本地评分/$NAME"          # 本目录内的脚本（如 run.py）原位运行，结果写入 工具/本地评分/结果/
else
    rsync -a "$SCRIPT" "$REMOTE:$RDIR/jobs/$NAME"
    TARGET="jobs/$NAME"                  # 临时研究脚本放在 jobs/ 下运行
fi
ssh -o BatchMode=yes "$REMOTE" "cd ~/$RDIR && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 ONLY=\"${ONLY:-}\" PYTHONPATH=工具/本地评分 .venv/bin/python $TARGET $*"
rsync -a "$REMOTE:$RDIR/工具/本地评分/结果/" "$HERE/结果/"
