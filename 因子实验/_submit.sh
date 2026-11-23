#!/usr/bin/env bash
# 提交评分的统一入口：在评测机本机（127.0.0.1:46220 可达）直接 curl，
# 在其它机器回落到走 ssh 的 工具/评分机/remote_score.sh。用法与 remote_score.sh 相同。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
URL="${JYC_SCORER_URL:-http://127.0.0.1:46220}"
if curl -s -m 2 -o /dev/null "$URL/api/jobs?limit=1"; then
  exec bash "$HERE/score_local.sh" "$@"
else
  exec bash "$ROOT/工具/评分机/remote_score.sh" "$@"
fi
