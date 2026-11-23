#!/usr/bin/env bash
# 在评测机本机提交评分（不需要 ssh；remote_score.sh 在本机不可用）。
# 用法：bash 因子实验/score_local.sh <文件...> [-- start=2023-01 end=2024-12 kind=platform name=fm_xxx baseline=ref:104]
# 结果取到 工具/评分机/结果/<任务号>/。可选环境变量 JYC_SCORER_URL（默认 http://127.0.0.1:46220）。
# factor_raw.parquet 默认不再下载（任务目录与 HDD 归档里都有原件；10-15 曾因此在 NVMe 上堆了 24 GB 副本）。需要时设 FETCH_RAW=1。
set -euo pipefail
BASE_URL="${JYC_SCORER_URL:-http://127.0.0.1:46220}"
OUT="${JYC_LOCAL_RESULTS:-$(cd "$(dirname "$0")/.." && pwd)/工具/评分机/结果}"
files=(); opts=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then shift; opts=("$@"); break; fi
  files+=("$1"); shift
done
args=()
for f in "${files[@]+"${files[@]}"}"; do args+=(-F "files=@$f"); done
for o in "${opts[@]+"${opts[@]}"}"; do args+=(-F "$o"); done
echo "提交到 $BASE_URL ……（等待完成，期间可在网页“任务”里看实时日志）"
JSON=$(curl -s "${args[@]}" "$BASE_URL/api/jobs?wait=true")
read -r JID STATUS < <(python3 -c 'import json,sys
d=json.loads(sys.stdin.read() or "{}")
print(d.get("id") or d.get("job_id") or "-", (d.get("status") or "") + " " + str(d.get("error") or d.get("detail") or ""))' <<<"$JSON")
if [[ "$JID" == "-" ]]; then echo "提交失败：$STATUS"; exit 1; fi
mkdir -p "$OUT/$JID"
for n in report.md result.json log.txt ${FETCH_RAW:+factor_raw.parquet}; do
  curl -s -o "$OUT/$JID/$n" "$BASE_URL/api/jobs/$JID/files/$n" || true
  [[ -s "$OUT/$JID/$n" ]] || rm -f "$OUT/$JID/$n"
done
echo "任务 $JID：$STATUS；结果在 $OUT/$JID/"
[[ -f "$OUT/$JID/report.md" ]] && cat "$OUT/$JID/report.md" || tail -30 "$OUT/$JID/log.txt"
