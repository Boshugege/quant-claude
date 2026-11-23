#!/usr/bin/env bash
# 把评测机上属于本轮探索、但本地还没取回的任务结果补齐。
# 在评测机本机直接 curl（含 factor_raw.parquet，供 values 类型重估）；在其它机器走 ssh + scp。
cd "$(dirname "$0")/.." || exit 1
HOST="${JYC_SCORER_HOST:-personal}"
BASE="${JYC_SCORER_REMOTE_DIR:-jyc_scorer}"
URL="${JYC_SCORER_URL:-http://127.0.0.1:46220}"
OUT="工具/评分机/结果"
LOCAL=0
if curl -s -m 2 -o /dev/null "$URL/api/jobs?limit=1"; then LOCAL=1; fi
if [ "$LOCAL" = 1 ]; then
  curl -s "$URL/api/jobs?limit=300" > /tmp/jyc_jobs_raw.json
else
  ssh -o BatchMode=yes "$HOST" "bash -lc \"curl -s 'http://127.0.0.1:46220/api/jobs?limit=300'\"" 2>/dev/null > /tmp/jyc_jobs_raw.json
fi
python3 -c '
import json, re, sys
d = json.load(open("/tmp/jyc_jobs_raw.json"))
jobs = d if isinstance(d, list) else d.get("jobs", [])
for j in jobs:
    nm = j.get("name") or ""
    if re.match(r"^(fm_|1[2-9]\d_|19\d_|2\d\d_|probe|smoke)", nm) and j.get("status") == "done":
        print(j["id"], nm)
' > /tmp/jyc_jobs.txt
while read -r id nm; do
  [ -z "$id" ] && continue
  if [ -s "$OUT/$id/result.json" ]; then continue; fi
  mkdir -p "$OUT/$id"
  if [ "$LOCAL" = 1 ]; then
    for n in report.md result.json log.txt ${FETCH_RAW:+factor_raw.parquet}; do
      curl -s -o "$OUT/$id/$n" "$URL/api/jobs/$id/files/$n" || true
      [ -s "$OUT/$id/$n" ] || rm -f "$OUT/$id/$n"
    done
  else
    for n in report.md result.json log.txt; do scp -q "$HOST:$BASE/jobs/$id/$n" "$OUT/$id/" 2>/dev/null || true; done
  fi
  echo "取回 $id $nm"
done < /tmp/jyc_jobs.txt
echo "取回完成"
