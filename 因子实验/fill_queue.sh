#!/usr/bin/env bash
# 保持评测机队列不超过 15：把 1115-1159 中尚未提交过的修复版按序补投（评测机任务名 fm_<编号>）。
set -uo pipefail
BASE="${JYC_SCORER_URL:-http://127.0.0.1:46220}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/user/jyc_scorer/.venv/bin/python

snapshot() {
  curl -s "$BASE/api/jobs?limit=100" | $PY -c '
import json,sys
d=json.load(sys.stdin); js=d.get("jobs",d) if isinstance(d,dict) else d
act=[j for j in js if j.get("status") in ("queued","producing","running","pending","scoring")]
print(len(act))
print(" ".join(sorted(str(j.get("name")) for j in js)))
'
}

for round in $(seq 1 400); do
  out=$(snapshot); active=$(echo "$out" | head -1); names=$(echo "$out" | tail -1)
  if [ "${active:-99}" -lt 15 ]; then
    for n in $(seq 1115 1159); do
      case " $names " in *" fm_$n "*) continue;; esac
      f=$(ls "$ROOT/因子代码/${n}_"*.py 2>/dev/null | head -1)
      [ -z "$f" ] && continue
      r=$(curl -s -F "files=@$f" -F start=2023-01 -F end=2024-12 -F kind=platform -F "name=fm_${n}" -F baseline=ref:104 "$BASE/api/jobs")
      echo "$(date +%H:%M:%S) submit $n -> $r"
      names="$names fm_$n"; active=$((active+1))
      [ "$active" -ge 15 ] && break
    done
  fi
  sleep 30
done
