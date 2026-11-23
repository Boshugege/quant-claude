#!/usr/bin/env bash
# 监控：打印已完成候选的队列剩余数量与配对比较排名（只读，不改动任何东西）。
# 用法：bash 因子实验/watch.sh [等待秒数]
cd "$(dirname "$0")/.." || exit 1
sleep "${1:-280}"
URL="${JYC_SCORER_URL:-http://127.0.0.1:46220}"
echo "=== 队列状态 ==="
curl -s -m 5 "$URL/api/jobs?limit=300" | python3 -c '
import json, re, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("  （读不到队列，可能是远端模式：请用 ssh personal 后本机运行）"); raise SystemExit
jobs = d if isinstance(d, list) else d.get("jobs", [])
pat = re.compile(r"^(fm_|1[2-9]\d_|19\d_|2\d\d_|probe|smoke)")
from collections import Counter
c = Counter(j.get("status") for j in jobs if pat.match(j.get("name") or ""))
print("  " + "  ".join("%s=%d" % kv for kv in sorted(c.items())) or "  （空）")
' 2>/dev/null
echo "=== 配对比较（综合百分位 z 排序，只信 paired） ==="
python3 因子实验/paired_table.py | head -25
