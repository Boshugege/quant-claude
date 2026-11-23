#!/usr/bin/env bash
# 注意：下面的 SRC、DST 是比赛期间服务器上的路径，自己使用时要改成自己的任务目录和归档目录。
# 把已结束的评分任务目录从 NVMe（~/jyc_scorer/jobs）搬到 HDD（/srv/media/jyc_scorer/jobs_archive），原位置留绝对路径符号链接。
# 容器里把 HDD 归档目录挂在同一路径（Quadlet: Volume=/srv/media/jyc_scorer/jobs_archive:/srv/media/jyc_scorer/jobs_archive），
# 所以宿主机和容器里链接都能解析，评分机照常读写（赋分表、重评、配对比较都不受影响，只是读得慢一点）。
# 只搬 status 为 done / failed / cancelled 且结束超过 AGE_H 小时的任务；运行中、排队中的不碰。
# 用法：bash archive_jobs.sh [AGE_H=12]   由 jyc-archive.timer 每 6 小时运行一次；NVMe 剩余 < 25 GB 时 AGE_H 自动降为 0。
set -euo pipefail
SRC=$HOME/jyc_scorer/jobs
DST=/srv/media/jyc_scorer/jobs_archive
AGE_H=${1:-12}
mkdir -p "$DST"
free_gb=$(df --output=avail -BG "$SRC" | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt 25 ]; then AGE_H=0; echo "NVMe 剩余 ${free_gb}G < 25G，本次不看时间、搬全部已结束任务"; fi
python3 - "$SRC" "$AGE_H" <<'PY' > /tmp/jyc_archive_list.txt
import json, os, sys, time
src, age_h = sys.argv[1], float(sys.argv[2])
now = time.time()
for j in sorted(os.listdir(src)):
    d = os.path.join(src, j)
    if os.path.islink(d) or not os.path.isdir(d):
        continue
    try:
        s = json.load(open(os.path.join(d, "state.json")))
    except Exception:
        continue
    if s.get("status") not in ("done", "failed", "cancelled"):
        continue
    if now - os.path.getmtime(os.path.join(d, "state.json")) < age_h * 3600:
        continue
    print(j)
PY
n=0; mb=0
while read -r j; do
    [ -n "$j" ] || continue
    s="$SRC/$j"; t="$DST/$j"
    rsync -a --delete "$s/" "$t/"
    a=$(find "$s" -type f | wc -l); b=$(find "$t" -type f | wc -l)
    if [ "$a" != "$b" ]; then echo "校验失败，跳过 $j（$a vs $b）"; continue; fi
    sz=$(du -sm "$s" | cut -f1)
    mv "$s" "$s.__moving" && ln -s "$t" "$s" && rm -rf "$s.__moving"
    n=$((n+1)); mb=$((mb+sz))
done < /tmp/jyc_archive_list.txt
echo "$(date '+%F %T') 归档 $n 个任务，${mb} MB；NVMe 剩余 $(df --output=avail -BG "$SRC" | tail -1 | tr -d ' ')"
