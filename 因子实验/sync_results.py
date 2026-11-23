# -*- coding: utf-8 -*-
"""把评测机上已结束任务的 report/result/log/factor_raw 同步到 工具/评分机/结果/<job_id>/。

用法：/home/user/jyc_scorer/.venv/bin/python 因子实验/sync_results.py [job_id|任务名 ...]
不带参数时同步所有状态为 done/failed/canceled 且本地还没有文件的任务。
"""
import json
import os
import pathlib
import sys
import urllib.request

BASE = os.environ.get("JYC_SCORER_URL", "http://127.0.0.1:46220")
OUT = pathlib.Path(__file__).resolve().parent.parent / "工具" / "评分机" / "结果"
FILES = ("report.md", "result.json", "log.txt", "factor_raw.parquet")
ENDED = ("done", "failed", "error", "canceled", "cancelled")


def main():
    want = set(sys.argv[1:])
    jobs = json.load(urllib.request.urlopen(BASE + "/api/jobs?limit=200"))
    jobs = jobs.get("jobs", jobs) if isinstance(jobs, dict) else jobs
    synced = 0
    for j in jobs:
        jid = j.get("id") or j.get("job_id")
        name = str(j.get("name") or "")
        if not jid or j.get("status") not in ENDED:
            continue
        if want and jid not in want and name not in want:
            continue
        d = OUT / jid
        d.mkdir(parents=True, exist_ok=True)
        got = []
        for f in FILES:
            p = d / f
            if p.exists() and p.stat().st_size > 0:
                continue
            try:
                data = urllib.request.urlopen("%s/api/jobs/%s/files/%s" % (BASE, jid, f)).read()
            except Exception:
                continue
            if data:
                p.write_bytes(data)
                got.append(f)
        synced += 1
        print("%s %s %s %s" % (name, jid, j.get("status"), ",".join(got) or "already"))
    print("synced_jobs=%d" % synced)


if __name__ == "__main__":
    main()
