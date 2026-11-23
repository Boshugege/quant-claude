# -*- coding: utf-8 -*-
"""第 3 级提速（任务书 §4.2）：不写合成头部代码，直接从评测机任务的 factor_raw.parquet
离线合成 B_w = rank(H) + w·rank(X)（09:30 截面置 0），写成三列 parquet，
用 kind=values 提交（baseline 设为 490 的任务号）。几十秒出五项指标与配对检验。

用法：
  build_values_composite.py --x fm_1107 --ws 0.20 --tag 1107 --submit
  build_values_composite.py --x fm_1120 --ws 0.20,0.35,0.50 --tag 1120 --submit
  build_values_composite.py --x fm_1107 --ws 0.20 --tag 1107 --no-submit   # 只写文件
约束：一律剔除 09:30 截面（输出置 0）；H 默认 490 的任务。

运行必须用 /home/user/jyc_scorer/.venv/bin/python，并用 agent-run 包一层（§4.1）。
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

JOBS = [pathlib.Path("/home/user/jyc_scorer/jobs"),
        pathlib.Path("/srv/media/jyc_scorer/jobs_archive")]
URL = "http://127.0.0.1:46220/api/jobs"
H_JOB_ID = "1012-153145-160b"      # fm_490_indneu25，作为 baseline 与 H
H_NAME = "fm_490_indneu25"


def find_job(spec):
    """按任务号目录名或 spec.json 的 name 找到任务目录。
    同名任务可能跑过多轮（旧的是 8 槽位修复前的版本，含重复键/15:00 槽位），
    这里一律取"有 result.json 的、任务号最新的"那一轮，保证用的是交付版数据。"""
    p = pathlib.Path(spec)
    if p.is_absolute() and p.exists():
        return p
    for base in JOBS:
        if not base.exists():
            continue
        cand = base / spec
        if (cand / "factor_raw.parquet").exists():
            return cand
    hits = []
    for base in JOBS:
        if not base.exists():
            continue
        for sp in base.glob("*/spec.json"):
            try:
                j = json.load(open(sp, encoding="utf-8"))
            except Exception:
                continue
            if j.get("name") == spec and (sp.parent / "factor_raw.parquet").exists():
                hits.append(sp.parent)
    if not hits:
        raise SystemExit("找不到任务 %s 的 factor_raw.parquet" % spec)
    # 优先有 result.json 的，再取任务号最大（最新）
    hits.sort(key=lambda d: ((d / "result.json").exists(), d.name))
    return hits[-1]


def read_raw(spec):
    d = find_job(spec)
    f = pd.read_parquet(d / "factor_raw.parquet")
    f["date"] = pd.to_datetime(f["date"])
    f = f[["date", "instrument", "factor"]].copy()
    f["factor"] = pd.to_numeric(f["factor"], errors="coerce")
    return d, f


def pct_rank(df):
    g = df.groupby("date")["factor"]
    return (g.rank(pct=True) - 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default=H_NAME, help="头部任务（任务号或名字）")
    ap.add_argument("--x", required=True, help="新根因子任务（任务号或名字）")
    ap.add_argument("--ws", required=True, help="逗号分隔的权重，例如 0.20,0.35")
    ap.add_argument("--tag", required=True, help="输出与任务名标签，例如 1107")
    ap.add_argument("--outdir", default="/srv/media/jyc_scorer/vc_tmp",
                    help="合成 parquet 的落盘目录（默认放 HDD；/tmp 是 tmpfs，会计入 agent.slice 的 10 GB 内存，§72）")
    ap.add_argument("--baseline", default=H_JOB_ID, help="配对基线任务号（默认 490）")
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--rank-head", action="store_true", dest="rank_head",
                    help="底座先做截面分位排名居中再混合：B = rank(H) + w·rank(X)（第十一轮 S2，默认关）")
    ap.add_argument("--mode", default="raw", choices=["raw", "rank"],
                    help="raw：B=头部原始值+w·(X分位-0.5)（不破坏头部分布，推荐）；"
                         "rank：B=rank(头部)+w·rank(X)（任务书 §4.2 字面口径，自带约 -0.0028 基线损失）")
    ap.add_argument("--ewma", type=float, default=0.0, help=">0 时先对 X 在同一股票连续截面上做 EWMA(该 α)，再排名合成（W1 降换手）")
    ap.add_argument("--inter", action="store_true", help="W2 交互口径：B=H原始值·(1+w·rank(X))（新根当开关，不当加数）")
    ap.add_argument("--x2", default="", help="第二个新根任务名（W5 双新根组合）")
    ap.add_argument("--w2s", default="", help="第二个新根的权重，逗号分隔（与 --ws 逐一配对）")
    ap.add_argument("--x3", default="", help="第三个新根任务名（三根组合）")
    ap.add_argument("--w3s", default="", help="第三个新根的权重，逗号分隔（与 --ws 逐一配对）")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()

    do_submit = args.submit and not args.no_submit
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    hd, hdf = read_raw(args.h)
    xd, xdf = read_raw(args.x)
    print("H=%s(%s) rows=%d ; X=%s(%s) rows=%d ; 构造=%s" % (args.h, hd.name, len(hdf), args.x, xd.name, len(xdf), args.mode))

    if args.ewma and args.ewma > 0:
        # 同一股票按时间连续的 EWMA（跨日连续），只用于降低 X 的换手；严格因果
        xdf = xdf.sort_values(["instrument", "date"]).reset_index(drop=True)
        a = float(args.ewma)
        sm = np.empty(len(xdf))
        cur = np.nan
        prev = None
        fac = xdf["factor"].to_numpy(float)
        insts = xdf["instrument"].to_numpy()
        for k in range(len(xdf)):
            if insts[k] != prev:
                cur = np.nan; prev = insts[k]
            v = fac[k]
            if np.isfinite(v):
                cur = v if not np.isfinite(cur) else (a * cur + (1 - a) * v)
            sm[k] = cur
        xdf["factor"] = sm
    xdf["rx"] = pct_rank(xdf)
    # 头部项：raw 模式保留头部原始值（排名会让平台 IC 从 0.07716 掉到 0.07438）；rank 模式才是任务书字面口径
    hdf["rh"] = hdf["factor"] if args.mode == "raw" else pct_rank(hdf)
    if args.rank_head:                                                   # 第十一轮 S2：底座先排名居中（默认关，改动不影响原行为）
        hdf["rh"] = pct_rank(hdf)

    x2df = None
    if args.x2:
        x2d, x2df = read_raw(args.x2)
        x2df["rx2"] = pct_rank(x2df)
        print("X2=%s(%s) rows=%d" % (args.x2, x2d.name, len(x2df)))
    x3df = None
    if args.x3:
        x3d, x3df = read_raw(args.x3)
        x3df["rx3"] = pct_rank(x3df)
        print("X3=%s(%s) rows=%d" % (args.x3, x3d.name, len(x3df)))
    m = hdf.merge(xdf[["date", "instrument", "rx"]], on=["date", "instrument"], how="inner")
    print("合成面板 rows=%d（交集）" % len(m))
    slot = m["date"].dt.strftime("%H%M")
    is930 = slot.eq("0930")
    print("09:30 行 %d（输出置 0）" % int(is930.sum()))

    # 补回 H 有、X 没有的行（用 H 单独值，权重部分置 0；保持 H 的覆盖面，避免无谓丢行）
    extra = hdf.merge(xdf[["date", "instrument"]], on=["date", "instrument"], how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"][["date", "instrument", "rh"]]
    extra["rx"] = 0.0
    base = pd.concat([m[["date", "instrument", "rh", "rx"]], extra], ignore_index=True)
    if x2df is not None:
        base = base.merge(x2df[["date", "instrument", "rx2"]], on=["date", "instrument"], how="left")
        base["rx2"] = base["rx2"].fillna(0.0)
    if x3df is not None:
        base = base.merge(x3df[["date", "instrument", "rx3"]], on=["date", "instrument"], how="left")
        base["rx3"] = base["rx3"].fillna(0.0)
    base930 = base["date"].dt.strftime("%H%M").eq("0930")
    print("最终输出面板 rows=%d（含 H 独有 %d，09:30 %d）" % (len(base), len(extra), int(base930.sum())))

    made = []
    w2s = [float(x) for x in args.w2s.split(",")] if args.w2s else []
    for iw, w in enumerate([float(s) for s in args.ws.split(",")]):
        if args.inter:
            v = base["rh"].to_numpy(float) * (1.0 + w * base["rx"].to_numpy(float))
        else:
            v = base["rh"].to_numpy(float) + w * base["rx"].to_numpy(float)
            if x2df is not None:
                w2 = w2s[iw] if iw < len(w2s) else w2s[-1]
                v = v + w2 * base["rx2"].to_numpy(float)
            if x3df is not None and args.w3s:
                w3s = [float(x) for x in args.w3s.split(",")]
                w3 = w3s[iw] if iw < len(w3s) else w3s[-1]
                v = v + w3 * base["rx3"].to_numpy(float)
        v[base930.to_numpy()] = 0.0
        o = pd.DataFrame({"date": base["date"].to_numpy(), "instrument": base["instrument"].to_numpy(),
                          "factor": np.nan_to_num(v, nan=0.0)})
        tagn = args.tag + (("_ewma%d" % int(args.ewma * 100)) if args.ewma else "") + ("_inter" if args.inter else "")
        if x3df is not None and args.w3s:
            _w3 = [float(x) for x in args.w3s.split(",")]
            tagn += "_w3" + ("%+.3f" % (_w3[iw] if iw < len(_w3) else _w3[-1])).replace("+", "p").replace("-", "m").replace(".", "")
        if x2df is not None and w2s:
            tagn += "_w2" + ("%+.3f" % (w2s[iw] if iw < len(w2s) else w2s[-1])).replace("+", "p").replace("-", "m").replace(".", "")
        name = "v_%s_%s_w%s" % (tagn, args.mode,
                               ("%+.3f" % w).replace("+", "p").replace("-", "m").replace(".", ""))
        f = out / (name + ".parquet")
        o.to_parquet(f, index=False)
        made.append((name, w, f))
        print("写出 %s  rows=%d" % (f, len(o)))

    if do_submit:
        for name, w, f in made:
            cmd = ["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values",
                   "-F", "start=%s" % args.start, "-F", "end=%s" % args.end,
                   "-F", "name=%s" % name, "-F", "baseline=%s" % args.baseline, URL]
            r = subprocess.run(cmd, capture_output=True, text=True)
            print("提交 %s -> %s" % (name, r.stdout.strip() or r.stderr.strip()))
            if '"job_id"' in r.stdout:                                  # 评测机已收下输入（任务目录里有副本），本地文件删掉
                f.unlink(missing_ok=True)
    return made


if __name__ == "__main__":
    main()
