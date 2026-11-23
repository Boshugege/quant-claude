# -*- coding: utf-8 -*-
"""换手变换（10-21 晚，用户："换手不太行"）。评分机的换手 = 多头腿（排名 > 0.9）／空头腿（≤ 0.1）在**同一交易日相邻截面**之间的
替换比例（跨日不计）。所以压换手只需要让处在腿边界附近的股票"粘"一点，不必整体平滑。在底座（如 1617 的 2023–24 代码输出）上比较：
  ew<α>    跨截面 EWMA（过去的权重 α）
  ewd<α>   同上，但每个交易日重置（只在日内平滑；跨日不计换手，日内平滑够用）
  buf<β>   缓冲带：x_t = z_t + β·(1[上一截面在多头腿] − 1[上一截面在空头腿])（β 为 z 分数单位，buf30 = 0.3），腿用变换后的值判定（迟滞），每日重置
  ewdXbufY 日内 EWMA 后再加缓冲带
  q<W>…   加 W/100·z(最后 10 分钟委托结构，y_s3_q_mix_late)（在 a 之后）
  a<W>…   最前面先加 W/100·z(上午相对 10:00 的回吐，s10_am_drift，10:30–11:30)，再做后面的变换
全部在逐截面 z 分数上做、输出 z 分数（排名会让多空降约 3），09:30 置 0，交评分机（kind=values）。
用法：START=2023-01 END=2024-12 python c105_turnover.py <前缀> <底座任务1+底座任务2> <变换> [<变换> ...]"""
import glob
import json
import os
import re
import subprocess
import sys
import numpy as np
import pandas as pd

JOBS = os.path.expanduser("~/jyc_scorer/jobs")
OUTD = "/srv/media/jyc_scorer/incoming/turn"; os.makedirs(OUTD, exist_ok=True)
START, END = os.environ.get("START", "2023-01"), os.environ.get("END", "2024-12")
Z930 = []                                                                       # 09:30 截面原样置 0 附回（评分机检查 8 个截面的覆盖）


def job_dir(name):
    best = None
    for sp in glob.glob(os.path.join(JOBS, "*", "spec.json")):
        try:
            s = json.load(open(sp, encoding="utf-8"))
        except Exception:
            continue
        d = os.path.dirname(sp)
        if s.get("name") == name and os.path.exists(os.path.join(d, "factor_raw.parquet")):
            if best is None or d > best:
                best = d
    if best is None:
        raise FileNotFoundError(name)
    return best


def load(names):
    parts = [pd.read_parquet(os.path.join(job_dir(n), "factor_raw.parquet"))[["date", "instrument", "factor"]] for n in names]
    d = pd.concat(parts, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d = d.drop_duplicates(["date", "instrument"], keep="last")
    Z930.append(d[d["date"].dt.hour * 100 + d["date"].dt.minute == 930][["date", "instrument"]].assign(factor=0.0))
    d = d[d["date"].dt.hour * 100 + d["date"].dt.minute != 930].copy()
    g = d.groupby("date")["factor"]
    d["r"] = ((d["factor"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)    # 逐截面 z 分数（不用排名：排名会伤多空）
    d["day"] = d["date"].dt.normalize()
    return d.sort_values(["instrument", "date"]).reset_index(drop=True)


def ewma(d, a, daily, col="r"):
    v, ins, day = d[col].to_numpy(float), d["instrument"].to_numpy(), d["day"].to_numpy()
    out = np.empty_like(v)
    prev, li, ld = 0.0, None, None
    for i in range(len(v)):
        if ins[i] != li or (daily and day[i] != ld):
            prev = v[i]
        else:
            prev = a * prev + (1 - a) * v[i]
        out[i], li, ld = prev, ins[i], day[i]
    return out


def buffer(d, col, b):
    """按截面时间顺序：本截面值 + β·(上一截面同日在多头腿 − 在空头腿)，腿按变换后的值的排名判定。"""
    x = d[["date", "instrument", "day", col]].rename(columns={col: "v"}).sort_values(["date", "instrument"]).reset_index(drop=True)
    out = np.empty(len(x))
    prev_leg, prev_day = {}, None
    for dt, idx in x.groupby("date", sort=True).indices.items():
        sub = x.iloc[idx]
        day = sub["day"].iloc[0]
        v = sub["v"].to_numpy(float).copy()
        if prev_day is not None and day == prev_day and prev_leg:
            leg = sub["instrument"].map(prev_leg).fillna(0).to_numpy(float)
            v = v + b * leg
        rr = pd.Series(v).rank(pct=True).to_numpy()
        out[idx] = v
        prev_leg = dict(zip(sub["instrument"].to_numpy(), np.where(rr > 0.9, 1.0, np.where(rr <= 0.1, -1.0, 0.0))))
        prev_day = day
    x["out"] = out
    return d[["date", "instrument"]].merge(x[["date", "instrument", "out"]], on=["date", "instrument"], how="left")["out"].to_numpy()


ANCHOR = {}


def anchor_z(d):
    """上午相对 10:00 的回吐（评分机任务 s10_am_drift 的值），逐截面 z 分数，10:30–11:30 以外置 0。"""
    if "z" not in ANCHOR:
        a = pd.read_parquet(os.path.join(job_dir("s10_am_drift"), "factor_raw.parquet"))[["date", "instrument", "factor"]]
        a["date"] = pd.to_datetime(a["date"])
        a = a.drop_duplicates(["date", "instrument"], keep="last")
        x = d[["date", "instrument"]].merge(a, on=["date", "instrument"], how="left")["factor"].astype(float).fillna(0.0)
        g = x.groupby(d["date"].to_numpy())
        z = ((x - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0).to_numpy()
        hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
        z[~np.isin(hm, [1030, 1100, 1130])] = 0.0
        ANCHOR["z"] = z
    return ANCHOR["z"]


def addon_z(d, job, slots=None):
    """附加信号（评分机任务的值），逐截面 z 分数；slots 给定时其余截面置 0。"""
    key = (job, tuple(slots) if slots else None)
    if key not in ANCHOR:
        a = pd.read_parquet(os.path.join(job_dir(job), "factor_raw.parquet"))[["date", "instrument", "factor"]]
        a["date"] = pd.to_datetime(a["date"])
        a = a.drop_duplicates(["date", "instrument"], keep="last")
        x = d[["date", "instrument"]].merge(a, on=["date", "instrument"], how="left")["factor"].astype(float).fillna(0.0)
        g = x.groupby(d["date"].to_numpy())
        z = ((x - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0).to_numpy()
        if slots:
            hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
            z[~np.isin(hm, slots)] = 0.0
        ANCHOR[key] = z
    return ANCHOR[key]


def transform(d, spec):
    m = re.fullmatch(r"(?:a(\d+))?(?:q(\d+))?(ewd|ew)?(\d+)?(?:X?buf(\d+))?", spec)
    if not m:
        raise ValueError(spec)
    aw, qw, kind, a, b = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    d = d.copy()
    d["t"] = d["r"]
    if qw:                                                                       # 加最后 10 分钟委托结构（y_s3_q_mix_late，全部截面）
        d["t"] = d["t"] + int(qw) / 100.0 * addon_z(d, os.environ.get("QJOB", "y_s3_q_mix_late"))     # 环境变量 QJOB 选附加项任务
        g = d.groupby("date")["t"]
        d["t"] = ((d["t"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    if aw:                                                                       # 先加时间锚点（z 分数单位的权重，a20 = 0.2），再重新标准化
        d["t"] = d["t"] + int(aw) / 100.0 * anchor_z(d)
        g = d.groupby("date")["t"]
        d["t"] = ((d["t"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    if kind:
        d["t"] = ewma(d, int(a) / 100.0, daily=(kind == "ewd"), col="t")          # 10-21 修：读加过锚点的 t（原来读 r，锚点被覆盖）
        g = d.groupby("date")["t"]
        d["t"] = ((d["t"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    if b:
        d["t"] = buffer(d, "t", int(b) / 100.0)
    return d["t"].to_numpy()


prefix, base = sys.argv[1], sys.argv[2].split("+")
D = load(base)
lv = D["date"]
D = D[(lv >= pd.Timestamp(START + "-01")) & (lv <= pd.Timestamp(END + "-01") + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23))].reset_index(drop=True)
print("底座", base, len(D), flush=True)
for spec in sys.argv[3:]:
    x = transform(D, spec)
    o = pd.concat([pd.DataFrame({"date": D["date"], "instrument": D["instrument"], "factor": x}), Z930[0]], ignore_index=True)
    o = o[(o["date"] >= D["date"].min().normalize()) & (o["date"] <= D["date"].max())]
    name = "%s_%s" % (prefix, spec)
    f = os.path.join(OUTD, "%s.parquet" % name)
    o.to_parquet(f, index=False)
    r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values", "-F", "start=%s" % START, "-F", "end=%s" % END,
                        "-F", "name=%s" % name, "http://127.0.0.1:46220/api/jobs"], capture_output=True, text=True)
    print("提交", name, r.stdout.strip()[:50], flush=True)
