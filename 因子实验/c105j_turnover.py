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
OUTD = "/srv/media/jyc_scorer/incoming/turn7"; os.makedirs(OUTD, exist_ok=True)
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


BS = {int(k): float(v) for k, v in (x.split(':') for x in os.environ.get('BSLOT', '').split(',') if x)}


_EXPO = {}
STY = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]


def resid_styles(d, col, same=False):
    """10-24（c105j）：缓冲带之前把合成分逐截面对官方 10 风格 + 申万一级行业做 OLS 取残差，让缓冲带的"腿"与评分机
    （风格剔除后的值排名）一致。默认用前一交易日的暴露（因子代码里只能拿到前一天的）；same=True 用当天（只作上界对照）。"""
    key = "same" if same else "lag"
    if key not in _EXPO:
        ED = os.path.expanduser("~/jyc_scorer/data/exposure")
        y0, y1 = int(START[:4]) - 1, int(END[:4])                               # 只读需要的年份与列（10-24：全读每个进程 6～7 GB，三个并行被 OOM）
        e = pd.concat([pd.read_parquet(os.path.join(ED, "exposure_%d.parquet" % y), columns=["date", "instrument", "industry_level1_code"] + STY)
                       for y in range(y0, y1 + 1) if os.path.exists(os.path.join(ED, "exposure_%d.parquet" % y))], ignore_index=True)
        e["day"] = pd.to_datetime(e["date"]).dt.normalize()
        e = e.drop_duplicates(["instrument", "day"]).sort_values(["instrument", "day"]).reset_index(drop=True)
        ind = sorted(e["industry_level1_code"].dropna().astype(str).unique())
        X = pd.DataFrame({c: e[c].astype(float) for c in STY})
        for c in ind:
            X["i_" + c] = (e["industry_level1_code"].astype(str) == c).astype(float)
        X[e["industry_level1_code"].isna().to_numpy()] = np.nan
        X["instrument"], X["day"] = e["instrument"].to_numpy(), e["day"].to_numpy()
        if not same:
            cols = [c for c in X.columns if c not in ("instrument", "day")]
            X[cols] = X.groupby("instrument")[cols].shift(1)
        mp = pd.read_csv(os.path.join(ED, "id_map.csv"), dtype=str)
        _EXPO[key] = (X, dict(zip(mp.loc[mp["ok"].str.lower() == "true", "local_id"], mp.loc[mp["ok"].str.lower() == "true", "instrument"])))
    X, i2r = _EXPO[key]
    inst = d["instrument"].map(lambda s: i2r.get(s, s))
    m = pd.DataFrame({"instrument": inst.to_numpy(), "day": d["day"].to_numpy()}).merge(X, on=["instrument", "day"], how="left")
    cols = [c for c in X.columns if c not in ("instrument", "day")]
    if os.environ.get("RESID") == "intra":                                       # 10-24：前一交易日暴露 + 斜率 × 当天至今收益（c126）
        if "intra" not in _EXPO:
            E = "/srv/media/jyc_scorer/incoming/e2eslot"
            rr = pd.read_parquet(E + "/c126_rsofar.parquet"); rr["date"] = pd.to_datetime(rr["date"])
            _EXPO["intra"] = (rr.drop_duplicates(["date", "instrument"]), json.load(open(E + "/c126_slopes.json")))
        rr, sl = _EXPO["intra"]
        r = pd.DataFrame({"date": d["date"].to_numpy(), "instrument": inst.to_numpy()}).merge(rr, on=["date", "instrument"], how="left")["r"].fillna(0.0).to_numpy()
        for k_, b_ in sl.items():
            m[k_] = m[k_].to_numpy(float) + b_ * r
    A = m[cols].to_numpy(float)
    v = d[col].to_numpy(float).copy()
    ok = np.isfinite(A).all(axis=1) & np.isfinite(v)
    out = v.copy()
    for dt, idx in d.groupby("date", sort=False).indices.items():
        k = idx[ok[idx]]
        if len(k) < 50:
            continue
        Ak = np.column_stack([np.ones(len(k)), A[k]])
        b = np.linalg.lstsq(Ak, v[k], rcond=None)[0]
        out[k] = v[k] - Ak @ b
    return out


def buffer(d, col, b, lm=None):
    """按截面时间顺序：本截面值 + β·(上一截面同日在多头腿 − 在空头腿)，腿按变换后的值的排名判定。"""
    x = d[["date", "instrument", "day", col] + (["_s"] if "_s" in d else [])].rename(columns={col: "v"}).sort_values(["date", "instrument"]).reset_index(drop=True)
    out = np.empty(len(x))
    prev_leg, prev_day, prev_leg2 = {}, None, {}
    B2 = float(os.environ.get("BUF2", "0"))
    for dt, idx in x.groupby("date", sort=True).indices.items():
        sub = x.iloc[idx]
        day = sub["day"].iloc[0]
        v = sub["v"].to_numpy(float).copy()
        if prev_day is not None and day == prev_day and prev_leg:
            leg = sub["instrument"].map(prev_leg).fillna(0).to_numpy(float)
            if "_s" in sub:                                                      # 10-26：按股票自适应的缓冲宽度（ADAPT）
                leg = leg * sub["_s"].to_numpy(float)
            v = v + BS.get(dt.hour * 100 + dt.minute, b) * leg                  # 10-22：分时段 β（BSLOT=1430:1.0,…，其余用 b）
            if B2 and prev_leg2:                                                 # 10-24：再上一截面（同日）也在腿里的再加 β2（两截面记忆）
                v = v + B2 * sub["instrument"].map(prev_leg2).fillna(0).to_numpy(float)
        rr = pd.Series(v).rank(pct=True).to_numpy()
        first = not (prev_day is not None and day == prev_day and prev_leg)
        if lm is None or (first and os.environ.get("LIFT_SKIP_FIRST")):
            out[idx] = v
        else:                                                                    # 最小抬升：腿按缓冲后的排名定；输出用原值，腿内低于腿外最高者的股票只抬到它之上 m 个标准差
            z0 = sub["v"].to_numpy(float)
            L, S = rr > 0.9, rr <= 0.1
            o = z0.copy()
            if L.any() and (~L).any():
                cL = z0[~L].max() + lm
                o = np.where(L, np.maximum(z0, cL + 1e-6 * (z0 - z0.min())), o)
            if S.any() and (~S).any():
                cS = z0[~S].min() - lm
                o = np.where(S, np.minimum(z0, cS - 1e-6 * (z0.max() - z0)), o)
            out[idx] = o
        prev_leg2 = prev_leg if (prev_day is not None and day == prev_day) else {}
        prev_leg = dict(zip(sub["instrument"].to_numpy(), np.where(rr > 0.9, 1.0, np.where(rr <= 0.1, -1.0, 0.0))))
        prev_day = day
    x["out"] = out
    return d[["date", "instrument"]].merge(x[["date", "instrument", "out"]], on=["date", "instrument"], how="left")["out"].to_numpy()


def noise_ratio(d, col):
    """10-26：每只股票的信号抖动 = 前 W 个交易日（不含当天）同日相邻截面 |Δz| 的日均值，除以当天截面中位数；
    缓冲宽度乘 (抖动比)^p（ADAPT=p，裁到 [1/3, 3]）。p > 0：信号抖的股票穿越腿边界多是噪声，缓冲宽；稳的股票穿越多是真变化，缓冲窄。"""
    p, W = float(os.environ["ADAPT"]), int(os.environ.get("ADAPT_W", "5"))
    x = d[["instrument", "day", "date", col]].sort_values(["instrument", "date"])
    x["dz"] = x.groupby(["instrument", "day"])[col].diff().abs()
    a = x.groupby(["instrument", "day"])["dz"].mean().rename("a").reset_index().sort_values(["instrument", "day"])
    a["s"] = a.groupby("instrument")["a"].transform(lambda s: s.shift(1).rolling(W, min_periods=2).mean())
    a["s"] = a["s"] / a.groupby("day")["s"].transform("median")
    r = d[["instrument", "day"]].merge(a[["instrument", "day", "s"]], on=["instrument", "day"], how="left")["s"].to_numpy(float)
    r = np.where(np.isfinite(r) & (r > 0), r, 1.0)
    return np.clip(r ** p, 1 / 3, 3)


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
        if os.environ.get("RESID_EACH"):
            d2 = d[["date", "instrument", "day"]].copy(); d2["_a"] = z
            z = resid_styles(d2, "_a")
            if slots:
                z[~np.isin(hm, slots)] = 0.0
            zz = pd.Series(z)
            gg = zz.groupby(d["date"].to_numpy())
            z = ((zz - gg.transform("mean")) / gg.transform("std").replace(0, np.nan)).fillna(0.0).to_numpy()
            if slots:
                z[~np.isin(hm, slots)] = 0.0
        ANCHOR[key] = z
    return ANCHOR[key]


def transform(d, spec):
    m = re.fullmatch(r"(?:a(\d+))?(?:q(\d+))?(ewd|ew)?(\d+)?(?:X?(?:L(\d+))?buf(\d+))?", spec)
    if not m:
        raise ValueError(spec)
    aw, qw, kind, a, lm, b = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)
    d = d.copy()
    d["t"] = d["r"]
    if os.environ.get("RESID_EACH"):                                             # 10-24：底座与每个附加项各自先剔除风格、再 z 分数，权重只作用在剔除后的部分
        d["t"] = resid_styles(d, "t")
        g = d.groupby("date")["t"]
        d["t"] = ((d["t"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    if qw:                                                                       # 加最后 10 分钟委托结构（y_s3_q_mix_late，全部截面）
        d["t"] = d["t"] + int(qw) / 100.0 * addon_z(d, os.environ.get("QJOB", "y_s3_q_mix_late"))     # 环境变量 QJOB 选附加项任务
        g = d.groupby("date")["t"]
        d["t"] = ((d["t"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    for kv in [x for x in os.environ.get("ADD", "").split(",") if x]:          # 10-22：重缓冲下加慢信号（ADD=任务:权重,…）
        job, w = kv.rsplit(":", 1)
        sl = None
        if "@" in job:                                                           # 10-23：任务@HHMM 只在该时段生效（其余截面置 0）
            job, hh = job.split("@"); sl = [int(hh)]
        d["t"] = d["t"] + float(w) * addon_z(d, job, sl)
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
    if os.environ.get("RESID"):                                                  # 10-24：缓冲带之前做风格 + 行业剔除（与评分机的腿一致）
        d["t"] = resid_styles(d, "t", same=(os.environ["RESID"] == "same"))
        g = d.groupby("date")["t"]
        d["t"] = ((d["t"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    if b and os.environ.get("ADAPT"):
        d["_s"] = noise_ratio(d, "t")
    if b:
        d["t"] = buffer(d, "t", int(b) / 100.0, None if lm is None else int(lm) / 100.0)
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
    if r.returncode == 0 and '"job_id"' in r.stdout:                         # 10-23：评分机在任务 input/ 里有自己的副本，本地这份不再留（HDD 压力）
        os.remove(f)
