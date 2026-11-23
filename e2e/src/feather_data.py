# -*- coding: utf-8 -*-
"""远端 Feather 数据后端：实现与提交脚本 DaiBackend 相同的接口（pool / sections / labels / bars）。

平台口径：价格、成交额单位为元；OHLC 缺失为 NaN；instrument 为字符串。
本地文件只有整数 instrument_id，没有股票代码，这里用 "ID" 加 4 位数字的伪代码（如 "ID0001"），只保证训练时一致。
标签：y = 官方 vwap_return 口径 = 截面之后 30 分钟窗口 (t, t+30] 的收盘价 / 同窗口 VWAP - 1
（14:30 截面对应 14:30–15:00；见 reports/12）。
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = os.environ.get("QUANT_ROOT", os.path.expanduser("~/Desktop/work/quant"))
DATA_1M = os.path.join(ROOT, "数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m")
SLOTS = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
# 读取全部可用字段（三档），训练脚本按 FIELD_SET 取子集
FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount", "deal_number",
                "bid_price1", "bid_price2", "bid_price3", "ask_price1", "ask_price2", "ask_price3",
                "bid_volume1", "bid_volume2", "bid_volume3", "ask_volume1", "ask_volume2", "ask_volume3",
                "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
                "ask_num_orders1", "ask_num_orders2", "ask_num_orders3"]
CENT_COLS = ["open", "high", "low", "close", "amount",
             "bid_price1", "bid_price2", "bid_price3", "ask_price1", "ask_price2", "ask_price3"]
OHLC = ["open", "high", "low", "close"]
# 1 分钟 bar -> 所属 30 分钟窗口（以结束时刻命名；集合竞价归入 09:30）
_BUCKETS = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"]


def month_list(start, end):
    return [p.strftime("%Y%m") for p in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")]


LABEL_COLS = ["high", "low", "close", "volume", "amount"]
READ_COLS = list(FEATURE_COLS) + ["adjust_factor"]   # 实际读取的字段；set_read_cols 可缩小到训练所需


def set_read_cols(cols):
    global READ_COLS
    READ_COLS = [c for c in FEATURE_COLS if c in set(cols) | set(LABEL_COLS)] + ["adjust_factor"]
    read_month.cache_clear()


@lru_cache(maxsize=2)
def read_month(month):
    """读取一个月的 READ_COLS 并转为平台口径（缓存最近 2 个月：当月与回看所需的上月）。"""
    path = os.path.join(DATA_1M, f"{month}.0.feather")
    if not os.path.exists(path):
        return None
    df = pd.read_feather(path, columns=["date", "instrument_id"] + READ_COLS)
    for c in READ_COLS:
        x = df[c].astype("float64")
        if c == "adjust_factor":
            df[c] = x
            continue
        if c in OHLC:
            x = x.where(x > 0)
        if c in CENT_COLS:
            x = x / 100.0
        df[c] = x.astype("float32") if c not in ("volume", "amount", "high", "low") else x
    df["instrument"] = "ID" + df["instrument_id"].astype(int).astype(str).str.zfill(4)
    return df.drop(columns="instrument_id")


def read_range(sd, ed, instruments=None):
    sd, ed = pd.Timestamp(sd), pd.Timestamp(ed)
    parts = []
    for m in month_list(sd, ed):
        d = read_month(m)
        if d is None:
            continue
        d = d[(d["date"] >= sd) & (d["date"] <= ed)]
        if instruments is not None:
            d = d[d["instrument"].isin(instruments)]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


_BUCKET_HHMM = np.array([930, 1000, 1030, 1100, 1130, 1330, 1400, 1430, 1500])
LABEL_CACHE = os.path.join(ROOT, "e2e", "cache", "labels_y_off.parquet")


def make_labels(bars):
    """bars（含 high/low/close/volume/amount）-> DataFrame[date, instrument, y]。

    y = 官方标签口径（2026-10-11 用平台导出的 cpt_jyc_2026_vwap 逐行核实）：
        截面 t 之后的窗口 (t, t+30] 的收盘价 / 同窗口 VWAP - 1；11:30 截面对应 13:00–13:30，14:30 截面对应 14:30–15:00。
        平台用更细的数据计算，本地用 1 分钟 bar 重建，截面秩相关约 0.94–1.0。
        VWAP 按分钟清洗（成交均价须在 [最低 × 0.98, 最高 × 1.02] 内），官方同样不含这些异常成交（reports/12）。"""
    hm = (bars["date"].dt.hour * 100 + bars["date"].dt.minute).to_numpy()
    b = np.minimum(np.searchsorted(_BUCKET_HHMM, hm, side="left"), len(_BUCKETS) - 1)
    w = bars.assign(day=bars["date"].dt.normalize(), w=b).sort_values(["instrument", "date"])   # 文件内不按时间排序
    g = w.groupby(["instrument", "day", "w"], sort=True).agg(
        volume=("volume", "sum"), amount=("amount", "sum"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    vwap = g["amount"] / g["volume"].where(g["volume"] > 0)
    vwap = vwap.where((vwap >= g["low"] * 0.98) & (vwap <= g["high"] * 1.02))
    v = vwap.unstack("w").reindex(columns=range(len(_BUCKETS)))
    c = g["close"].where(g["close"] > 0).unstack("w").reindex(columns=range(len(_BUCKETS)))
    rows = []
    for k, s in enumerate(SLOTS):              # 截面 k 的标签窗口 = 下一窗口 k+1
        r = pd.DataFrame({"y": c[k + 1] / v[k + 1] - 1.0}).reset_index()
        r["date"] = r["day"] + pd.Timedelta(f"{s}:00")
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)[["date", "instrument", "y"]]
    return out.replace([np.inf, -np.inf], np.nan)


_LABELS = None


def cached_labels():
    """全区间标签缓存（scripts/build_label_cache.py 生成）；不存在时返回 None。"""
    global _LABELS
    if _LABELS is None and os.path.exists(LABEL_CACHE):
        _LABELS = pd.read_parquet(LABEL_CACHE)
    return _LABELS


DATA_30M = os.path.join(ROOT, "数据/raw_bar30m/cpt_jyc_2026_e2e_bar30m")


@lru_cache(maxsize=12)
def read_month_30m(month):
    """30 分钟数据（本地）→ 与平台 windows30 SQL 同口径的窗口表（尚未计算昨收）。"""
    path = os.path.join(DATA_30M, f"{month}.0.feather")
    if not os.path.exists(path):
        return None
    cols = ["date", "instrument_id", "adjust_factor", "open", "high", "low", "close", "volume", "amount", "deal_number",
            "bid_price1", "ask_price1"]
    d = pd.read_feather(path, columns=cols)
    for c in ("open", "high", "low", "close"):
        d[c] = d[c].astype("float64").where(d[c] > 0) / 100.0
    for c in ("bid_price1", "ask_price1", "amount"):
        d[c] = d[c].astype("float64") / 100.0
    d["volume"] = d["volume"].astype("float64")
    hm = d["date"].dt.hour * 100 + d["date"].dt.minute
    d["date"] = d["date"].where(hm != 925, d["date"].dt.normalize() + pd.Timedelta("09:30:00"))   # 集合竞价归入 09:30 窗口
    vw = d["amount"] / d["volume"].where(d["volume"] > 0)
    ok = (d["volume"] > 0) & (d["amount"] > 0) & (d["low"] > 0) & (vw >= d["low"] * 0.98) & (vw <= d["high"] * 1.02)
    mid = (d["bid_price1"] + d["ask_price1"]) / 2
    out = pd.DataFrame({
        "date": d["date"],
        "instrument": "ID" + d["instrument_id"].astype(int).astype(str).str.zfill(4),
        "sum_amt": d["amount"].where(ok, 0.0),
        "sum_vol": d["volume"].where(ok, 0.0),
        "last_px": mid.where((d["bid_price1"] > 0) & (d["ask_price1"] > d["bid_price1"]), d["close"]),
        "close": d["close"],
        "adj": d["adjust_factor"].astype("float64"),
        "w_open": d["open"], "w_high": d["high"], "w_low": d["low"], "w_close": d["close"],
        "raw_amt": d["amount"], "w_deals": d["deal_number"].astype("float64"),
    })
    return out


WIN30_COLS = ["sum_amt", "sum_vol", "last_px", "pre_close", "w_open", "w_high", "w_low", "w_close", "raw_amt", "w_deals"]


def windows30(sd, ed, instruments):
    """本地 30 分钟窗口表：date, instrument, sum_amt, sum_vol, last_px, pre_close。
    昨收 = 前一交易日最后收盘 × 前一日复权因子 / 当日复权因子（与本地评分器一致）。"""
    sd, ed = pd.Timestamp(sd), pd.Timestamp(ed)
    months = month_list(sd - pd.offsets.MonthBegin(1) - pd.Timedelta(days=1), ed)
    parts = [x for x in (read_month_30m(m) for m in months) if x is not None]
    if not parts:                                   # 本地数据之前的月份（平台上同样返回空表）
        return pd.DataFrame(columns=["date", "instrument"] + WIN30_COLS)
    w = pd.concat(parts, ignore_index=True)
    w = w[w["instrument"].isin(set(instruments))]
    w = w[w["date"] <= ed]
    w["day"] = w["date"].dt.normalize()
    w = w.sort_values(["instrument", "date"])
    last = w.groupby(["instrument", "day"], sort=False).agg(dc=("close", "last"), da=("adj", "last"))
    g = last.groupby(level=0)
    last["pre_close"] = g["dc"].shift(1) * g["da"].shift(1) / last["da"]
    w = w.merge(last[["pre_close"]].reset_index(), on=["instrument", "day"], how="left")
    w["last_px"] = w["last_px"].where(w["last_px"] > 0)
    w = w[w["date"] >= sd]
    return w[["date", "instrument"] + WIN30_COLS].reset_index(drop=True)


_EXPO = None
IND_COLS = ["AGRIFOREST", "MINING", "CHEM", "IRONSTEEL", "NONFERMETAL", "ELECTRONICS", "AUTO", "HOUSEAPP", "FOODBEVER", "TEXTILE",
            "LIGHTINDUS", "HEALTH", "UTILITIES", "TRANSPORTATION", "REALESTATE", "COMMETRADE", "LEISERVICE", "BANK",
            "NONBANKFINAN", "CONGLOMERATES", "CONMAT", "BUILDDECO", "ELECEQP", "MACHIEQUIP", "AERODEF", "COMPUTER", "MEDIA",
            "TELECOM", "COAL", "PETRO", "ENVP", "BEAUTY"]   # 申万一级行业哑变量（平台风格剔除同时剔除行业）


def exposure_local(sd, ed, instruments):
    """官方风格暴露（本地）：暴露表按真实代码索引，经对照表 id_map.csv 接到本地伪代码上。只用于训练。
    对照表和暴露表的本地副本都不随仓库发布，文件不存在时本函数在读文件处报错。"""
    global _EXPO
    if _EXPO is None:
        import glob
        mp = pd.read_csv(os.path.join(ROOT, "数据/exposure/id_map.csv"))
        mp = mp[mp["ok"]]
        cols = ["date", "instrument", "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY",
                "EARNYILD", "GROWTH", "LEVERAGE"] + IND_COLS
        e = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(os.path.join(ROOT, "数据/exposure/exposure_*.parquet")))])
        e["date"] = pd.to_datetime(e["date"]).dt.normalize()
        e = e.merge(mp[["instrument", "local_id"]], on="instrument", how="inner")
        _EXPO = e.drop(columns="instrument").rename(columns={"local_id": "instrument"})
    sd, ed = pd.Timestamp(sd).normalize(), pd.Timestamp(ed).normalize()
    e = _EXPO[(_EXPO["date"] >= sd) & (_EXPO["date"] <= ed)]
    return e[e["instrument"].isin(set(instruments))]


BOOK30_CACHE = os.path.join(ROOT, "e2e", "cache", "book30")


@lru_cache(maxsize=4)
def _book30_month(month):
    """优先读第三版缓存（多 b_mp、b_dep 两列，其余列与旧版逐位相同）。"""
    for path in (os.path.join(BOOK30_CACHE + "v3", f"{month}.parquet"), os.path.join(BOOK30_CACHE, f"{month}.parquet")):
        if os.path.exists(path):
            return pd.read_parquet(path)
    return None


def book30(sd, ed, instruments):
    """本地盘口窗口表（scripts/build_book30_cache.py 用提交脚本的 _BOOK30_SQL 在本地 1 分钟数据上生成）。"""
    sd, ed = pd.Timestamp(sd), pd.Timestamp(ed)
    parts = [x for x in (_book30_month(m) for m in month_list(sd, ed)) if x is not None]
    cols = ["date", "instrument", "b_imb", "b_spr", "b_conc", "b_oimb", "b_imb1", "b_dimb", "b_sprv", "b_vpo", "b_mp", "b_dep"]
    if not parts:
        return pd.DataFrame(columns=cols)
    w = pd.concat(parts, ignore_index=True)
    cols = [c for c in cols if c in w.columns]                    # 旧缓存只有前四列（只用 book 组时足够）
    w = w[(w["date"] >= sd) & (w["date"] <= ed) & w["instrument"].isin(set(instruments))]
    return w[cols].astype({c: "float64" for c in cols[2:]}).reset_index(drop=True)


AUC_CACHE = os.path.join(ROOT, "e2e", "cache", "auc")


@lru_cache(maxsize=4)
def _auc_month(month):
    path = os.path.join(AUC_CACHE, f"{month}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def auc_daily(sd, ed, instruments):
    """本地收盘集合竞价表（scripts/build_auc_cache.py 用提交脚本的 _AUC_SQL 在本地 1 分钟数据上生成）：
    date（当天 15:00）, instrument, a_c1456, a_c1500, a_v1500, a_vday。"""
    sd, ed = pd.Timestamp(sd), pd.Timestamp(ed)
    cols = ["date", "instrument", "a_c1456", "a_c1500", "a_v1500", "a_vday"]
    parts = [x for x in (_auc_month(m) for m in month_list(sd, ed)) if x is not None]
    if not parts:
        return pd.DataFrame(columns=cols)
    w = pd.concat(parts, ignore_index=True)
    w = w[(w["date"] >= sd) & (w["date"] <= ed) & w["instrument"].isin(set(instruments))]
    return w[cols].astype({c: "float64" for c in cols[2:]}).reset_index(drop=True)


class FeatherBackend(object):
    """与 DaiBackend 同接口。股票池 = 当日有行情的股票（本地数据每天约 1000 只即当期成分股）。"""

    def __init__(self, max_days=0):
        self.max_days = max_days          # >0 时每个分片只取前 N 个交易日（smoke 用）

    def pool(self, sd, ed):
        """区间内出现过的全部股票（与平台 DaiBackend.pool 的 DISTINCT 语义一致）。"""
        ids = set()
        for m in month_list(sd, ed):
            path = os.path.join(DATA_1M, f"{m}.0.feather")
            if os.path.exists(path):
                ids.update(pd.read_feather(path, columns=["instrument_id"])["instrument_id"].unique().tolist())
        return sorted("ID" + str(int(i)).zfill(4) for i in ids)

    def sections(self, sd, ed, instruments):
        bars = read_range(sd, ed, instruments)
        pairs = bars[["instrument"]].assign(day=bars["date"].dt.normalize()).drop_duplicates()
        if self.max_days:
            days = sorted(pairs["day"].unique())[:self.max_days]
            pairs = pairs[pairs["day"].isin(days)]
        out = pd.concat([pairs.assign(date=pairs["day"] + pd.Timedelta(f"{s}:00")) for s in SLOTS],
                        ignore_index=True)
        out = out[(out["date"] >= pd.Timestamp(sd)) & (out["date"] <= pd.Timestamp(ed))]
        return out[["date", "instrument"]].reset_index(drop=True)

    def labels(self, sd, ed, instruments):
        allv = cached_labels()
        if allv is not None:
            lab = allv[(allv["date"] >= pd.Timestamp(sd)) & (allv["date"] <= pd.Timestamp(ed))]
            return lab[lab["instrument"].isin(instruments)]
        lab = make_labels(read_range(sd, ed, instruments))
        return lab[(lab["date"] >= pd.Timestamp(sd)) & (lab["date"] <= pd.Timestamp(ed))]

    def windows30(self, sd, ed, instruments):
        return windows30(sd, ed, instruments)

    def exposure(self, sd, ed, instruments):
        return exposure_local(sd, ed, instruments)

    def book30(self, sd, ed, instruments):
        return book30(sd, ed, instruments)

    def auc_daily(self, sd, ed, instruments):
        return auc_daily(sd, ed, instruments)

    def bars(self, sd, ed, instruments, cols=None):
        return read_range(sd, ed, instruments)[["date", "instrument"] + list(cols or READ_COLS)]
