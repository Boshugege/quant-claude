# -*- coding: utf-8 -*-
"""本地评分核心：读取 30 分钟 bar、构造官方标签、按平台 jyc_eval 的流程计算五项指标。

与平台一致的环节（2026-10-11 用平台导出的标签表、因子值与自测结果核实，见 e2e/reports/12）：
- 标签 y_off = 截面之后 30 分钟窗口 (t, t+30] 的收盘价 / 同窗口 VWAP - 1（14:30 截面对应 14:30–15:00）。
  优先读 1 分钟数据重建的缓存 e2e/cache/labels_y_off.parquet，没有缓存时用 30 分钟 bar 计算（两者与官方的截面秩相关都约 0.97）。
- 因子处理：inf 处理 → 截面标准化 → 对当日 10 个官方风格暴露和申万一级行业哑变量回归取残差（平台探测 02 核实：与平台处理后的因子值相关 1.000000）。
  本地股票若没有对应的官方暴露（伪代码未匹配），不参与评估。去极值对秩类指标几乎没有影响，不做。
- IC：残差因子与标签的截面 Spearman 秩相关；IC_IR = 均值 / 标准差。
- 多空夏普：残差因子前 10% 减后 10% 的等权标签收益，按每年 242 × 8 个截面年化。
- 压力：按截面标签离散度（截面标准差）的中位数分高、低两组，取较弱一组的 IC_IR。
- 换手：多空两腿（各 10%）成分在**同一交易日内**相邻截面之间被替换的比例，跨日那一次不计；两腿取平均。
去极值：平台有这一步，但对排名类因子几乎没有影响（探测 02 中各种去极值方式的结果相同），本地不做。
本地综合分：在本地评估过的因子池内对五项做百分位排名（换手越低越好），各占 20%。
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "数据", "raw_bar30m", "cpt_jyc_2026_e2e_bar30m"))

# 平台的 8 个评估时点（HHMM）。本地 09:25 集合竞价 bar 对应平台 09:30 截面。
EVAL_SLOTS = [930, 1000, 1030, 1100, 1130, 1330, 1400, 1430]
PRICE_FIELDS = ["open", "high", "low", "close", "amount",
                "ask_price1", "ask_price2", "ask_price3", "bid_price1", "bid_price2", "bid_price3"]
OHLC_FIELDS = ["open", "high", "low", "close"]
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
LABEL_CACHE = os.path.join(ROOT, "e2e", "cache", "labels_y_off.parquet")
STYLE_COLS = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
IND_COLS = ["AGRIFOREST", "MINING", "CHEM", "IRONSTEEL", "NONFERMETAL", "ELECTRONICS", "AUTO", "HOUSEAPP", "FOODBEVER", "TEXTILE",
            "LIGHTINDUS", "HEALTH", "UTILITIES", "TRANSPORTATION", "REALESTATE", "COMMETRADE", "LEISERVICE", "BANK",
            "NONBANKFINAN", "CONGLOMERATES", "CONMAT", "BUILDDECO", "ELECEQP", "MACHIEQUIP", "AERODEF", "COMPUTER", "MEDIA",
            "TELECOM", "COAL", "PETRO", "ENVP", "BEAUTY"]   # 申万一级行业哑变量
NEUTRAL_COLS = STYLE_COLS + IND_COLS   # 平台风格剔除的回归变量：10 个风格 + 行业（平台探测 02 核实）


EXPO_DIR = os.path.join(ROOT, "数据", "exposure")


def attach_exposure(p):
    """把当日 10 个官方风格暴露并入面板（伪代码经 id_map.csv 映射到真实代码）；没有暴露的行记 NaN。
    暴露文件不存在时整列记 NaN，evaluate 会提示并退回不剔除风格。"""
    files = sorted(glob.glob(os.path.join(EXPO_DIR, "exposure_*.parquet")))
    if not files or not os.path.exists(os.path.join(EXPO_DIR, "id_map.csv")):
        print(f"[localeval] 未找到官方风格暴露（{EXPO_DIR}），评估将不做风格剔除，与平台口径不一致")
        for c in NEUTRAL_COLS:
            p[c] = np.nan
        return
    mp = pd.read_csv(os.path.join(EXPO_DIR, "id_map.csv"))
    mp = mp[mp["ok"]]
    yrs = set(p["day"].dt.year.unique())
    e = pd.concat([pd.read_parquet(f, columns=["date", "instrument"] + NEUTRAL_COLS) for f in files
                   if int(os.path.basename(f)[9:13]) in yrs], ignore_index=True)
    e["day"] = pd.to_datetime(e["date"]).dt.normalize()
    e = e.merge(mp[["instrument", "local_id"]], on="instrument").drop(columns=["instrument", "date"])
    m = p[["day", "instrument"]].merge(e.rename(columns={"local_id": "instrument"}), on=["day", "instrument"], how="left")
    for c in NEUTRAL_COLS:
        p[c] = m[c].to_numpy()


# ---------------------------------------------------------------- 数据
def load_bars(start="2023-01", end="2024-12", data_dir=DATA_DIR):
    """读取 [start, end] 月份（含）的 30 分钟 bar，并还原为与云端一致的表示（元、NaN）。"""
    lo, hi = start.replace("-", ""), end.replace("-", "")
    files = [f for f in sorted(glob.glob(os.path.join(data_dir, "*.0.feather")))
             if lo <= os.path.basename(f)[:6] <= hi]
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有 {start}~{end} 的数据")
    df = pd.concat([pd.read_feather(f) for f in files], ignore_index=True)
    for c in OHLC_FIELDS:
        df[c] = df[c].astype("float64").where(df[c] != -1)
    for c in PRICE_FIELDS:
        df[c] = df[c].astype("float64") / 100.0
    for c in df.columns:
        if c.endswith(("volume1", "volume2", "volume3", "orders1", "orders2", "orders3")) or c in ("volume", "deal_number"):
            df[c] = df[c].astype("float64")
    return df


def build_panel(df):
    """构造面板：时段、中间价、窗口 VWAP、推导的昨收、标签以及常用中间量。"""
    p = df.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    p["day"] = p["date"].dt.normalize()
    hhmm = p["date"].dt.hour * 100 + p["date"].dt.minute
    p["slot"] = hhmm.where(hhmm != 925, 930)

    valid_book = (p["bid_price1"] > 0) & (p["ask_price1"] > p["bid_price1"])
    mid = (p["bid_price1"] + p["ask_price1"]) / 2.0
    p["px"] = mid.where(valid_book, p["close"].where(p["close"] > 0))
    vwap_raw = (p["amount"] / p["volume"].where(p["volume"] > 0)).where(p["amount"] > 0)
    # 数据清洗：VWAP 必须落在该 bar 的 [最低, 最高] 区间内（留 2% 容差）。2021 年有约 0.19% 的 bar
    # （多为 15:00 bar，疑似盘后/大宗成交计入）VWAP 偏离收盘价数倍，会制造虚假的极端收益。
    # 这些 bar 的 VWAP 视为无效，其成交额/成交量不参与任何 VWAP 累加（amount_c / volume_c）。
    vwap_ok = vwap_raw.notna() & (vwap_raw >= p["low"] * 0.98) & (vwap_raw <= p["high"] * 1.02)
    p["vwap_w"] = vwap_raw.where(vwap_ok)
    p["amount_c"] = p["amount"].where(vwap_ok, 0.0)
    p["volume_c"] = p["volume"].where(vwap_ok, 0.0)

    # 昨收：前一交易日最后一根 bar 的收盘价，按复权因子换算到今日价格口径。
    last = p.groupby(["instrument_id", "day"], sort=False).agg(day_close=("close", "last"), day_adj=("adjust_factor", "last"))
    last = last.reset_index().sort_values(["instrument_id", "day"])
    last["prev_close_adj"] = last.groupby("instrument_id")["day_close"].shift(1) * last.groupby("instrument_id")["day_adj"].shift(1)
    p = p.merge(last[["instrument_id", "day", "prev_close_adj"]], on=["instrument_id", "day"], how="left")
    p["pre_close"] = (p["prev_close_adj"] / p["adjust_factor"]).where(p["adjust_factor"] > 0)

    hh = (p["slot"] // 100).astype(int).astype(str).str.zfill(2) + ":" + (p["slot"] % 100).astype(int).astype(str).str.zfill(2)
    p["key_date"] = p["day"] + pd.to_timedelta(hh + ":00")      # 平台截面时间（09:25 竞价 bar 记为 09:30 截面）
    p["instrument"] = "ID" + p["instrument_id"].astype(int).astype(str).str.zfill(4)

    # 官方标签 y_off：下一根 30 分钟 bar（窗口 (t, t+30]）的收盘价 / VWAP - 1；优先用 1 分钟重建的缓存。
    g = p.groupby("instrument_id", sort=False)
    nxt_same_day = g["day"].shift(-1) == p["day"]
    p["next_vwap"] = g["vwap_w"].shift(-1).where(nxt_same_day)   # factor_lib 的同时段历史特征使用（滞后一天后才进入因子）
    next_close = g["close"].shift(-1).where(nxt_same_day)
    p["y_off"] = next_close.where(next_close > 0) / p["next_vwap"] - 1.0
    if os.path.exists(LABEL_CACHE):
        lab = pd.read_parquet(LABEL_CACHE, columns=["date", "instrument", "y"])
        lab = lab[(lab["date"] >= p["key_date"].min()) & (lab["date"] <= p["key_date"].max())]
        y = p[["key_date", "instrument"]].merge(lab.rename(columns={"date": "key_date"}), on=["key_date", "instrument"], how="left")["y"]
        p["y_off"] = y.to_numpy()
    else:
        print(f"[localeval] 未找到 {LABEL_CACHE}，标签改用 30 分钟 bar 计算")
    attach_exposure(p)
    p["is_eval"] = p["slot"].isin(EVAL_SLOTS)
    return p


# ---------------------------------------------------------------- 指标
def style_residual(factor, panel):
    """平台的因子处理：逐截面 inf 处理 → 标准化 → 对 10 个官方风格 + 申万一级行业哑变量（含截距）OLS 取残差。
    只处理评估截面；没有官方暴露的行与常数截面返回 NaN。factor 中的缺失先补 0（提交代码同样补 0）。"""
    v = pd.Series(factor, index=panel.index).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float64)
    B = panel[NEUTRAL_COLS].to_numpy(np.float64)
    has = np.isfinite(B).all(axis=1)
    ev = panel["is_eval"].to_numpy() & has
    out = np.full(len(panel), np.nan)
    idx = np.flatnonzero(ev)
    keys = panel["key_date"].to_numpy()[idx]
    order = np.argsort(keys, kind="stable")
    idx, keys = idx[order], keys[order]
    cut = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    for ii in np.split(idx, cut):
        x = v[ii]
        sd = x.std()
        if len(ii) < 50 or not sd > 0:
            continue
        z = (x - x.mean()) / sd
        X = np.column_stack([np.ones(len(ii)), B[ii]])
        coef, *_ = np.linalg.lstsq(X, z, rcond=None)
        out[ii] = z - X @ coef
    return pd.Series(out, index=panel.index)


def evaluate(factor, panel, label="y_off", min_stocks=100, neutralize=True):
    """平台口径的五项原始指标（见文件头）。factor 与 panel 行对齐；neutralize=False 时跳过风格剔除（只作对照）。"""
    if neutralize and not np.isfinite(panel[STYLE_COLS].to_numpy(np.float64)).all(axis=1).any():
        neutralize = False                       # 没有暴露数据（见 attach_exposure 的提示）
    f = style_residual(factor, panel) if neutralize else \
        pd.Series(factor, index=panel.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    m = panel["is_eval"].values
    d = pd.DataFrame({"date": panel["key_date"].values[m], "day": panel["day"].values[m], "slot": panel["slot"].values[m],
                      "inst": panel["instrument_id"].values[m], "f": f.values[m], "y": panel[label].values[m]})
    d = d[np.isfinite(d["y"]) & np.isfinite(d["f"])]
    rf = d.groupby("date")["f"].rank(pct=True)
    ry = d.groupby("date")["y"].rank(pct=True)
    d["rf"] = rf - rf.groupby(d["date"]).transform("mean")
    d["ry"] = ry - ry.groupby(d["date"]).transform("mean")
    agg = d.assign(xy=d.rf * d.ry, xx=d.rf ** 2, yy=d.ry ** 2).groupby("date").agg(
        xy=("xy", "sum"), xx=("xx", "sum"), yy=("yy", "sum"), n=("xy", "size"), day=("day", "first"), slot=("slot", "first"))
    agg = agg[agg["n"] >= min_stocks]
    ic = (agg["xy"] / np.sqrt(agg["xx"] * agg["yy"])).replace([np.inf, -np.inf], np.nan).dropna()

    def _ir(x):
        return x.mean() / x.std() if len(x) > 1 and x.std() > 0 else np.nan

    # 多空组合：截面因子前 10% 减后 10% 的等权标签收益
    top, bot = rf > 0.9, rf <= 0.1
    ls = (d["y"].where(top).groupby(d["date"]).mean() - d["y"].where(bot).groupby(d["date"]).mean()).reindex(ic.index).dropna()
    sharpe = ls.mean() / ls.std() * np.sqrt(242 * len(EVAL_SLOTS)) if ls.std() > 0 else np.nan

    # 换手：两腿成分在同一交易日内相邻截面之间的替换比例（跨日不计），两腿平均
    day_of = agg.loc[ic.index, "day"]

    def leg_turnover(mask):
        mem = pd.DataFrame({"date": d["date"], "inst": d["inst"], "in": mask}).pivot_table(
            index="date", columns="inst", values="in", aggfunc="max", fill_value=False).astype(bool)
        mem = mem.reindex(ic.index).astype("float64").fillna(0.0) > 0
        prev = mem.shift(1).astype("float64").fillna(0.0) > 0
        size = mem.sum(axis=1)
        t = 1.0 - (mem & prev).sum(axis=1) / size.where(size > 0)
        same = day_of.eq(day_of.shift(1))
        return t[same].mean()
    turnover = np.nanmean([leg_turnover(top), leg_turnover(bot)])

    # 压力：按截面标签离散度中位数分高低两组，取较弱一组的 IC_IR
    ysd = d.groupby("date")["y"].std().reindex(ic.index)
    stress = min(_ir(ic[ysd > ysd.median()]), _ir(ic[ysd <= ysd.median()]))

    slots = ic.groupby(agg.loc[ic.index, "slot"]).mean()
    return {"ic_mean": ic.mean(), "ic_ir": _ir(ic), "ls_sharpe": sharpe, "stress": stress, "turnover": turnover,
            "n_xsec": len(ic), "slots": {int(k): float(v) for k, v in slots.items()}}


def composite(metrics_table):
    """在本地因子池内对五项指标做百分位排名（换手率越低越好），各占 20% 得到本地综合分。"""
    t = metrics_table
    r = pd.DataFrame(index=t.index)
    for c in ["ic_mean", "ic_ir", "ls_sharpe", "stress"]:
        r[c] = t[c].rank(pct=True)
    r["turnover"] = (-t["turnover"]).rank(pct=True)
    return r.mean(axis=1)
