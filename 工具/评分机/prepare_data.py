# -*- coding: utf-8 -*-
"""一次性预处理（可重复运行，已完成的步骤自动跳过；--force 重做）。

  python prepare_data.py platform   # 1 分钟 Feather（机械盘）→ 平台格式 parquet（NVMe），以及股票池、标签、代码表
  python prepare_data.py core       # 全量面板 → 评分核心（正交基）+ 按月面板缓存
  python prepare_data.py refs       # 参照池：factor_lib 中有公榜分数的因子，逐截面序列
  python prepare_data.py all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

CENT = ["open", "high", "low", "close", "amount", "bid_price1", "bid_price2", "bid_price3",
        "ask_price1", "ask_price2", "ask_price3"]
OHLC = ["open", "high", "low", "close"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def id_map():
    m = pd.read_csv(C.DATA / "exposure" / "id_map.csv")
    m = m[m["ok"].astype(str).str.lower() == "true"]
    return dict(zip(m["local_id"], m["instrument"]))


def months():
    return [p.strftime("%Y%m") for p in pd.period_range(C.FULL_START, C.FULL_END, freq="M")]


# ---------------------------------------------------------------- 平台数据表
def _day_close(df):
    """每只股票每天最后一个有效收盘价和当天的复权因子。"""
    d = df[["date", "instrument_id", "close", "adjust_factor"]].copy()
    d["day"] = d["date"].dt.normalize()
    d = d[d["close"] > 0]
    return d.groupby(["instrument_id", "day"]).agg(dc=("close", "last"), da=("adjust_factor", "last")).reset_index()


def convert_month(args):
    m, prev, mp, out_dir = args
    out = Path(out_dir) / f"{m}.parquet"
    src = C.RAW_1M / f"{m}.0.feather"
    df = pd.read_feather(src)
    days = _day_close(df)
    if prev is not None and (C.RAW_1M / f"{prev}.0.feather").exists():
        pdf = pd.read_feather(C.RAW_1M / f"{prev}.0.feather", columns=["date", "instrument_id", "close", "adjust_factor"])
        pdays = _day_close(pdf)
        pdays = pdays[pdays["day"] == pdays["day"].max()]
        days = pd.concat([pdays, days], ignore_index=True)
    days = days.sort_values(["instrument_id", "day"])
    g = days.groupby("instrument_id")
    days["prev_dc"], days["prev_da"] = g["dc"].shift(1), g["da"].shift(1)
    for c in df.columns:
        if c in ("date", "instrument_id"):
            continue
        x = df[c].astype("float64")
        if c in OHLC:
            x = x.where(x > 0)
        if c in CENT:
            x = x / 100.0
        df[c] = x
    df["day"] = df["date"].dt.normalize()
    df = df.merge(days[["instrument_id", "day", "prev_dc", "prev_da"]], on=["instrument_id", "day"], how="left")
    # 昨收：前一交易日最后一个有效收盘价按复权因子换算到当日口径（价格单位：元）
    df["pre_close"] = (df["prev_dc"] / 100.0 * df["prev_da"] / df["adjust_factor"]).where(df["adjust_factor"] > 0)
    df["instrument_id"] = df["instrument_id"].astype("int32")
    pseudo = "ID" + df["instrument_id"].astype(str).str.zfill(4)
    df["instrument"] = pseudo.map(mp).fillna(pseudo)
    for lv in (4, 5):                      # 本地只有 3 档；4–5 档补空列，并在报告里提示
        for c in ("bid_price", "ask_price", "bid_volume", "ask_volume", "bid_num_orders", "ask_num_orders"):
            df[f"{c}{lv}"] = np.nan
    pool = df[["day", "instrument"]].drop_duplicates()
    df = df.drop(columns=["day", "prev_dc", "prev_da"]).sort_values(["date", "instrument_id"]).reset_index(drop=True)
    df["date"] = df["date"].astype("datetime64[us]")
    tmp = out.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="snappy", row_group_size=500_000)
    tmp.rename(out)
    return m, len(df), pool


def prepare_platform(force=False, workers=4):
    out_dir = C.PLATFORM_DIR / "bar1m"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp = id_map()
    ms = months()
    todo = [(m, ms[i - 1] if i else None, mp, str(out_dir)) for i, m in enumerate(ms)
            if force or not (out_dir / f"{m}.parquet").exists()]
    pools = []
    if todo:
        log(f"转换 1 分钟数据 {len(todo)} 个月（{workers} 进程，机械盘顺序读）")
        with ProcessPoolExecutor(workers) as ex:
            for m, n, pool in ex.map(convert_month, todo):
                pools.append(pool)
                log(f"  {m}: {n:,} 行")
    pool_file = C.PLATFORM_DIR / "instruments.parquet"
    if pools or force or not pool_file.exists():
        if not pools:                      # 只重建股票池
            for m in ms:
                d = pd.read_parquet(out_dir / f"{m}.parquet", columns=["date", "instrument"])
                pools.append(d.assign(day=d["date"].dt.normalize())[["day", "instrument"]].drop_duplicates())
        pool = pd.concat(pools, ignore_index=True).drop_duplicates()
        old = pd.read_parquet(pool_file) if pool_file.exists() and not force else None
        slots = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
        sec = pd.concat([pool.assign(date=pool["day"] + pd.Timedelta(f"{s}:00")) for s in slots], ignore_index=True)
        sec = sec[["date", "instrument"]]
        if old is not None:
            sec = pd.concat([old, sec]).drop_duplicates()
        sec.sort_values(["date", "instrument"]).to_parquet(pool_file, index=False)
        log(f"股票池 cpt_jyc_2026_instruments：{len(sec):,} 行（当日有行情的股票 × 8 个截面）")
    ai = C.PLATFORM_DIR / "all_instruments.parquet"
    if force or not ai.exists():
        ids = pd.read_parquet(out_dir / f"{ms[-1]}.parquet", columns=["instrument_id", "instrument"]).drop_duplicates()
        for m in ms[:-1]:
            ids = pd.concat([ids, pd.read_parquet(out_dir / f"{m}.parquet", columns=["instrument_id", "instrument"]).drop_duplicates()])
        ids.drop_duplicates().sort_values("instrument_id").to_parquet(ai, index=False)
        log(f"代码表 all_instruments：{ids['instrument_id'].nunique()} 只")
    vw = C.PLATFORM_DIR / "vwap.parquet"
    if force or not vw.exists():
        lab = pd.read_parquet(C.DATA / "labels_y_off.parquet", columns=["date", "instrument", "y"])
        lab["instrument"] = lab["instrument"].map(mp).fillna(lab["instrument"])
        lab.rename(columns={"y": "vwap_return"}).to_parquet(vw, index=False)
        log(f"标签 cpt_jyc_2026_vwap：{len(lab):,} 行（本地由 1 分钟重建，与官方截面秩相关约 0.97）")


# ---------------------------------------------------------------- 评分核心与面板缓存
def prepare_core(force=False):
    import engine
    import localeval as le
    import panel_store
    if (C.CORE_DIR / "meta.json").exists() and not force:
        log("评分核心已存在，跳过")
        return
    panel_store.configure_localeval()
    t0 = time.time()
    p = le.build_panel(le.load_bars(C.FULL_START, C.FULL_END, data_dir=le.DATA_DIR))
    log(f"全量面板 {p.shape}，{time.time() - t0:.0f}s")
    meta = engine.build_core(p, le.NEUTRAL_COLS, C.CORE_DIR, workers=C.ENGINE_THREADS)
    log("评分核心：", json.dumps(meta, ensure_ascii=False))
    panel_store.save_panel(p)
    log(f"按月面板缓存写入 {C.PANEL_DIR}")


# ---------------------------------------------------------------- 参照池
def prepare_refs(force=False):
    import factor_lib
    import refs
    known = [x for x in factor_lib.FACTORS if x[2] is not None]
    log(f"参照池：{len(known)} 个有公榜分数的因子")
    refs.build_all(known, force=force, log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["platform", "core", "refs", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.step in ("platform", "all"):
        prepare_platform(a.force, a.workers)
    if a.step in ("core", "all"):
        prepare_core(a.force)
    if a.step in ("refs", "all"):
        prepare_refs(a.force)


if __name__ == "__main__":
    main()
