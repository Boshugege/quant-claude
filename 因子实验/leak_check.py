# -*- coding: utf-8 -*-
"""截断重跑泄漏检测（未来信息自动检测）。

原理：同一个因子文件跑两次——一次截止到区间末尾，一次把 end_date 截断到某个交易日的中午 11:00。
两次运行中，截止时刻**之前**的所有截面（更早的交易日，以及截断日当天 09:30/10:00/10:30/11:00）
因子值必须完全一致。任何差异（容差 1e-9）都说明该因子在早期截面上用到了未来数据。

用法：
    python3 因子实验/leak_check.py <因子文件> [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--cutoff "YYYY-MM-DD HH:MM:SS"]

默认参数针对本地 store（bar1m 覆盖 202001–202412）。返回码：0=干净(clean)，2=检出泄漏(leak)，1=运行错误。
"""
import argparse
import glob
import importlib.util
import pathlib
import sys
import types

import duckdb
import pandas as pd

STORE = pathlib.Path("/home/user/jyc_scorer/store/platform")
ALL_MONTHS = sorted(p.stem for p in STORE.glob("bar1m/*.parquet"))
INST = str(STORE / "instruments.parquet")
ALLINST = str(STORE / "all_instruments.parquet")
EXP = sorted(glob.glob("/home/user/jyc_scorer/data/exposure/exposure_*.parquet"))

TOL = 1e-9


def months_between(lo_ts, hi_ts):
    out = []
    for m in ALL_MONTHS:
        y, mo = int(m[:4]), int(m[4:6])
        first = pd.Timestamp(year=y, month=mo, day=1)
        last = first + pd.offsets.MonthEnd(0)
        if last >= lo_ts and first <= hi_ts:
            out.append(m)
    return out


def make_dai(months):
    bars = [str(STORE / "bar1m" / ("%s.parquet" % m)) for m in months]

    def mkconn():
        c = duckdb.connect()
        c.execute("SET threads=2")
        c.execute("CREATE VIEW cpt_jyc_2026_stock_bar1m AS SELECT * FROM read_parquet(%r, union_by_name=true)" % bars)
        c.execute("CREATE VIEW cpt_jyc_2026_instruments AS SELECT * FROM read_parquet(%r, union_by_name=true)" % INST)
        c.execute("CREATE VIEW all_instruments AS SELECT * FROM read_parquet(%r, union_by_name=true)" % ALLINST)
        if EXP:
            c.execute("CREATE VIEW cpt_jyc_2026_exposure AS SELECT * FROM read_parquet(%r, union_by_name=true)" % EXP)
        return c

    def run_df(sql, lo, hi):
        # 每次查询都新建一个 duckdb 连接；用完必须 close，否则连接与其视图缓冲会一直累积
        # （合成版里头部会发很多条 SQL，泄漏连接会把 10 GB 的 agent.slice 撑爆——2026-10-14 复现）。
        c = mkconn()
        try:
            q = "SELECT * FROM (%s) AS _t WHERE _t.date >= TIMESTAMP '%s' AND _t.date <= TIMESTAMP '%s'" % (sql, lo, hi)
            cur = c.execute(q)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        finally:
            c.close()
        return pd.DataFrame(rows, columns=cols)

    def fake_query(sql, filters=None, compression=True, **kw):
        lo, hi = (filters or {}).get("date", ["2000-01-01 00:00:00", "2100-01-01 00:00:00"])
        return types.SimpleNamespace(df=lambda: run_df(sql, lo, hi))

    dai = types.ModuleType("dai")
    dai.query = fake_query
    return dai


def run_factor(path, start, end, lead_days=60):
    # 因子内部会自行向前取 400 天做同伴/滚动历史。为控制运行时间，这里只把视图覆盖
    # [start - lead_days, end]；两次运行使用完全相同的视图窗口，泄漏比较依然有效。
    spec = importlib.util.spec_from_file_location("fac_" + pathlib.Path(path).stem, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["dai"] = make_dai(months_between(pd.Timestamp(start) - pd.Timedelta(days=lead_days), pd.Timestamp(end)))
    spec.loader.exec_module(m)
    out = m.main({"bar1m": "cpt_jyc_2026_stock_bar1m"}, start, end)
    if list(out.columns) != ["date", "instrument", "factor"]:
        raise SystemExit("输出列不是 [date, instrument, factor]：%s" % list(out.columns))
    out = out.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("factor")
    ap.add_argument("--start", default="2023-04-03")
    ap.add_argument("--end", default="2023-08-31 23:59:59")
    ap.add_argument("--cutoff", default="2023-06-15 11:00:00")
    ap.add_argument("--lead-days", type=int, default=60)
    a = ap.parse_args()
    cutoff = pd.Timestamp(a.cutoff)

    print("[leak_check] %s" % pathlib.Path(a.factor).name)
    print("  区间: %s ~ %s，截断点: %s" % (a.start, a.end, a.cutoff))
    full = run_factor(a.factor, a.start, a.end, a.lead_days)
    trunc = run_factor(a.factor, a.start, a.cutoff, a.lead_days)

    f = full[full["date"] <= cutoff].copy()
    t = trunc[trunc["date"] <= cutoff].copy()
    f = f.sort_values(["date", "instrument"]).reset_index(drop=True)
    t = t.sort_values(["date", "instrument"]).reset_index(drop=True)

    key = ["date", "instrument"]
    merged = f.merge(t, on=key, how="outer", suffixes=("_full", "_trunc"), indicator=True)
    only_full = int((merged["_merge"] == "left_only").sum())
    only_trunc = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"].copy()
    d = (both["factor_full"].fillna(-9e18) - both["factor_trunc"].fillna(-9e18)).abs()
    n_diff = int((d > TOL).sum())
    max_diff = float(d.max()) if len(d) else 0.0

    print("  截断前截面行数 full=%d trunc=%d；仅full有=%d 仅trunc有=%d" % (len(f), len(t), only_full, only_trunc))
    print("  不一致行数=%d 最大绝对差=%.3e (容差 %.0e)" % (n_diff, max_diff, TOL))

    if n_diff == 0 and only_full == 0 and only_trunc == 0:
        print("  结论: CLEAN 未发现未来信息")
        return 0
    bad = both[d > TOL].copy()
    show = bad.head(5)
    for _, r in show.iterrows():
        print("   DIFF %s %s  full=%.8f trunc=%.8f" % (r["date"], r["instrument"], r["factor_full"], r["factor_trunc"]))
    # 差异分布：按日期与按时分，帮助定位泄漏来源（越早的截面差异越多 = 全天聚合型泄漏）。
    bad["_d"] = bad["date"].dt.strftime("%Y-%m-%d")
    bad["_t"] = bad["date"].dt.strftime("%H:%M")
    print("   差异按日期（Top10）: %s" % ", ".join("%s=%d" % kv for kv in bad["_d"].value_counts().head(10).items()))
    print("   差异按时分: %s" % ", ".join("%s=%d" % kv for kv in bad["_t"].value_counts().sort_index().items()))
    try:
        bad[["date", "instrument", "factor_full", "factor_trunc"]].to_csv("/tmp/leakdiff_%s.csv" % pathlib.Path(a.factor).stem, index=False)
        print("   差异明细已写入 /tmp/leakdiff_%s.csv" % pathlib.Path(a.factor).stem)
    except Exception:
        pass
    print("  结论: LEAK 使用了未来信息（截断重跑结果不一致）")
    return 2


if __name__ == "__main__":
    sys.exit(main())
