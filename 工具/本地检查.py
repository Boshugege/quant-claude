# -*- coding: utf-8 -*-
"""因子代码的本地结构与语法检查（不连平台）。

用法：
    python 工具/本地检查.py [要检查的目录，默认 ../因子代码]

它会做两件事：
 1. 用 sqlite3 对代码里的 SQL 做真实语法编译（DAI 私有函数用同名占位函数），
    能抓出尾逗号、括号不匹配一类的 SQL 语法错误；
 2. 用合成数据模拟 UDF 引擎（逐分钟算 UDF、再按 30 分钟截面聚合），检查输出
    列名、采样点、重复键和数值是否有限。

重要：本地没有平台提供的 dai / bigmodule，无法真实取数。这里只验证代码结构，
真实评分必须在赛事平台内完成。出现 ModuleNotFoundError: No module named 'dai'
不是代码错误。
"""
import glob
import os
import re
import sqlite3
import sys
import types

import numpy as np
import pandas as pd

PHYS_TABLE = "cpt_jyc_2026_stock_bar1m"
VOL = [f"bid_volume{i}" for i in range(1, 6)] + [f"ask_volume{i}" for i in range(1, 6)]
PRICE = [f"bid_price{i}" for i in range(1, 6)] + [f"ask_price{i}" for i in range(1, 6)]
ORDERS = [f"bid_num_orders{i}" for i in range(1, 6)] + [f"ask_num_orders{i}" for i in range(1, 6)]
BAR = ["adjust_factor", "pre_close", "high", "open", "low", "close", "deal_number", "volume", "amount"]
ALL_FIELDS = BAR + PRICE + VOL + ORDERS
DAI_FUNCS = ["lpad", "rpad", "strptime", "concat", "exp", "sqrt", "log", "abs", "sign"]
ALLOWED_TS = {"09:30:00", "10:00:00", "10:30:00", "11:00:00", "11:30:00",
              "13:30:00", "14:00:00", "14:30:00"}

# ---------- 合成数据 ----------
DAYPAD = "2023-01-03"
INSTR = {"1": "TEST01.SZ", "2": "TEST02.SZ"}
SECTIONS = [93000, 100000, 103000, 110000, 113000, 133000, 140000, 143000]


def _minutes(h0, m0, h1, m1):
    t = pd.Timestamp(f"{DAYPAD} {h0:02d}:{m0:02d}:05")
    end = pd.Timestamp(f"{DAYPAD} {h1:02d}:{m1:02d}:00")
    out = []
    while t <= end:
        out.append(t)
        t += pd.Timedelta(minutes=1)
    return out


def _build_bars():
    rs = np.random.RandomState(7)
    rows = []
    for iid in INSTR:
        base_price = 8.0 if iid == "1" else 15.0
        tick = 0.01
        for t in _minutes(9, 1, 11, 30) + _minutes(13, 1, 14, 30):
            v = abs(rs.normal(1200, 400)) + 1
            r = {"date": t, "instrument_id": iid, "instrument": INSTR[iid]}
            minute_no = int((t.hour * 60 + t.minute) % 240)
            mid = base_price + 0.0005 * minute_no + rs.normal(0, 0.002)
            spread = tick
            r["adjust_factor"] = 1.0
            r["pre_close"] = base_price
            r["bid_price1"] = mid - spread / 2.0
            r["ask_price1"] = mid + spread / 2.0
            for i in range(2, 6):
                r[f"bid_price{i}"] = mid - (i - 0.5) * spread
                r[f"ask_price{i}"] = mid + (i - 0.5) * spread
            for i in range(1, 6):
                r[f"bid_volume{i}"] = v * (0.7 - 0.1 * (i - 1)) * abs(rs.normal(1, 0.25))
                r[f"ask_volume{i}"] = v * (0.6 - 0.1 * (i - 1)) * abs(rs.normal(1, 0.25))
            if rs.rand() < 0.5:
                for i in range(1, 6):
                    r[f"bid_volume{i}"] *= 2.2
            for i in range(1, 6):
                r[f"bid_num_orders{i}"] = max(1, int(round(r[f"bid_volume{i}"] / abs(rs.normal(180, 35)))))
                r[f"ask_num_orders{i}"] = max(1, int(round(r[f"ask_volume{i}"] / abs(rs.normal(180, 35)))))
            r["open"] = mid + rs.normal(0, 0.004)
            r["close"] = mid + rs.normal(0, 0.004)
            r["high"] = max(r["open"], r["close"]) + abs(rs.normal(0.003, 0.001))
            r["low"] = min(r["open"], r["close"]) - abs(rs.normal(0.003, 0.001))
            r["volume"] = max(1, int(abs(rs.normal(50000, 18000))))
            r["amount"] = r["volume"] * (r["open"] + r["close"]) / 2.0
            r["deal_number"] = max(1, int(abs(rs.normal(60, 20))))
            rows.append(r)
    return pd.DataFrame(rows)


BARS = _build_bars()
POOL = pd.DataFrame(
    [{"date": pd.Timestamp(f"{DAYPAD} {s // 10000:02d}:{(s // 100) % 100:02d}:{s % 100:02d}"),
      "instrument": INSTR[i]} for s in SECTIONS for i in INSTR])


# ---------- SQL 语法检查 ----------
def _lint_sql(sql):
    con = sqlite3.connect(":memory:")
    cols = ["date", "instrument_id", "instrument"] + ALL_FIELDS
    con.execute(f"CREATE TABLE {PHYS_TABLE} ({', '.join(c + ' NUMERIC' for c in cols)})")
    con.execute("CREATE TABLE all_instruments (instrument_id NUMERIC, instrument TEXT)")
    con.execute("CREATE TABLE cpt_jyc_2026_instruments (date NUMERIC, instrument TEXT)")
    for fn in DAI_FUNCS:
        con.create_function(fn, -1, lambda *a: 0)
    for m in re.finditer(r'([A-Za-z_]\w*)\s*\(', sql):
        nm = m.group(1).lower()
        if nm not in ("select", "from", "where", "group", "by", "as", "on", "using", "with",
                      "and", "or", "avg", "sum", "min", "max", "count", "cast", "case", "when",
                      "then", "else", "end", "coalesce", "null", "strftime", "distinct", "join",
                      "left", "inner", "order", "not", "is", "in"):
            try:
                con.create_function(m.group(1), -1, lambda *a: 0)
            except Exception:
                pass
    try:
        con.execute(sql)
        return None
    except sqlite3.Error as e:
        return str(e)


# ---------- UDF 引擎模拟 ----------
def _simulate(sql, udf_map):
    d = BARS.copy()
    for name, _args, alias in re.findall(r'([A-Za-z_]\w*)\((.*?)\)\s+AS\s+([A-Za-z_]\w*)',
                                         sql, re.DOTALL):
        if name not in udf_map:
            continue
        fn = udf_map[name]
        params = list(__import__("inspect").signature(fn).parameters)
        if len(params) == 1:
            d[alias] = d["date"].apply(fn)
        else:
            missing = [c for c in params if c not in d.columns]
            if missing:
                raise KeyError(f"UDF {name} 缺少合成字段: {missing}")
            d[alias] = d.apply(lambda r: fn(*[float(r[c]) for c in params]), axis=1)
    if "time_segment" not in d:
        return None
    d["trading_day"] = d["date"].dt.normalize()
    d = d[d["time_segment"] != -1].copy()
    mt = re.search(r'cte_terms AS \(\s*SELECT\s*\*,(.*?)FROM cte_snapshot', sql, re.DOTALL)
    if mt:
        for e, a in re.findall(r'([A-Za-z_]\w*(?:\s*\*\s*[A-Za-z_]\w*)?)\s+AS\s+([A-Za-z_]\w*)', mt.group(1)):
            d[a] = d.eval(e)
    mw = re.search(r'cte_window AS \(\s*SELECT(.*?)FROM', sql, re.DOTALL)
    aggs = re.findall(r'AVG\(\s*([A-Za-z_]\w*)\s*\)\s+AS\s+([A-Za-z_]\w*)', mw.group(1))
    g = d.groupby(["trading_day", "time_segment", "instrument_id", "instrument"], as_index=False)
    out = g[[e for e, _ in aggs]].mean()
    out.columns = ["trading_day", "time_segment", "instrument_id", "instrument"] + [a for _, a in aggs]
    mf = re.search(r'([^\n]+?)\s+AS factor\s*\n\s*FROM cte_window', sql)
    env = {a: out[a] for _, a in aggs}
    out["factor"] = eval(re.sub(r'f\.(\w+)', r'\1', mf.group(1)), {"__builtins__": {}}, env)
    out["date"] = (out["trading_day"]
                   + pd.to_timedelta(out["time_segment"].astype(str).str.zfill(6).str[:2].astype(int), unit="h")
                   + pd.to_timedelta(out["time_segment"].astype(str).str.zfill(6).str[2:4].astype(int), unit="m"))
    return out[["date", "instrument", "factor"]]


def _check_file(path):
    problems = []
    captured = {"sql": []}
    dai = types.ModuleType("dai")

    def query(sql, filters=None, compression=True, udf_list=None, **kw):
        err = _lint_sql(sql)
        captured["sql"].append((sql, err))
        class R:
            def __init__(s, df): s._df = df
            def df(s): return s._df.copy()
        if "cpt_jyc_2026_instruments" in sql:
            return R(POOL)
        out = _simulate(sql, {u.name: u.function for u in (udf_list or [])})
        return R(out if out is not None else BARS)

    dai.query = query

    class U:
        def __init__(s, name, function): s.name = name; s.function = function
    dai.DaiUDF = U
    sys.modules["dai"] = dai

    spec = __import__("importlib.util", fromlist=["u"]).spec_from_file_location("chk", path)
    mod = __import__("importlib.util", fromlist=["u"]).module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        out = mod.main({"bar1m": PHYS_TABLE}, f"{DAYPAD} 00:00:00", f"{DAYPAD} 23:59:59")
    except Exception as e:
        return ["运行异常：%r" % e]
    for _sql, err in captured["sql"]:
        if err:
            problems.append("SQL 语法错误：%s" % err)
    if list(out.columns) != ["date", "instrument", "factor"]:
        problems.append("输出列不对：%s" % list(out.columns))
    if out.duplicated(["date", "instrument"]).sum():
        problems.append("存在重复的 date+instrument")
    ts = set(pd.to_datetime(out["date"]).dt.strftime("%H:%M:%S"))
    if ts - ALLOWED_TS:
        problems.append("出现非采样点时间：%s" % sorted(ts - ALLOWED_TS))
    if pd.to_numeric(out["factor"], errors="coerce").isna().all():
        problems.append("factor 全为缺失")
    return problems


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "因子代码")
    files = sorted(glob.glob(os.path.join(target, "*.py")))
    if not files:
        print("没有找到 .py 文件：", target)
        return 1
    bad = 0
    for p in files:
        problems = _check_file(p)
        if problems:
            bad += 1
            print("不通过 ", os.path.basename(p))
            for x in problems:
                print("        ->", x)
        else:
            print("通过   ", os.path.basename(p))
    print("\n%d/%d 通过" % (len(files) - bad, len(files)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
