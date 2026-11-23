# -*- coding: utf-8 -*-
"""第三轮多根合成的正式代码版生成器：B = 490 头部原始值 + Σ w_i·rank(根_i)，季节性根先做 EWMA 再排名。
头部 = 490 的 main 原样改名 _head_impl；根 = 1260 框架（第二版）原样改成 _root_impl(…, STAT, _cache)，
同一调用内分钟 SQL 只执行一次（_cache）。口径与 build_values_composite.py 的 raw 模式 + --ewma 完全一致。
用法：python3 因子实验/build_multiroot.py <编号> <标题> <L_lab10 权重> <M_q_lin> <N_dir> <W_vdir> <W_zero> [EWMA α=0.75]"""
import io, os, re, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
# 通用用法：python3 因子实验/build_multiroot.py <编号> <标题> STAT=权重 ... [--alpha 0.75] [--framework 1320_多日分钟框架.py]
#   例：1283 … L_lab10=0.20 M_q_lin=0.10 N_dir=-0.10 W_vdir=-0.10 W_zero=0.10 Y_sd5=0.10 X_n5d=-0.10 --framework 1320_多日分钟框架.py
#   框架须是 1260 的超集（1270、1320 都是：同一 STAT 在各版本里算法相同）；以 L_ 开头的根先做 EWMA(α) 再排名。
num, title = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
FW = "1260_量价同向.py"
LEAN = False
HEADQ = False
MONTHLY = False                                                             # --monthly：分钟类根按自然月分批计算（平台整段调用时内存有上限）
alpha = 0.75
if rest and "=" in rest[0]:
    ws = {}
    i = 0
    while i < len(rest):
        if rest[i] == "--alpha":
            alpha = float(rest[i + 1]); i += 2
        elif rest[i] == "--framework":
            FW = rest[i + 1]; i += 2
        elif rest[i] == "--monthly":
            MONTHLY = True; i += 1
        elif rest[i] == "--lean":                                           # 分钟 SQL 只保留所选根用到的列（框架其余列不算）
            LEAN = True; i += 1
        elif rest[i] == "--headq":                                          # 头部按自然季度循环计算（490 每次调用回看 400 天）
            HEADQ = True; i += 1
        else:
            k, v = rest[i].split("="); ws[k] = float(v); i += 1
else:                                                                       # 旧用法（1280–1282）：五个位置参数
    ws = dict(zip(["L_lab10", "M_q_lin", "N_dir", "W_vdir", "W_zero"], [float(x) for x in rest[:5]]))
    alpha = float(rest[5]) if len(rest) > 5 else 0.75


DESC = {"L_lab10": '同时段季节性（下一个半小时段过去 10 日"收盘 ÷ VWAP − 1"均值）',
        "M_q_lin": "聪明钱（|r|/v 加权的当日 VWAP ÷ 当日 VWAP − 1）",
        "N_dir": "大单方向（Σ r·(ln 单笔额 − 之前分钟均值) ÷ Σ|r|）",
        "W_vdir": "量价同向（Σ r·(ln 成交量 − 之前分钟均值) ÷ Σ|r|）",
        "W_zero": "当日零收益分钟占比",
        "X_n5d": "近 5 日大单方向",
        "Y_sd5": "近 5 日单笔成交额离散度",
        "Z_mfr60": "近 60 分钟上涨分钟成交量占比（资金流）",
        "Z_mfr90": "近 90 分钟上涨分钟成交量占比（资金流）",
        "A_kurt": "日内已实现峰度（当日至今分钟收益）",
        "A_skew5": "近 5 日已实现偏度（1200 分钟收益）",
        "A_kurt5": "近 5 日已实现峰度（1200 分钟收益）"}


def body(path, newsig):
    s = io.open(path, encoding="utf-8").read()
    s = s.split('\nif __name__ == "__main__":')[0].rstrip() + "\n"
    return re.sub(r"^def main\(datasources, start_date, end_date\):", newsig, s, count=1, flags=re.M)


head = body(os.path.join(D, "490_自造特征加行业中性化前日市值强度二点五探针.py"), "def _head_impl(datasources, start_date, end_date):")
root = body(os.path.join(D, FW), "def _root_impl(datasources, start_date, end_date, STAT, _cache):")
root, k_ = re.subn(r'^    STAT = "[^"]*"\n', "", root, count=1, flags=re.M)
assert k_ == 1
root, k_ = re.subn(r'^        df = dai\.query\(sql_min, filters=\{"date": \[(\w+), end_date\]\}, compression=True\)\.df\(\)\n',
                   lambda m: ('        if "min" not in _cache:                                                    # 同一调用内只查一次\n'
                              '            _cache["min"] = dai.query(sql_min, filters={"date": [%s, end_date]}, compression=True).df()\n'
                              '        df = _cache["min"].copy()\n' % m.group(1)), root, count=1, flags=re.M)
assert k_ == 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lean_sql import lean_root                                               # noqa: E402


if LEAN:
    root = lean_root(root, [k for k in ws if ws[k] != 0 and not k.startswith("L_")])

desc_txt = "".join("      %-8s %s\n" % (k, DESC.get(k, "")) for k in ws if ws[k] != 0).rstrip("\n")
wtxt = ", ".join(f'"{k}": {v:+.2f}' for k, v in ws.items() if v != 0)
main = f'''def main(datasources, start_date, end_date):
    """【新根因子 第三轮·多根合成 {num}】{title}

    B = 490 头部原始值 + Σ w·rank(根)，09:30 置 0，股票池骨架、缺失填 0。根来自多统计量框架 {FW}（原样复用），
    以 L_ 开头的季节性根先在同一股票连续截面上做 EWMA（α={alpha}）再排名。各根含义：
{desc_txt}
    权重：{{{wtxt}}}。口径与评测机 values 合成（build_values_composite.py raw 模式）一致。
    """
    import numpy as np
    import pandas as pd
    import dai

    W = {{{wtxt}}}
    ALPHA = {alpha}
    MONTHLY = {MONTHLY}                                                       # True：分钟类根按月分批（控制平台整段调用的内存）
    HEADQ = {HEADQ}                                                           # True：头部按自然季度分批
    _cache = {{}}
    _roots = {{}}
    _mparts = {{}}
    if MONTHLY:                                                               # 分钟类根按月分批：每月一次分钟 SQL，算完该月所有根就释放
        _mins = [st for st in W if not st.startswith("L_")]
        for _ms in pd.period_range(pd.Timestamp(start_date).to_period("M"), pd.Timestamp(end_date).to_period("M"), freq="M"):
            _s = max(pd.Timestamp(start_date), _ms.start_time)
            _e = min(pd.Timestamp(end_date), _ms.end_time)
            if _s > _e:
                continue
            _c = {{}}
            for st in _mins:
                _mparts.setdefault(st, []).append(
                    _root_impl(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S"), st, _c))
            del _c
    for st in W:
        if st.startswith("L_"):
            warm = (pd.to_datetime(start_date) - pd.Timedelta(days=21)).strftime("%Y-%m-%d %H:%M:%S")
            x = _root_impl(datasources, warm, end_date, st, {{}})
            x["date"] = pd.to_datetime(x["date"])
            x = x.sort_values(["instrument", "date"]).reset_index(drop=True)
            x["factor"] = x.groupby("instrument", sort=False)["factor"].transform(
                lambda s: s.ewm(alpha=1 - ALPHA, adjust=False).mean())
            x = x[x["date"] >= pd.Timestamp(start_date)]
        elif MONTHLY:
            x = pd.concat(_mparts.pop(st), ignore_index=True)
            x["date"] = pd.to_datetime(x["date"])
        else:
            x = _root_impl(datasources, start_date, end_date, st, _cache)
            x["date"] = pd.to_datetime(x["date"])
        x["instrument"] = x["instrument"].astype(str)
        x["r_" + st] = x.groupby("date")["factor"].rank(pct=True) - 0.5
        _roots[st] = x[["date", "instrument", "r_" + st]]
    import gc
    gc.collect()                                                              # 先算完各根（只留很小的结果），最后才算头部：峰值≈头部本身
    if HEADQ:                                                                 # 头部按自然季度循环（与评测机按季度评分逐位一致）
        _hp = []
        for _q in pd.period_range(pd.Timestamp(start_date).to_period("Q"), pd.Timestamp(end_date).to_period("Q"), freq="Q"):
            _s = max(pd.Timestamp(start_date), _q.start_time)
            _e = min(pd.Timestamp(end_date), _q.end_time)
            if _s > _e:
                continue
            _hp.append(_head_impl(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S")))
            gc.collect()
        _head = pd.concat(_hp, ignore_index=True).drop_duplicates(["date", "instrument"])
        del _hp
    else:
        _head = _head_impl(datasources, start_date, end_date)
    _head = _head.rename(columns={{"factor": "fh"}})
    _head["date"] = pd.to_datetime(_head["date"])
    _head["instrument"] = _head["instrument"].astype(str)
    _m = _head
    for st, x in _roots.items():
        _m = _m.merge(x, on=["date", "instrument"], how="left")
        _m["r_" + st] = _m["r_" + st].fillna(0.0)
    _m["factor"] = _m["fh"].astype("float64")
    for st, w in W.items():
        _m["factor"] = _m["factor"] + w * _m["r_" + st]

    _slots = {{"0930", "1000", "1030", "1100", "1130", "1330", "1400", "1430"}}
    _inst_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    _pool = dai.query(f"SELECT date, instrument FROM {{_inst_tbl}}",
                      filters={{"date": [start_date, end_date]}}, compression=True).df()
    _pool["date"] = pd.to_datetime(_pool["date"])
    _pool["instrument"] = _pool["instrument"].astype(str)
    _pool = _pool[_pool["date"].dt.strftime("%H%M").isin(_slots)].drop_duplicates(["date", "instrument"])
    _out = _pool.merge(_m[["date", "instrument", "factor"]].drop_duplicates(["date", "instrument"]), on=["date", "instrument"], how="left")
    _out["factor"] = _out["factor"].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    _out.loc[_out["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    return _out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


'''
tail = io.open(os.path.join(D, FW), encoding="utf-8").read().split('\nif __name__ == "__main__":')[1]
out = os.path.join(D, f"{num}_{title}.py")
io.open(out, "w", encoding="utf-8").write(main + head + "\n\n" + root + '\n\nif __name__ == "__main__":' + tail)
print("wrote", out)
