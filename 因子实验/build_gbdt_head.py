# -*- coding: utf-8 -*-
"""非线性堆叠（第十四轮 H2）的平台代码版生成器：浅层 GBDT（远端导出的纯 numpy JSON）作用在多个分量上。
特征口径与远端 h2_stack.build_X 逐列一致（`feat_order` 的顺序）：
  - 排名轴：分量（根、队友头部）在对齐到 A 的键之后逐截面百分位排名 − 0.5（= engine._group_rank_pct − 0.5，只在有值的行上排），
    缺失记 0；可先做同一股票连续截面的 EWMA（新值权重 1 − α，与 d2_optimize.axis 的 alpha 相同）。
  - 原始值：A（1363）本身的原始值。
  - 截面时刻 one-hot：slot1000…slot1430。
  - 季节性族（L_5／VSH20／PACE20／DOW10）：由 1 分钟 bar 聚成 30 分钟段（与 1232 同一段 SQL），口径同 因子实验/h1_season.py，再排名。
  09:30 截面全部特征与输出置 0。分箱边界用 JSON 里拟合窗上算好的 edges，不重算。
每个分量按自然季度循环计算、算完只留一列，控制平台整段调用时的内存。
用法：python3 因子实验/build_gbdt_head.py <编号> <标题> <模型 JSON> <A 文件> <原始值特征名>=A <特征名>=<文件>[@<EWMA α>][:说明] ...
"""
import io, json, os, re, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
num, title, fj, fa = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
J = json.load(open(fj, encoding="utf-8"))
F = J["feat_order"]
SEAS = ["L_5", "VSH20", "PACE20", "DOW10"]
SLOTS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430"]
raw_name, specs = None, []
for a in sys.argv[5:]:
    nm, rest = a.split("=", 1)
    if rest == "A":
        raw_name = nm
        continue
    path, _, desc = rest.partition(":")
    path, _, ew = path.partition("@")
    specs.append((nm, path, float(ew) if ew else 0.0, desc or os.path.basename(path)))
need = set(F) - set(SEAS) - {"slot" + s for s in SLOTS} - {raw_name} - {s[0] for s in specs}
assert not need, ("缺少特征", need)
assert all(s[0] in F for s in specs) and raw_name in F


def load(path):
    s = io.open(path, encoding="utf-8").read()
    return s.split('\nif __name__ == "__main__":')


def rename_defs(src, suffix):
    names = re.findall(r"^def (\w+)\(", src, flags=re.M)
    for n in sorted(set(names), key=len, reverse=True):
        src = re.sub(r"\b%s\(" % re.escape(n), "%s%s(" % (n, suffix), src)
    return src


pa = load(fa)
ba = re.sub(r"^def main\(datasources, start_date, end_date\):", "def _main_a(datasources, start_date, end_date):",
            pa[0].rstrip() + "\n", count=1, flags=re.M)
tail = pa[1] if len(pa) > 1 else None
bodies, calls, docs = [], [], []
for i, (nm, path, ew, desc) in enumerate(specs, 1):
    b = rename_defs(load(path)[0].rstrip() + "\n", f"_b{i}")
    assert re.search(r"^def main_b%d\(datasources, start_date, end_date\):" % i, b, flags=re.M), path
    bodies.append(b)
    calls.append(f"        ({F.index(nm)}, main_b{i}, {ew}),")
    docs.append(f"      {nm}（第 {F.index(nm)} 列）{'先做 EWMA α=%g 再' % ew if ew else ''}排名：{desc}")
names_a = set(re.findall(r"^def (\w+)\(", ba, flags=re.M))
for b in bodies:
    assert not (names_a & set(re.findall(r"^def (\w+)\(", b, flags=re.M)))
P = J["model"]["params"]
model_txt = json.dumps({"feat_order": F, "edges": J["edges"], "base": J["model"]["base"], "lr": P["lr"],
                        "depth": P["depth"], "trees": J["model"]["trees"]}, ensure_ascii=False, separators=(",", ":"))
doc_txt = "\n".join(docs)
call_txt = "\n".join(calls)

main = f'''def main(datasources, start_date, end_date):
    """【{num}：{title}】非线性堆叠：浅层 GBDT（{P["n_trees"]} 棵树、深度 {P["depth"]}、学习率 {P["lr"]}，只在 {J["fit_span"][0]}～{J["fit_span"][1]} 拟合）作用在 {len(F)} 列特征上。
      {raw_name}（第 {F.index(raw_name)} 列）：A = {os.path.basename(fa)} 的原始值
{doc_txt}
      slot1000…slot1430：截面时刻 one-hot
      L_5／VSH20／PACE20／DOW10：同时段季节性族（下一段 lab 过去 5 日均值、下一段成交额占比过去 20 日均值、今日累计成交额 ÷ 过去 20 日同一时刻均值、过去 10 个同星期几的下一段 lab 均值），再排名
    排名轴 = 对齐到 A 的键之后逐截面百分位排名 − 0.5（只在有值的行上排），缺失记 0；09:30 截面全部特征与输出置 0。
    分箱边界与树都写死在 _GBDT 里（拟合窗上算好，不重算）。每个分量按自然季度循环计算，控制内存。
    """
    import gc
    import numpy as np
    import pandas as pd

    a = _main_a(datasources, start_date, end_date)[["date", "instrument", "factor"]].copy()
    a["date"] = pd.to_datetime(a["date"])
    a["instrument"] = a["instrument"].astype(str)
    key = a[["date", "instrument"]]
    n = len(a)
    hm = a["date"].dt.strftime("%H%M").to_numpy()
    z930 = hm == "0930"
    feat = _GBDT["feat_order"]
    X = np.zeros((n, len(feat)), np.float32)
    X[:, {F.index(raw_name)}] = np.nan_to_num(pd.to_numeric(a["factor"], errors="coerce").to_numpy(float), nan=0.0)
    for _s in ("1000", "1030", "1100", "1130", "1330", "1400", "1430"):
        X[:, feat.index("slot" + _s)] = (hm == _s)
    del a
    gc.collect()

    def _quarters(fn):
        parts = []
        for _q in pd.period_range(pd.Timestamp(start_date).to_period("Q"), pd.Timestamp(end_date).to_period("Q"), freq="Q"):
            _st = max(pd.Timestamp(start_date), _q.start_time)
            _en = min(pd.Timestamp(end_date), _q.end_time)
            if _st > _en:
                continue
            parts.append(fn(datasources, _st.strftime("%Y-%m-%d %H:%M:%S"), _en.strftime("%Y-%m-%d %H:%M:%S")))
            gc.collect()
        b = pd.concat(parts, ignore_index=True)
        b["date"] = pd.to_datetime(b["date"])
        b["instrument"] = b["instrument"].astype(str)
        return b.drop_duplicates(["date", "instrument"])

    def _rank_on_key(b, col):
        """对齐到 A 的键，再逐截面百分位排名 − 0.5（只在有值的行上排），缺失记 0。"""
        m = key.merge(b[["date", "instrument", col]], on=["date", "instrument"], how="left")
        x = pd.to_numeric(m[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        r = x.groupby(m["date"]).rank(pct=True) - 0.5
        return r.fillna(0.0).to_numpy(np.float32)

    for _j, _fn, _ew in [
{call_txt}
    ]:
        b = _quarters(_fn)[["date", "instrument", "factor"]]
        b["factor"] = pd.to_numeric(b["factor"], errors="coerce")
        if _ew > 0:                                                       # 同一股票按时间连续的 EWMA（跨日），只用过去
            b = b.sort_values(["instrument", "date"]).reset_index(drop=True)
            b["factor"] = b.groupby("instrument", sort=False)["factor"].transform(lambda x: x.ewm(alpha=1 - _ew, adjust=False, ignore_na=True).mean())
        X[:, _j] = _rank_on_key(b, "factor")
        del b
        gc.collect()

    s = _quarters(_seas_impl)
    for _c in ("L_5", "VSH20", "PACE20", "DOW10"):
        X[:, feat.index(_c)] = _rank_on_key(s, _c)
    del s
    gc.collect()

    X[z930] = 0.0
    v = _gbdt_score(X)
    v[z930] = 0.0
    out = pd.DataFrame({{"date": key["date"].to_numpy(), "instrument": key["instrument"].to_numpy(), "factor": np.nan_to_num(v, nan=0.0)}})
    return out.sort_values(["date", "instrument"]).reset_index(drop=True)


def _gbdt_score(X):
    """纯 numpy 推理：逐列按拟合窗分箱（searchsorted, side=left），每棵树从根往下走（bin <= 切分点走左 2n+1，否则右 2n+2）。"""
    import numpy as np
    E = np.asarray(_GBDT["edges"], np.float32)
    B = np.empty(X.shape, np.uint8)
    for j in range(X.shape[1]):
        B[:, j] = np.searchsorted(E[j], X[:, j], side="left").astype(np.uint8)
    score = np.full(len(X), float(_GBDT["base"]))
    ar = np.arange(len(X))
    for tr in _GBDT["trees"]:
        m = max(int(it[1]) for it in tr) + 1
        LV, SF, SK, HS = np.zeros(m), np.zeros(m, np.int64), np.zeros(m, np.int64), np.zeros(m, bool)
        for it in tr:
            if it[0] == "split":
                SF[int(it[1])], SK[int(it[1])], HS[int(it[1])] = int(it[2]), int(it[3]), True
            else:
                LV[int(it[1])] = float(it[2])
        node = np.zeros(len(X), np.int64)
        for _ in range(int(_GBDT["depth"])):
            hs = HS[node]
            if not hs.any():
                break
            left = B[ar, SF[node]] <= SK[node]
            node = np.where(hs, np.where(left, 2 * node + 1, 2 * node + 2), node)
        score += float(_GBDT["lr"]) * LV[node]
    return score


def _seas_impl(datasources, start_date, end_date):
    """同时段季节性族的原始值（截面 10:00…14:30），口径同 因子实验/h1_season.py：
    30 分钟段 (09:30,10:00] … (14:30,15:00] 由 1 分钟 bar 聚合（与 1232 同一段 SQL）；多日统计按同一股票自己的交易日滚动再错开 1 日。
      L_5     下一段 lab（段收盘 ÷ 段 VWAP − 1）过去 5 日均值（至少 3 日）
      VSH20   下一段成交额占全天比重的过去 20 日均值（至少 10 日）
      PACE20  今日至本段累计成交额 ÷ 过去 20 日同一时刻累计成交额均值（至少 10 日）
      DOW10   过去 10 个同星期几的下一段 lab 均值（至少 5 个）
    """
    import numpy as np
    import pandas as pd
    import dai

    SEGS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430", "1500"]
    CS = SEGS[:-1]
    NX = SEGS[1:]
    bar1m = datasources["bar1m"]
    q0 = (pd.to_datetime(start_date) - pd.Timedelta(days=130)).strftime("%Y-%m-%d %H:%M:%S")
    sql_seg = f"""
        WITH b AS (
            SELECT b.instrument_id, ai.instrument, b.date, CAST(b.date AS DATE) AS d,
                CAST(b.open AS DOUBLE) AS o, CAST(b.close AS DOUBLE) AS c,
                CAST(COALESCE(b.volume, 0) AS DOUBLE) AS v, CAST(COALESCE(b.amount, 0) AS DOUBLE) AS a,
                CASE
                    WHEN strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '100000' THEN '1000'
                    WHEN strftime(b.date, '%H%M%S') > '100000' AND strftime(b.date, '%H%M%S') <= '103000' THEN '1030'
                    WHEN strftime(b.date, '%H%M%S') > '103000' AND strftime(b.date, '%H%M%S') <= '110000' THEN '1100'
                    WHEN strftime(b.date, '%H%M%S') > '110000' AND strftime(b.date, '%H%M%S') <= '113000' THEN '1130'
                    WHEN strftime(b.date, '%H%M%S') > '130000' AND strftime(b.date, '%H%M%S') <= '133000' THEN '1330'
                    WHEN strftime(b.date, '%H%M%S') > '133000' AND strftime(b.date, '%H%M%S') <= '140000' THEN '1400'
                    WHEN strftime(b.date, '%H%M%S') > '140000' AND strftime(b.date, '%H%M%S') <= '143000' THEN '1430'
                    WHEN strftime(b.date, '%H%M%S') > '143000' AND strftime(b.date, '%H%M%S') <= '150000' THEN '1500'
                END AS seg
            FROM {{bar1m}} b LEFT JOIN all_instruments ai USING (instrument_id)
            WHERE strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '150000'
              AND b.close > 0 AND b.open > 0
        )
        SELECT MAX(date) AS date, instrument_id, MAX(instrument) AS instrument, d, seg,
            SUM(a) AS amt, SUM(v) AS vol, arg_max(c, date) AS c_last, arg_min(o, date) AS o_first
        FROM b WHERE seg IS NOT NULL GROUP BY instrument_id, d, seg
        """
    sg = dai.query(sql_seg, filters={{"date": [q0, end_date]}}, compression=True).df()
    cols = ["date", "instrument"] + ["L_5", "VSH20", "PACE20", "DOW10"]
    if not len(sg):
        return pd.DataFrame(columns=cols)
    sg["instrument"] = sg["instrument"].astype(str)
    sg["d"] = pd.to_datetime(sg["d"])
    amt = sg.pivot_table(index=["instrument", "d"], columns="seg", values="amt", aggfunc="last").reindex(columns=SEGS)
    vol = sg.pivot_table(index=["instrument", "d"], columns="seg", values="vol", aggfunc="last").reindex(columns=SEGS)
    cl = sg.pivot_table(index=["instrument", "d"], columns="seg", values="c_last", aggfunc="last").reindex(columns=SEGS)
    del sg
    amt, vol, cl = amt.sort_index(), vol.sort_index(), cl.sort_index()
    with np.errstate(all="ignore"):
        lab = cl / (amt / vol.where(vol > 0)) - 1.0
        daytot = amt.sum(axis=1)
        vsh = amt.div(daytot.where(daytot > 0), axis=0)
        cum = amt.cumsum(axis=1)

    def roll_prev(fr, N, minp):
        g = fr.groupby(level=0, sort=False)
        m = g.rolling(N, min_periods=minp).mean()
        m.index = m.index.droplevel(0)
        m = m.reindex(fr.index)
        return m.groupby(level=0, sort=False).shift(1)

    def nxt(fr):
        x = fr[NX].copy()
        x.columns = CS
        return x

    feats = {{}}
    feats["L_5"] = roll_prev(nxt(lab), 5, 3)
    feats["VSH20"] = roll_prev(nxt(vsh), 20, 10)
    pc = roll_prev(cum[CS], 20, 10)
    with np.errstate(all="ignore"):
        feats["PACE20"] = cum[CS] / pc.where(pc > 0)
    dw = nxt(lab)                                                          # 同一股票、同一星期几的序列上：先错开 1 个，再取 10 个的均值
    wd = pd.DatetimeIndex(dw.index.get_level_values(1)).weekday
    dw.index = pd.MultiIndex.from_arrays([dw.index.get_level_values(0), wd, dw.index.get_level_values(1)], names=["instrument", "wd", "d"])
    dw = dw.sort_index()
    sh = dw.groupby(level=[0, 1], sort=False).shift(1)
    rr = sh.groupby(level=[0, 1], sort=False).rolling(10, min_periods=5).mean()
    rr.index = rr.index.droplevel([0, 1])
    feats["DOW10"] = rr.reindex(dw.index).droplevel(1).sort_index()
    d0 = pd.Timestamp(pd.to_datetime(start_date).normalize())
    out = None
    for k, fr in feats.items():
        fr = fr[fr.index.get_level_values(1) >= d0]
        st = fr.stack().rename(k).reset_index()
        st.columns = ["instrument", "d", "slot", k]
        st["date"] = st["d"] + pd.to_timedelta(st["slot"].str[:2].astype(int), "h") + pd.to_timedelta(st["slot"].str[2:].astype(int), "m")
        st = st[["date", "instrument", k]]
        out = st if out is None else out.merge(st, on=["date", "instrument"], how="outer")
    out = out[(out["date"] >= pd.to_datetime(start_date)) & (out["date"] <= pd.to_datetime(end_date))]
    return out[cols].reset_index(drop=True)


_GBDT = __import__("json").loads(r"""{model_txt}""")


'''
out = os.path.join(D, f"{num}_{title}.py")
src = main + ba + "\n\n" + "\n\n".join(bodies)
if tail is not None:
    src += '\n\nif __name__ == "__main__":' + tail
io.open(out, "w", encoding="utf-8").write(src)
print("wrote", out, len(src) // 1024, "KB")
