# -*- coding: utf-8 -*-
"""第六十一批（429+）：两条全新方向 + 把第六十批的新轴搬到当前头部 390 上。

新方向：
- 432 伪行业（收益相关性贪心聚类）内部的阶梯残差：任务说明第 6 节列出但一直没做；用相关性结构划"伪行业"再在组内取残差。
- 433 上影线出现的时点重心（日级）：冲高回落在早盘还是尾盘发生。
组合（把 405/425/426/421 的轴搬到真正的头部 390）：
- 429 390 + 分钟内收盘位置；430 390 中性化 2.0；431 390 + 同伴残差跳跃性；
- 434 390 + 分钟收益已实现偏度；435 390 + 挂单量不平衡；436 390 + 同伴上一时段收益溢出。
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
B390 = D / "390_上影线EWMA加分槽位权重.py"

A_COLS = '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
A_W = '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15]), (len(result), 1))\n'
A_WLATE = '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10])\n'
A_UPSH = '    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n'
A_LADDER = '    result["ladder"] = peer_ladder()\n'
A_RV8 = '    df["r_rv8"] = rolling_vwap_rev(8)\n'
A_BODY_SNAP = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_RAW = '            CAST(COALESCE(deal_number, 0) AS DOUBLE) AS deals\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,\n        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",\n              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:\n'
A_NEU15 = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:90], t.count(old))
    return t.replace(old, new, 1)


def add_keep(s, cols, num):
    m = re.search(r"    keep = \[(.*?)\]\n", s, re.S)
    assert m, num
    body = m.group(1)
    for c in cols:
        if '"%s"' % c not in body:
            body = body.rstrip() + ', "%s"' % c
    return s[:m.start()] + "    keep = [%s]\n" % body + s[m.end():]


def transparent(s, num, snap, win, names):
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, num + "_snap")
    s = repl(s, A_DEALS_WIN, win + A_DEALS_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, " + ", ".join("f.%s" % c for c in names) + "\n"), num + "_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", ' + ", ".join('"%s"' % c for c in names) + "]"), num + "_cast")
    return s


def slot_add(s, num, col, wb, wl):
    s = repl(s, A_COLS, A_COLS.replace(']', ', "%s"]' % col), num + "_cols")
    s = repl(s, A_W, '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, %.2f]), (len(result), 1))\n' % wb, num + "_w")
    return repl(s, A_WLATE, '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, %.2f])\n' % wl, num + "_wlate")


def neu(s, frac, num):
    return repl(s, A_NEU15, '    result["factor"] = neutralize("signal", CHARS4, %.2f)\n' % frac, num + "_neu")


# ---- 429 分钟内收盘位置 ----
def c429(s):
    s = transparent(s, "429",
                    '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low THEN (px_close - px_low) / NULLIF(px_high - px_low, 0) ELSE 0 END AS pos_c,\n',
                    '            AVG(pos_c) AS avg_pos,\n', ("avg_pos",))
    code = '    df["c_pos"] = roll(df["avg_pos"], 40, "mean", 10)\n'
    s = repl(s, A_UPSH, A_UPSH + code, "429_code")
    s = add_keep(s, ["c_pos"], "429")
    return slot_add(s, "429", "c_pos", 0.15, 0.10)


def c430(s):
    return neu(s, 2.0, "430")


# ---- 431 同伴残差跳跃性 ----
def c431(s):
    snap = '            CASE WHEN px_open > 0 AND px_close > 0 THEN ABS((px_close - px_open) / px_open) ELSE 0 END AS absr1_c,\n'
    win = '            AVG(absr1_c) AS avg_absr,\n            MAX(absr1_c) AS max_absr,\n'
    names = ("avg_absr", "max_absr")
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, "431_snap")
    s = repl(s, A_DEALS_WIN, win + A_DEALS_WIN, "431_win")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, " + ", ".join("f.%s" % c for c in names) + "\n"), "431_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", ' + ", ".join('"%s"' % c for c in names) + "]"), "431_cast")
    code = ('    _jmp = df["max_absr"] / df["avg_absr"].where(df["avg_absr"] > 0)\n'
            '    _jmp = _jmp.where(lambda v: (v > 0) & np.isfinite(v))\n'
            '    df["c_jump"] = -_jmp.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n')
    s = repl(s, A_UPSH, A_UPSH + code, "431_code")
    s = add_keep(s, list(names) + ["c_jump", "c_jump_p"], "431")
    resid = '    result["c_jump_p"] = result["c_jump"] - peer_mean("c_jump").fillna(result.groupby("date")["c_jump"].transform("mean"))\n'
    s = repl(s, A_LADDER, A_LADDER + resid, "431_resid")
    return slot_add(s, "431", "c_jump_p", 0.05, 0.05)


# ---- 432 伪行业（收益相关性贪心聚类）内部残差 ----
def c432(s):
    helper = (
        '    def pseudo_industry_resid(col):\n'
        '        """按"收益相关性贪心聚类"划伪行业，再在伪行业内部取残差（任务说明第 6 节，首次实现）。"""\n'
        '        out = np.full(len(result), np.nan)\n'
        '        for m, rows in result.groupby(result_month, sort=False).indices.items():\n'
        '            rec = peers_by_month.get(m)\n'
        '            if rec is None:\n'
        '                continue\n'
        '            idx, _ = rec\n'
        '            n = len(daily_ret.columns)\n'
        '            adj = np.zeros((n, n), dtype=bool)\n'
        '            sub = np.where((idx >= 0).any(axis=1))[0]\n'
        '            valid = idx >= 0\n'
        '            ii = np.arange(n)[:, None]\n'
        '            adj[ii[valid], idx[valid]] = True\n'
        '            adj &= adj.T\n'
        '            lab = -np.ones(n, dtype=np.int64)\n'
        '            nxt = 0\n'
        '            for v in sub:\n'
        '                if lab[v] >= 0:\n'
        '                    continue\n'
        '                seen = {v}\n'
        '                stack = [v]\n'
        '                while stack:\n'
        '                    u = stack.pop()\n'
        '                    for w in np.where(adj[u])[0]:\n'
        '                        if w not in seen:\n'
        '                            seen.add(w)\n'
        '                            stack.append(int(w))\n'
        '                for u in seen:\n'
        '                    lab[u] = nxt\n'
        '                nxt += 1\n'
        '            sub_df = result.iloc[rows]\n'
        '            wide = sub_df.pivot(index="date", columns="instrument", values=col).reindex(columns=peer_insts)\n'
        '            wv = wide.values\n'
        '            r_pos = wide.index.get_indexer(sub_df["date"])\n'
        '            c_pos = pd.Index(peer_insts).get_indexer(sub_df["instrument"])\n'
        '            lv = np.where(c_pos >= 0, lab[np.maximum(c_pos, 0)], -1)\n'
        '            cl_mean = np.full(wv.shape, np.nan)\n'
        '            for c in np.unique(lv[lv >= 0]):\n'
        '                cols_c = np.where(lab == c)[0]\n'
        '                blk = wv[:, cols_c]\n'
        '                cl_mean[:, cols_c] = np.nanmean(blk, axis=1)[:, None]\n'
        '            got = cl_mean[r_pos, np.maximum(c_pos, 0)]\n'
        '            out[rows] = np.where(c_pos >= 0, wide.values[r_pos, np.maximum(c_pos, 0)] - got, np.nan)\n'
        '        return pd.Series(out, index=result.index)\n')
    ladder = (
        '    def peer_ladder_pi():\n'
        '        cols = []\n'
        '        for c in ["r_30", "r_rv4", "r_rv8", "r_day"]:\n'
        '            result[c + "_pi"] = pseudo_industry_resid(c)\n'
        '            cols.append(c + "_pi")\n'
        '        return rank_blend(cols)\n')
    s = repl(s, '    def peer_ladder():\n', helper + ladder + '    def peer_ladder():\n', "432_helpers")
    s = repl(s, A_LADDER, '    result["ladder"] = peer_ladder_pi()\n', "432_ladder")
    return add_keep(s, ["r_30_pi", "r_rv4_pi", "r_rv8_pi", "r_day_pi"], "432")


# ---- 433 上影线时点重心（日级） ----
def c433(s):
    seg = ('            CASE\n'
           '                WHEN hms > \'090000\' AND hms <= \'093000\' THEN 1\n'
           '                WHEN hms > \'093000\' AND hms <= \'100000\' THEN 2\n'
           '                WHEN hms > \'100000\' AND hms <= \'103000\' THEN 3\n'
           '                WHEN hms > \'103000\' AND hms <= \'110000\' THEN 4\n'
           '                WHEN hms > \'110000\' AND hms <= \'113000\' THEN 5\n'
           '                WHEN hms > \'130000\' AND hms <= \'133000\' THEN 6\n'
           '                WHEN hms > \'133000\' AND hms <= \'140000\' THEN 7\n'
           '                WHEN hms > \'140000\' AND hms <= \'143000\' THEN 8\n'
           '                WHILE_PLACEHOLDER\n'
           '            END AS seg_i,\n')
    seg = seg.replace('                WHILE_PLACEHOLDER\n',
                      "                WHEN hms > '143000' AND hms <= '150000' THEN 9\n                ELSE 0\n")
    snap = seg + '            (upsh_c_case) AS upsh_t_c,\n'
    upsh_case = ('CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 '
                 'THEN (px_high - GREATEST(px_open, px_close)) * seg_i ELSE 0 END')
    snap = snap.replace('upsh_c_case', upsh_case)
    win = '            SUM(upsh_t_c) AS sum_upsh_t,\n'
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, "433_snap")
    s = repl(s, A_DEALS_WIN, win + A_DEALS_WIN, "433_win")
    names = ("sum_upsh_t",)
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, " + ", ".join("f.%s" % c for c in names) + "\n"), "433_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", ' + ", ".join('"%s"' % c for c in names) + "]"), "433_cast")
    code = ('    # 上影线时点重心（日级）：Σ(时段序号 × 上影线) / Σ上影线，越大=冲高回落越晚。\n'
            '    _g = df.groupby(["instrument", "day"], sort=False)\n'
            '    _cen = _g["sum_upsh_t"].transform("sum") / _g["sum_upsh"].transform("sum").where(_g["sum_upsh"].transform("sum") > 0)\n'
            '    df["c_utime"] = -(_cen - 5.0)\n')
    s = repl(s, A_RV8, A_RV8 + code, "433_code")
    s = add_keep(s, ["c_utime"], "433")
    return slot_add(s, "433", "c_utime", 0.15, 0.15)


# ---- 434 分钟收益已实现偏度 ----
SK_SNAP = ('            CASE WHEN px_open > 0 THEN (px_close - px_open) / px_open ELSE 0 END AS r1_c,\n'
           '            CASE WHEN px_open > 0 THEN ((px_close - px_open) / px_open) * ((px_close - px_open) / px_open) ELSE 0 END AS r1sq_c,\n'
           '            CASE WHEN px_open > 0 THEN ((px_close - px_open) / px_open) * ((px_close - px_open) / px_open) * ((px_close - px_open) / px_open) ELSE 0 END AS r1cb_c,\n')
SK_WIN = '            SUM(r1sq_c) AS sum_r1sq,\n            SUM(r1cb_c) AS sum_r1cb,\n'


def c434(s):
    s = transparent(s, "434", SK_SNAP, SK_WIN, ("sum_r1sq", "sum_r1cb"))
    code = ('    _sk = df["sum_r1cb"] / (df["sum_r1sq"] ** 1.5).where(df["sum_r1sq"] > 0)\n'
            '    _sk = _sk.where(np.isfinite(_sk))\n'
            '    df["c_skew"] = -_sk.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n')
    s = repl(s, A_UPSH, A_UPSH + code, "434_code")
    s = add_keep(s, ["c_skew"], "434")
    return slot_add(s, "434", "c_skew", 0.15, 0.10)


# ---- 435 挂单量不平衡 ----
OB_RAW = ('            CAST(COALESCE(bid_volume1, 0) AS DOUBLE) AS bv1,\n'
          '            CAST(COALESCE(bid_volume2, 0) AS DOUBLE) AS bv2,\n'
          '            CAST(COALESCE(bid_volume3, 0) AS DOUBLE) AS bv3,\n'
          '            CAST(COALESCE(bid_volume4, 0) AS DOUBLE) AS bv4,\n'
          '            CAST(COALESCE(bid_volume5, 0) AS DOUBLE) AS bv5,\n'
          '            CAST(COALESCE(ask_volume1, 0) AS DOUBLE) AS av1,\n'
          '            CAST(COALESCE(ask_volume2, 0) AS DOUBLE) AS av2,\n'
          '            CAST(COALESCE(ask_volume3, 0) AS DOUBLE) AS av3,\n'
          '            CAST(COALESCE(ask_volume4, 0) AS DOUBLE) AS av4,\n'
          '            CAST(COALESCE(ask_volume5, 0) AS DOUBLE) AS av5,\n')
OB_SNAP = ('            (bv1 + bv2 + bv3 + bv4 + bv5) AS bidv_c,\n'
           '            (av1 + av2 + av3 + av4 + av5) AS askv_c,\n')
OB_WIN = '            SUM(bidv_c) AS sum_bidv,\n            SUM(askv_c) AS sum_askv,\n'


def c435(s):
    s = repl(s, A_RAW, A_RAW[:-1] + ",\n" + OB_RAW, "435_raw")
    s = transparent(s, "435", OB_SNAP, OB_WIN, ("sum_bidv", "sum_askv"))
    code = ('    _den = (df["sum_bidv"] + df["sum_askv"])\n'
            '    _ob = (df["sum_bidv"] - df["sum_askv"]) / _den.where(_den > 0)\n'
            '    df["c_ob"] = -_ob.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n')
    s = repl(s, A_UPSH, A_UPSH + code, "435_code")
    s = add_keep(s, ["c_ob"], "435")
    return slot_add(s, "435", "c_ob", 0.15, 0.10)


# ---- 436 同伴上一时段收益溢出 ----
def c436(s):
    code = ('    _dd = result["date"].dt.normalize()\n'
            '    _pr = peer_mean("r_30").fillna(result.groupby("date")["r_30"].transform("mean"))\n'
            '    df2 = pd.DataFrame({"instrument": result["instrument"], "day": _dd, "pv": _pr})\n'
            '    df2 = df2.sort_values(["instrument", "day"])\n'
            '    result["c_spill"] = df2.groupby("instrument", sort=False)["pv"].shift(1).reindex(result.index)\n')
    s = repl(s, A_LADDER, A_LADDER + code, "436_code")
    s = add_keep(s, ["c_spill"], "436")
    return slot_add(s, "436", "c_spill", 0.10, 0.10)


CANDS = [("429", "头部加分钟内收盘位置", c429),
         ("430", "头部中性化二点零", c430),
         ("431", "头部加同伴残差跳跃性", c431),
         ("432", "伪行业内部残差阶梯", c432),
         ("433", "上影线时点重心", c433),
         ("434", "头部加已实现偏度", c434),
         ("435", "头部加挂单不平衡", c435),
         ("436", "头部加同伴收益溢出", c436)]

if __name__ == "__main__":
    for num, name, fn in CANDS:
        s = fn(B390.read_text(encoding="utf-8"))
        p = D / ("%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", p)
