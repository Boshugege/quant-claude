# -*- coding: utf-8 -*-
"""第一百八十三批：999（833+最高价时段，五维全超）的确认与扩展。
- 1038：999 的 2.5 倍中性化（探针，注明私榜风险）
- 1039：999 的 1.75 倍中性化
- 1040：833 + 高低点时段差 (c_hitime - c_lotime) 0.15（路径时点不对称）
- 1041：999 + 集合竞价占比 0.15（与 IR 族叠加，需补 SQL sum_auc）
"""
import io, os, re

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")

def load(num):
    for f in sorted(os.listdir(D)):
        if f.startswith(str(num) + "_") and f.endswith(".py"):
            return io.open(os.path.join(D, f), encoding="utf-8").read()
    raise SystemExit("missing base %d" % num)

def save(num, name, text):
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write(text)
    print("wrote %d_%s.py" % (num, name))

def sub(t, old, new):
    assert old in t, old[:120]
    return t.replace(old, new, 1)

def blend_add(t, tag, w):
    m = re.search(r'(    result\["signal"\] = rank_blend\(\[)(.*?)(\], \[)(.*?)(\]\))', t)
    assert m, "no rank_blend"
    return t[:m.start()] + m.group(1) + m.group(2) + ', "%s"' % tag + m.group(3) + m.group(4) + ", %s" % w + m.group(5) + t[m.end():]

HILOTIME = ('    _pivP = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
            '    _arrP = _pivP.to_numpy(dtype="float64")\n'
            '    _allnan = np.all(np.isnan(_arrP), axis=1)\n'
            '    _idxH = np.nanargmax(np.where(np.isnan(_arrP), -np.inf, _arrP), axis=1).astype("float64")\n'
            '    _idxL = np.nanargmin(np.where(np.isnan(_arrP), np.inf, _arrP), axis=1).astype("float64")\n'
            '    _idxH[_allnan] = np.nan\n'
            '    _idxL[_allnan] = np.nan\n'
            '    _fracHL = (_idxH - _idxL) / max(_arrP.shape[1] - 1, 1)\n'
            '    _serHL = pd.Series(_fracHL, index=pd.MultiIndex.from_tuples(_pivP.index))\n'
            '    _keyP = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
            '    df["c_hilo"] = _serHL.reindex(_keyP).to_numpy()\n')

A833 = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'

def sql_auc(t):
    t = sub(t, "            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n",
               "            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n"
               "            CASE WHEN hms = '092500' THEN amt_c ELSE 0 END AS auc_amt,\n")
    t = sub(t, "            SUM(upsh_c) AS sum_upsh,", "            SUM(upsh_c) AS sum_upsh,\n            SUM(auc_amt) AS sum_auc,")
    t = sub(t, "        f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals, f.sum_upsh_t",
               "        f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_auc, f.sum_deals, f.sum_upsh_t")
    t = sub(t, '"sum_rng", "sum_upsh", "sum_deals", "sum_upsh_t"]', '"sum_rng", "sum_upsh", "sum_auc", "sum_deals", "sum_upsh_t"]')
    return t
AUC = ('    _hmsU = df["date"].dt.strftime("%H%M%S")\n'
       '    _dU = df.groupby(["instrument", "day"], sort=False)["sum_amt"].transform("sum")\n'
       '    _aU = df["sum_auc"].groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
       '    df["c_auction"] = -(_aU / _dU.where(_dU > 0))\n')

t999 = load(999)
save(1038, "最高价时段999二点五倍中性化", sub(t999, 'neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 2.5)'))
save(1039, "最高价时段999一点七五倍中性化", sub(t999, 'neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 1.75)'))

t = sub(load(833), A833, HILOTIME + A833)
t = sub(t, '"c_body", "c_utime"]\n', '"c_body", "c_utime", "c_hilo"]\n')
save(1040, "行业市值中性化时点重心加高低点时段差", blend_add(t, "c_hilo", "0.15"))

t = sub(t999, A833, AUC + A833)
t = sub(t, '"c_body", "c_utime", "c_hitime"]\n', '"c_body", "c_utime", "c_hitime", "c_auction"]\n')
save(1041, "最高价时段999加集合竞价占比", blend_add(sql_auc(t), "c_auction", "0.15"))
