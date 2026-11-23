# -*- coding: utf-8 -*-
"""第一百七十九批：把本轮最有效的两个新源（集合竞价占比 c_auction、日内成交自相关 c_amtac）
叠加到 IC_IR 榜首 847 臂上，另附最低价时段、成交集中度两条新源。目标：突破 IC_IR/压力前沿。
- 1022：847 + 集合竞价占比 c_auction（需补 SQL sum_auc）
- 1023：847 + 日内成交自相关 c_amtac
- 1024：847 + 最低价时段 c_lotime
- 1025：847 + 成交集中度 c_amthhi
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

ANCHOR = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
KEEP = '"c_utime", "c_indmwrev"]\n'

def add(text, code, tag, w):
    t = sub(text, ANCHOR, code + ANCHOR)
    t = sub(t, KEEP, KEEP.replace('"]\n', '", "%s"]\n' % tag))
    return blend_add(t, tag, w)

def sql_auc847(t):
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
AMT_AC = ('    _pivA = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="sum_amt", aggfunc="sum")\n'
          '    _arrA = _pivA.to_numpy(dtype="float64")\n'
          '    if _arrA.shape[1] >= 3:\n'
          '        _xa, _ya = _arrA[:, 1:], _arrA[:, :-1]\n'
          '        _mxa, _mya = np.nanmean(_xa, axis=1, keepdims=True), np.nanmean(_ya, axis=1, keepdims=True)\n'
          '        _cova = np.nanmean((_xa - _mxa) * (_ya - _mya), axis=1)\n'
          '        _sxa, _sya = np.nanstd(_xa, axis=1), np.nanstd(_ya, axis=1)\n'
          '        _acA = np.where((_sxa > 0) & (_sya > 0), _cova / np.where((_sxa * _sya) > 0, _sxa * _sya, 1.0), np.nan)\n'
          '    else:\n'
          '        _acA = np.full(len(_pivA), np.nan)\n'
          '    _serA = pd.Series(_acA, index=pd.MultiIndex.from_tuples(_pivA.index))\n'
          '    _keyA = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    df["c_amtac"] = -_serA.reindex(_keyA).to_numpy()\n')
LOTIME = ('    _pivL = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
          '    _arrL = _pivL.to_numpy(dtype="float64")\n'
          '    _allnanL = np.all(np.isnan(_arrL), axis=1)\n'
          '    _idxL = np.nanargmin(np.where(np.isnan(_arrL), np.inf, _arrL), axis=1).astype("float64")\n'
          '    _idxL[_allnanL] = np.nan\n'
          '    _fracL = _idxL / max(_arrL.shape[1] - 1, 1)\n'
          '    _serL = pd.Series(_fracL, index=pd.MultiIndex.from_tuples(_pivL.index))\n'
          '    _keyL2 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    df["c_lotime"] = _serL.reindex(_keyL2).to_numpy()\n')
HHI = ('    _pivH = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="sum_amt", aggfunc="sum")\n'
       '    _shareH = _pivH.div(_pivH.sum(axis=1).replace(0.0, np.nan), axis=0)\n'
       '    _hhiH = (_shareH.pow(2)).sum(axis=1)\n'
       '    _keyH = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_amthhi"] = -_hhiH.reindex(_keyH).to_numpy()\n')

save(1022, "ICIR臂847加集合竞价占比", add(sql_auc847(load(847)), AUC, "c_auction", "0.15"))
save(1023, "ICIR臂847加日内成交自相关", add(load(847), AMT_AC, "c_amtac", "0.15"))
save(1024, "ICIR臂847加最低价时段", add(load(847), LOTIME, "c_lotime", "0.15"))
save(1025, "ICIR臂847加成交集中度", add(load(847), HHI, "c_amthhi", "0.15"))
