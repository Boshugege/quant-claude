# -*- coding: utf-8 -*-
"""第一百二十四批：
    732 = 日收益偏度（20 日滚动偏度，取负）——彩票偏好新源
    733 = 收盘位于日内区间位置（40 窗口均值取负）——压力位新源
    734 = 合法头部（688 配方）+ 2.0 倍中性化（确认实验）
    735 = 合法头部（688 配方）+ 2.5 倍中性化（探针，私榜风险）
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
BASE = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")

KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "c_indpeer"]'
assert KEEP_OLD in BASE

BLOCK = '''    # ===== 新轴 732/733（彩票偏好与日内区间位置） =====
    _dl4 = df.groupby(["day", "instrument"], sort=False).agg(
        _last=("last_px", "last"), _pc=("pre_close", "first"),
        _hi=("win_high", "max"), _lo=("win_low", "min")).reset_index()
    _dl4["_dr"] = np.where((_dl4["_last"] > 0) & (_dl4["_pc"] > 0), np.log(_dl4["_last"] / _dl4["_pc"]), np.nan)
    _dl4 = _dl4.sort_values(["instrument", "day"]).reset_index(drop=True)
    _g4 = _dl4.groupby("instrument", sort=False)
    _dl4["_skew20"] = _g4["_dr"].transform(lambda x: x.shift(1).rolling(20, min_periods=12).skew())
    _k4 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _m4 = _dl4.set_index(["instrument", "day"])
    df["c_skew"] = -_m4["_skew20"].reindex(_k4).to_numpy()
    _clpos = (df["last_px"] - df["win_low"]) / (df["win_high"] - df["win_low"]).where((df["win_high"] - df["win_low"]) > 0)
    df["c_clpos"] = -roll(_clpos, 40, "mean", 10)

'''
KEEP_NEW = KEEP_OLD[:-1] + ', "c_skew", "c_clpos"]'
base2 = BASE.replace(KEEP_OLD, BLOCK + KEEP_NEW, 1)
SIG_OLD = '    result["signal"] = result["c_indpeer"]'
assert SIG_OLD in base2

for name, col, doc in [
    ("732_日收益偏度", "c_skew",
     "日收益偏度（20 日滚动偏度、滞后一日，取负）——彩票偏好新源；全部暴露表列错一天）"),
    ("733_日内区间位置", "c_clpos",
     "收盘位于日内高低区间位置（40 窗口均值取负）——压力/追高新源；全部暴露表列错一天）"),
]:
    S = base2.replace(SIG_OLD, '    result["signal"] = result["%s"]' % col, 1)
    S = S.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", doc, 1)
    (CODE / (name + ".py")).write_text(S, encoding="utf-8")
    print("写出", name)

# 734/735：直接复用 688 头部配方，只改中性化倍数
head = (CODE / "688_头部加span5同伴045与多风格条件反转020.py").read_text(encoding="utf-8")
OLD = '    result["factor"] = neutralize("signal", CHARS4, 1.5)'
assert OLD in head
for name, frac, tag in [
    ("734_合法头部二点零倍中性化", "2.0", "2.0 倍中性化确认实验（合法性：暴露表全部错一天）"),
    ("735_合法头部二点五倍中性化探针", "2.5", "2.5 倍中性化探针（更强中性化，存在私榜风险，仅作探针）"),
]:
    S = head.replace(OLD, '    result["factor"] = neutralize("signal", CHARS4, %s)' % frac, 1)
    S = S.replace("合法头部 + span5 平滑同伴 0.45 + 多风格条件反转平滑 0.20（暴露表错一天）。",
                  "合法头部（688 配方）+ %s。" % tag, 1)
    (CODE / (name + ".py")).write_text(S, encoding="utf-8")
    print("写出", name)
