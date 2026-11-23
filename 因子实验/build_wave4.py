# -*- coding: utf-8 -*-
"""第四批（141+）：同伴维度的边界与新定义、多杠杆叠加。全部含 09:30 修复。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import (BASE, CONST_OLD, PEER_OLD, PEER_NEW_SOFT, PEER_MEAN_OLD, PEER_MEAN_NEW,  # noqa: E402
                            TAIL_OLD, KEEP_OLD, add_rv16, keep, tail, doc)  # noqa: E402
from build_121_122 import PA_SNIPPET  # noqa: E402
from build_wave2 import industry_peers, IND_SQL, IND_MASK  # noqa: E402

ANCHOR = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'

PEER_MEAN_AMT = '''            rec = peers_by_month.get(m)
            if rec is None:
                continue
            idx, wts = rec
            sub = result.iloc[rows]
            wide = sub.pivot(index="date", columns="instrument", values=col).reindex(columns=peer_insts)
            amtw = sub.pivot(index="date", columns="instrument", values="amt_p").reindex(columns=peer_insts)
            valid = idx >= 0
            g = wide.values[:, np.where(valid, idx, 0)]
            g = np.where(valid[None, :, :], g, np.nan)
            aw = amtw.values[:, np.where(valid, idx, 0)]
            aw = np.where(valid[None, :, :], aw, np.nan)
            ww = np.where(valid[None, :, :], wts[None, :, :], 0.0)
            okm = (~np.isnan(g)) & (~np.isnan(aw))
            cnt = (~np.isnan(g)).sum(axis=2)
            wsum = (ww * aw * okm).sum(axis=2)
            pm = np.where((cnt >= 3) & (wsum > 0),
                          np.nansum(np.where(okm, g, 0.0) * ww * aw, axis=2) / np.where(wsum > 0, wsum, 1.0),
                          np.nan)'''

DAILY_OLD = '''    daily = df.groupby(["day", "instrument"], sort=False).agg(px_last=("last_px", "last"), pc=("pre_close", "first"))
    daily_ret = safe_log_ratio(daily["px_last"], daily["pc"]).unstack("instrument").sort_index()'''
DAILY_INTRA = '''    daily_ret = ret30.to_frame("r").assign(
        instrument=df["instrument"].values, date=df["date"].values
    ).pivot(index="date", columns="instrument", values="r").sort_index()'''


def soft(src, k=100, power=2.0, lookback=60, minobs=30):
    src = src.replace(CONST_OLD, "    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = %d, %d, %d, %.1f"
                      % (k, lookback, minobs, power))
    src = src.replace(PEER_OLD, PEER_NEW_SOFT)
    return src.replace(PEER_MEAN_OLD, PEER_MEAN_NEW)


def amt_weight(src):
    src = src.replace("        PEER_K,", "        PEER_K,")
    a = '''    df["c_lowpx"] = -np.log(px.where(px > 0))\n'''
    src = src.replace(a, a + '    df["amt_p"] = roll(df["sum_amt"], 40, "sum", 10)\n')
    src = src.replace(KEEP_OLD, KEEP_OLD.replace(']', ', "amt_p"]'))
    out = src.replace(PEER_MEAN_NEW, PEER_MEAN_AMT)
    assert out != src
    return out


def intraday_peer(src):
    assert src.count(DAILY_OLD) == 1
    src = src.replace(DAILY_OLD, DAILY_INTRA)
    return src.replace("        PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER",
                       "        PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER")


def path_asym(src):
    return src.replace(ANCHOR, ANCHOR + PA_SNIPPET)


SPECS = {
    "因子代码/141_行业内部两百只软同伴.py": dict(
        k=200, power=3.0, ind=True, rv16=False, amt=False, intra=False, pasym=False, extras=(), keepx=(), wts=(),
        text="""行业内部软同伴（k=200、相关度 3 次方）+ 残差阶梯 + 四特征 1.5 倍中性化（09:30 修复版）。

    把两个已验证有效的杠杆叠加：同伴候选限制在同一申万一级行业内（132 号），以及更软的同伴定义
      （k=200、相关度 3 次方，128 号代理分 0.834 为本轮最高）。行业码取月初之前最近交易日的官方行业暴露。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/142_六尺度两百只软同伴.py": dict(
        k=200, power=3.0, ind=False, rv16=True, amt=False, intra=False, pasym=False, extras=(), keepx=("r_rv16",), wts=(),
        text="""六尺度残差阶梯 + 更软同伴（k=200、3 次方）+ 四特征 1.5 倍中性化（09:30 修复版）。

    叠加 119 号（约 2 日尺度分量，换手百分位 0.70 最高）与 128 号（更软同伴，代理分 0.834 最高）两个杠杆。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/143_成交额加权同伴.py": dict(
        k=200, power=3.0, ind=False, rv16=False, amt=True, intra=False, pasym=False, extras=(), keepx=(), wts=(),
        text="""成交额加权软同伴残差阶梯 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    同伴均值不再等权：按「同伴的相关度权重 × 同伴最近 40 窗口成交额」加权，即让同板块中成交更活跃、
      信息含量更高的同伴主导基准。依据：行业/板块收益率的信息含量与成交活跃度正相关，成交额大的同伴
      更能代表板块的共同因子（因子分析资料库第四节建议的「同伴均值用成交额加权」）。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/144_长窗口软同伴.py": dict(
        k=200, power=3.0, ind=False, rv16=False, amt=False, intra=False, pasym=False, extras=(), keepx=(), wts=(), lookback=240,
        text="""长窗口软同伴残差阶梯 + 四特征 1.5 倍中性化（09:30 修复版）。

    软同伴定义不变（k=200、3 次方），但把相关性估计窗口从过去 60 个交易日拉长到过去 240 个交易日：
      更长的窗口给出更稳定的相似性结构（长期同业/同风格关系），而短窗口可能被近期共同事件主导。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/145_日内相关同伴.py": dict(
        k=200, power=3.0, ind=False, rv16=False, amt=False, intra=True, pasym=False, extras=(), keepx=(), wts=(), lookback=160, minobs=60,
        text="""日内相关同伴残差阶梯 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    **同伴的定义换一个数据源**：不再用过去 60 个交易日的日收益相关性，而是用过去约 20 个交易日
      （160 个 30 分钟窗口）**同日 30 分钟收益**的相关性来挑同伴，捕捉的是「日内共同交易节奏」（同一批
      资金同时进出的股票），而不是隔夜/日频的共同因子。参考：日内高频共同波动的分解文献
      （Barigozzi & Hallin 2017 关于高频共同成分）表明日内共同成分与日频共同成分并不一致。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/146_多杠杆叠加.py": dict(
        k=200, power=3.0, ind=False, rv16=True, amt=False, intra=False, pasym=True, extras=("c_pa",), keepx=("r_rv16", "c_pa"), wts=(0.15,),
        text="""多杠杆叠加（更软同伴 + 约 2 日尺度 + 窗口买卖压力）+ 四特征 1.5 倍中性化（09:30 修复版）。

    把本轮各自单独验证有增量的三个杠杆放在一起：软同伴 k=200/3（128）、约 2 日尺度分量（119）、
      窗口收盘价相对窗口 VWAP 位置（121，代理分 0.815），后两项按 0.15 权重进入排名合成。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
}

if __name__ == "__main__":
    for path, sp in SPECS.items():
        s = soft(BASE, k=sp["k"], power=sp["power"], lookback=sp.get("lookback", 60), minobs=sp.get("minobs", 30))
        if sp["intra"]:
            s = intraday_peer(s)
        if sp["ind"]:
            s = s.replace("    instruments_table = datasources.get(\"instruments\", \"cpt_jyc_2026_instruments\")",
                          "    instruments_table = datasources.get(\"instruments\", \"cpt_jyc_2026_instruments\")\n"
                          "    exposure_table = datasources.get(\"exposure\", \"cpt_jyc_2026_exposure\")")
            s = industry_peers(s)
        if sp["rv16"]:
            s = add_rv16(s)
        if sp["amt"]:
            s = amt_weight(s)
        if sp["keepx"]:
            s = keep(s, sp["keepx"])
        if sp["pasym"]:
            s = path_asym(s)
        s = tail(s, sp["extras"], sp["wts"]) if not sp["ind"] else tail(s, (), ())
        s = doc(s, sp["text"])
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
