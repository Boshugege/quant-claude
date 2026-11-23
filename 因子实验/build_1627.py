# -*- coding: utf-8 -*-
"""由 1626 生成 1627（非线性堆叠十六模型平均加委托笔数，全量拟合至 2024）：
  1）文档字符串加委托笔数六个特征；2）_seas_impl 的同一段 SQL 里多算每分钟的委托笔数指标并取段内均值（不额外扫表）；
  3）pandas 生成三个快版本与三个 20 日慢版本（口径同 因子实验/m15_queue.py）；4）main 里按训练口径（缺失补 0 再排名）填入；
  5）_GBDT 换成合并后的模型 JSON。用法：python3 因子实验/build_1627.py <合并模型.json>"""
import json
import sys

SRC = "因子代码/1626_非线性堆叠十六模型平均全量拟合至2024.py"
import os
DST = os.environ.get("DST", "因子代码/1627_非线性堆叠十六模型平均加委托笔数全量拟合至2024.py")
s = open(SRC, encoding="utf-8").read()
J = json.load(open(sys.argv[1], encoding="utf-8"))                     # merge_gbdt.py 的输出 → 代码里嵌入的精简格式（同 build_gbdt_head.py）
if "model" in J:
    P = J["model"]["params"]
    M = {"feat_order": J["feat_order"], "edges": J["edges"], "base": J["model"]["base"], "lr": P["lr"], "depth": P["depth"], "trees": J["model"]["trees"]}
else:
    M = J
OQ = ["OQ_NIMB", "OQ_SIZE", "OQ_DSIZE", "OQ_NIMB_20", "OQ_SIZE_20", "OQ_DSIZE_20"]
fo = M["feat_order"]
assert all(c in fo for c in OQ), fo
assert fo[:27] == json.loads(s[s.index('r"""', s.index("_GBDT = ")) + 4:s.index('"""', s.index('r"""', s.index("_GBDT = ")) + 4)])["feat_order"][:27], "前 27 列须与 1626 相同"


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:80])
    s = s.replace(a, b)


def rep_seas(a, b):
    """只在 _seas_impl 函数体内替换（同样的 SQL 片段在其他分量函数里也有）。"""
    global s
    f0 = s.index("\ndef _seas_impl(")
    f1 = s.index("\ndef ", f0 + 1)
    body = s[f0:f1]
    assert body.count(a) == 1, (body.count(a), a[:80])
    s = s[:f0] + body.replace(a, b) + s[f1:]


i0 = s.index('"""', s.index("def main(")) + 3
i1 = s.index("\n", i0)
s = s[:i0] + "【1627：非线性堆叠十六模型平均加委托笔数全量拟合至2024】= 1626 的 16 个配置与种子原样，特征加盘口委托笔数六列（快 3 + 慢 3），拟合 2020-01-01～2024-12-31；本地（拟合 2020–22 的同配置）相对 1617 在 2023／2024 两年 IC +0.0017～+0.0018、IC_IR +0.049～+0.058、压力 +0.044～+0.058、多空 +0.6～+1.1，换手 +0.004。" + s[i1:]
rep('''    排名轴 = 对齐到 A 的键之后逐截面百分位排名 − 0.5''', '''      OQ_NIMB／OQ_SIZE／OQ_DSIZE（第 %d～%d 列）：盘口委托笔数——一档笔数失衡 (买 − 卖) ÷ (买 + 卖)、一档平均单笔挂单规模之比 log(买 ÷ 卖)、
        1～3 档合计的同一比值；截面所在半小时段内各分钟的均值（排队的人数方向，大单挂在哪一边：小单堆起来的需求随后上涨，大单压盘随后下跌）
      OQ_NIMB_20／OQ_SIZE_20／OQ_DSIZE_20（第 %d～%d 列）：上面三项日均值的过去 20 日均值（至少 10 日，只用前几日）
      委托笔数六列与训练口径一致：缺失记 0 后参与逐截面排名。
    排名轴 = 对齐到 A 的键之后逐截面百分位排名 − 0.5''' % (fo.index("OQ_NIMB"), fo.index("OQ_DSIZE"), fo.index("OQ_NIMB_20"), fo.index("OQ_DSIZE_20")))
rep('''        r = x.groupby(m["date"]).rank(pct=True) - 0.5
        return r.fillna(0.0).to_numpy(np.float32)
''', '''        r = x.groupby(m["date"]).rank(pct=True) - 0.5
        return r.fillna(0.0).to_numpy(np.float32)

    def _rank_fill0(b, col):
        """委托笔数列（1627）：与训练口径一致——对齐到 A 的键，缺失补 0，再逐截面百分位排名 − 0.5。"""
        m = key.merge(b[["date", "instrument", col]], on=["date", "instrument"], how="left")
        x = pd.to_numeric(m[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        r = x.groupby(m["date"]).rank(pct=True) - 0.5
        return r.to_numpy(np.float32)
''')
rep('''    for _c in ("L_5", "VSH20", "PACE20", "DOW10"):
        X[:, feat.index(_c)] = _rank_on_key(s, _c)
''', '''    for _c in ("L_5", "VSH20", "PACE20", "DOW10"):
        X[:, feat.index(_c)] = _rank_on_key(s, _c)
    for _c in ("OQ_NIMB", "OQ_SIZE", "OQ_DSIZE", "OQ_NIMB_20", "OQ_SIZE_20", "OQ_DSIZE_20"):
        X[:, feat.index(_c)] = _rank_fill0(s, _c)
''')
rep_seas('''      DOW10   过去 10 个同星期几的下一段 lab 均值（至少 5 个）
    """''', '''      DOW10   过去 10 个同星期几的下一段 lab 均值（至少 5 个）
      OQ_*    盘口委托笔数（1627，口径同 因子实验/m15_queue.py）：一档笔数失衡、一档单笔挂单规模比、1～3 档单笔规模比的段内分钟均值；
              *_20 = 三项日均值（10:00…14:30 七段）的过去 20 日均值（至少 10 日，错开 1 日）
    """''')
rep_seas('''                CAST(COALESCE(b.volume, 0) AS DOUBLE) AS v, CAST(COALESCE(b.amount, 0) AS DOUBLE) AS a,
                CASE''', '''                CAST(COALESCE(b.volume, 0) AS DOUBLE) AS v, CAST(COALESCE(b.amount, 0) AS DOUBLE) AS a,
                CAST(COALESCE(b.bid_num_orders1, 0) AS DOUBLE) AS bn1, CAST(COALESCE(b.ask_num_orders1, 0) AS DOUBLE) AS an1,
                CAST(COALESCE(b.bid_volume1, 0) AS DOUBLE) AS bv1, CAST(COALESCE(b.ask_volume1, 0) AS DOUBLE) AS av1,
                CAST(COALESCE(b.bid_num_orders1, 0) + COALESCE(b.bid_num_orders2, 0) + COALESCE(b.bid_num_orders3, 0) AS DOUBLE) AS bn,
                CAST(COALESCE(b.ask_num_orders1, 0) + COALESCE(b.ask_num_orders2, 0) + COALESCE(b.ask_num_orders3, 0) AS DOUBLE) AS an,
                CAST(COALESCE(b.bid_volume1, 0) + COALESCE(b.bid_volume2, 0) + COALESCE(b.bid_volume3, 0) AS DOUBLE) AS bv,
                CAST(COALESCE(b.ask_volume1, 0) + COALESCE(b.ask_volume2, 0) + COALESCE(b.ask_volume3, 0) AS DOUBLE) AS av,
                CASE''')
rep_seas('''            SUM(a) AS amt, SUM(v) AS vol, arg_max(c, date) AS c_last, arg_min(o, date) AS o_first
        FROM b WHERE seg IS NOT NULL GROUP BY instrument_id, d, seg
        """
    sg = dai.query(sql_seg, filters={"date": [q0, end_date]}, compression=True).df()
    cols = ["date", "instrument"] + ["L_5", "VSH20", "PACE20", "DOW10"]''', '''            SUM(a) AS amt, SUM(v) AS vol, arg_max(c, date) AS c_last, arg_min(o, date) AS o_first,
            AVG(CASE WHEN bn1 > 0 AND an1 > 0 AND bv1 > 0 AND av1 > 0 THEN (bn1 - an1) / (bn1 + an1) END) AS oq_nimb,
            AVG(CASE WHEN bn1 > 0 AND an1 > 0 AND bv1 > 0 AND av1 > 0 THEN ln((bv1 / bn1) / (av1 / an1)) END) AS oq_size,
            AVG(CASE WHEN bn > 0 AND an > 0 AND bv > 0 AND av > 0 THEN ln((bv / bn) / (av / an)) END) AS oq_dsize
        FROM b WHERE seg IS NOT NULL GROUP BY instrument_id, d, seg
        """
    sg = dai.query(sql_seg, filters={"date": [q0, end_date]}, compression=True).df()
    cols = ["date", "instrument"] + ["L_5", "VSH20", "PACE20", "DOW10"] + ["OQ_NIMB", "OQ_SIZE", "OQ_DSIZE", "OQ_NIMB_20", "OQ_SIZE_20", "OQ_DSIZE_20"]''')
rep_seas('''    cl = sg.pivot_table(index=["instrument", "d"], columns="seg", values="c_last", aggfunc="last").reindex(columns=SEGS)
    del sg''', '''    cl = sg.pivot_table(index=["instrument", "d"], columns="seg", values="c_last", aggfunc="last").reindex(columns=SEGS)
    oqw = {nm: sg.pivot_table(index=["instrument", "d"], columns="seg", values=col, aggfunc="last").reindex(columns=CS).sort_index()
           for nm, col in (("OQ_NIMB", "oq_nimb"), ("OQ_SIZE", "oq_size"), ("OQ_DSIZE", "oq_dsize"))}
    del sg''')
rep_seas('''    d0 = pd.Timestamp(pd.to_datetime(start_date).normalize())
    out = None''', '''    for nm, fr in oqw.items():                                             # 委托笔数（1627）：段内均值；_20 = 日均值的过去 20 日均值（错开 1 日）
        feats[nm] = fr
        sl = roll_prev(fr.mean(axis=1).to_frame("x"), 20, 10)["x"]
        feats[nm + "_20"] = pd.DataFrame({s0: sl.to_numpy() for s0 in CS}, index=fr.index)
    d0 = pd.Timestamp(pd.to_datetime(start_date).normalize())
    out = None''')
g0 = s.index('r"""', s.index("_GBDT = ")) + 4
g1 = s.index('"""', g0)
s = s[:g0] + json.dumps(M, ensure_ascii=False, separators=(",", ":")) + s[g1:]
open(DST, "w", encoding="utf-8").write(s)
print("写出", DST, "特征", len(fo), "树", len(M["trees"]), "lr", M["lr"], "depth", M["depth"])
