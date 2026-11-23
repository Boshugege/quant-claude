# -*- coding: utf-8 -*-
"""拆解本地与平台自测的差异：同一区间（2024-01-02 至 03-01），
A = 本地因子值（factor_lib，30 分钟 bar）；B = 平台因子值（平台探测导出的 probe_fac_*.parquet）。
两者都用本地评分器（平台口径：官方标签 + 10 风格与 32 个行业剔除）评估，并与平台自测结果对比；同时比较 A、B 的截面秩相关。
用法：bash 工具/本地评分/remote.sh e2e/scripts/plat_fac_eval.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "工具/本地评分")
import localeval as le  # noqa: E402
from factor_lib import FACTORS, add_components  # noqa: E402

PLAT = {"90": (0.0633, 0.885, 35.2, 0.739, 0.608), "98": (0.0573, 0.783, 23.8, 0.577, 0.596),
        "100": (0.0569, 0.757, 25.4, 0.564, 0.635), "102": (0.0150, 0.235, 5.0, 0.188, 0.101),
        "104": (0.0659, 1.088, 35.8, 0.937, 0.615)}

p = add_components(le.build_panel(le.load_bars("2023-12", "2024-03")))   # 多取一个月做预热
win = (p["key_date"] >= "2024-01-02") & (p["key_date"] <= "2024-03-01 23:59:59")
p["is_eval"] = p["is_eval"] & win
mp = pd.read_csv("数据/exposure/id_map.csv")
mp = mp[mp.ok]
mp["instrument_id"] = mp.local_id.str[2:].astype(int)


facs = {f[0]: f for f in FACTORS}
rows = []
for fid, v in PLAT.items():
    rows.append({"id": fid, "src": "平台自测", **dict(zip(["ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"], v))})
    a = facs[fid][3](p)
    pf = pd.read_parquet(f"数据/平台探测/probe_fac_{fid}.parquet").merge(mp[["instrument", "instrument_id"]], on="instrument")
    pf = pf.rename(columns={"date": "key_date"})
    bvals = p[["key_date", "instrument_id"]].merge(pf[["key_date", "instrument_id", "factor"]], on=["key_date", "instrument_id"], how="left")["factor"]
    bser = pd.Series(bvals.to_numpy(), index=p.index)
    d = pd.DataFrame({"k": p["key_date"], "a": a, "b": bser})[p["is_eval"].to_numpy()]
    rc = d.dropna().groupby("k").apply(lambda g: g["a"].rank().corr(g["b"].rank())).mean()
    for name, f in [("本地因子", a), ("平台因子", bser)]:
        r = le.evaluate(f, p)
        rows.append({"id": fid, "src": name + " 本地评测", **{k: r[k] for k in ["ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"]},
                     "a_b_rankcorr": rc})
    # 不剔除风格的平台因子，看风格剔除这一步的影响
    r = le.evaluate(bser, p, neutralize=False)
    rows.append({"id": fid, "src": "平台因子 不剔除风格", **{k: r[k] for k in ["ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"]}})
    print(fid, "done", flush=True)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(4).to_string(index=False))
