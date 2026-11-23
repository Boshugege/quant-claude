# -*- coding: utf-8 -*-
"""同伴挑选 + 行业屏蔽片段的形状自测：直接从因子文件里抽出真实循环体执行。

用法：python3 因子实验/test_peer_select.py 因子代码/132_行业内部软同伴.py
合成一个「部分股票历史不足（NaN）」的面板，验证 ok/ind_wide 的列筛选形状一致。
"""
import pathlib
import re
import sys
import textwrap

import numpy as np
import pandas as pd

src = pathlib.Path(sys.argv[1]).read_text()
start = src.index("    for m in out_months:")
end = src.index("    df = df.replace([np.inf, -np.inf], np.nan)")
body = textwrap.dedent(src[start:end])
m = re.search(r"^\s*(PEER_\w+(?:\s*,\s*PEER_\w+)*)\s*=\s*([0-9.,\s]+)$", src, flags=re.M)
assert m, "找不到 PEER_* 参数行"
names = [x.strip() for x in m.group(1).split(",")]
raw = [x.strip() for x in m.group(2).split(",")]
consts = {n: (float(v) if "." in v else int(v)) for n, v in zip(names, raw)}

rng = np.random.default_rng(11)
n_inst, n_day = 900, 900
cols = ["%06d.SZ" % i for i in range(n_inst)]
days = pd.date_range("2021-01-04", periods=n_day, freq="B")
ret = pd.DataFrame(rng.normal(0, 0.02, (n_day, n_inst)), index=days, columns=cols)
ret.iloc[:25, :60] = np.nan                      # 前 60 只历史不足
ret.iloc[rng.random((n_day, n_inst)) < 0.05] = np.nan
ind = pd.DataFrame({c: rng.choice(["A", "B", "C"], n_day) for c in cols}, index=days)
ns = {"np": np, "pd": pd, "daily_ret": ret, "ind_wide": ind, "peer_insts": np.array(cols),
      "all_days": ret.index, "day_month": ret.index.to_period("M"),
      "out_months": pd.period_range("2024-01", "2024-02", freq="M"), "peers_by_month": {},
      "in_month": None}
mh = re.search(r"^\s*HIST_HALFLIFE\s*=\s*([0-9.]+)", src, flags=re.M)
if mh:
    consts["HIST_HALFLIFE"] = float(mh.group(1))
ns.update(consts)
exec("def _run():\n" + textwrap.indent(body, "    ") + "\n    return peers_by_month\n", ns)
got = ns["_run"]()
assert got, "没有产出任何月份的同伴表"
idx, w = got[list(got)[0]]
assert idx.shape == (n_inst, int(consts["PEER_K"])), idx.shape
assert w.shape == idx.shape
assert w.sum(axis=1).max() <= 1 + 1e-9
# 行业屏蔽是否真的生效：逐月取该月实际使用的行业码（月初之前最近一个交易日），正权重同伴应同行业
K = int(consts["PEER_K"])
has_mask = "_same" in body
n_eff = 0
for mon, (idx, w) in got.items():
    last_day = ind.index[ind.index < mon.start_time][-1]
    codes = ind.loc[last_day].values
    for i_ in range(n_inst):
        for k in range(K):
            if idx[i_, k] >= 0 and w[i_, k] > 0:
                n_eff += 1
                if has_mask:
                    j_ = idx[i_, k]
                    assert codes[j_] == codes[i_], (mon, i_, j_, codes[i_], codes[j_])
if has_mask:
    assert n_eff > 0, "没有任何正权重同伴，屏蔽逻辑可疑"
print("同伴挑选片段自测通过：月份数 %d，正权重同伴数 %d，权重行和<=1，%s" % (len(got), n_eff, "行业屏蔽生效" if has_mask else "无行业屏蔽"))
