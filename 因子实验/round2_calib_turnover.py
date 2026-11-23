# -*- coding: utf-8 -*-
"""§1.2 换手校准：用第一轮已提交的 values 任务，拟合「平台 Δ换手 = a·离线 Δ换手 + b」。

离线 Δ换手 = approx_turnover(合成值) − approx_turnover(H原始值)，合成值与提交的 values 任务同口径
（B = H原始值 + w·(rank(X)−0.5)，剔除 09:30）。平台 Δ换手 = 该 values 任务 result.json 的 turnover − 490 的 turnover。
只用 raw 口径任务（名字含 `_raw_w`）。用法：agent-run python 因子实验/round2_calib_turnover.py
"""
import glob
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/quant-claude/因子实验")
import root_eval as RE  # noqa: E402

H = "fm_490_indneu25"
tk490 = 0.55106  # 490 的平台换手（result.json，2.5.3 复评）


def parse_tag_w(name):
    m = re.match(r"v_(.+)_raw_w([pm])(\d+)$", name)
    if not m:
        return None
    tag, sign, digits = m.groups()
    w = int(digits) / 10 ** (len(digits) - 1)
    if sign == "m":
        w = -w
    tag = re.sub(r"raw$", "", tag)
    mm = re.match(r"([A-Za-z]*)(\d+)$", tag)
    if not mm:
        return None
    return ("fm_" + mm.group(2)), w


def main():
    h_raw, _, _ = RE.load(H)
    c = RE.core()
    ok, did, slot = RE.base_rows("2023-01-01", "2024-12-31")
    rows = ok & np.isfinite(h_raw)
    idx = np.flatnonzero(rows)
    days = pd.to_datetime([pd.Timestamp(int(c.dates[int(k)])) for k in did[idx]]).normalize()
    slots = slot[idx]
    inst = c.inst[idx]
    base_turn = RE.approx_turnover(h_raw[idx], did[idx], days, slots, inst)
    print("离线 H 近似换手 %.4f（平台 %.4f）" % (base_turn, tk490))

    xcache = {}
    pts = []
    for d in sorted(glob.glob("/home/user/jyc_scorer/jobs/*/spec.json")):
        j = json.load(open(d, encoding="utf-8"))
        name = j.get("name", "")
        pw = parse_tag_w(name)
        if not pw:
            continue
        rp = pathlib.Path(d).parent / "result.json"
        if not rp.exists():
            continue
        xname, w = pw
        if xname not in xcache:
            try:
                xcache[xname] = RE.load(xname)[0]
            except SystemExit:
                continue
        x = xcache[xname]
        ok2 = rows & np.isfinite(x)
        i2 = np.flatnonzero(ok2)
        rx = pd.Series(x[i2]).groupby(did[i2]).rank(pct=True).to_numpy() - 0.5
        b = h_raw[i2] + w * rx
        off = RE.approx_turnover(b, did[i2],
                                 pd.to_datetime([pd.Timestamp(int(c.dates[int(k)])) for k in did[i2]]).normalize(),
                                 slot[i2], c.inst[i2])
        plat = json.load(open(rp, encoding="utf-8"))["metrics"]["turnover"]
        pts.append((name, xname, w, off - base_turn, plat - tk490))
        print("  %-26s X=%-8s w=%+.3f 离线Δ%+.4f 平台Δ%+.4f" % (name, xname, w, off - base_turn, plat - tk490))

    A = np.array([[p[3], 1.0] for p in pts])
    y = np.array([p[4] for p in pts])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print("\n=== 拟合 平台Δ换手 = %.4f·离线Δ换手 + %.4f ；R²=%.4f ；n=%d ===" % (coef[0], coef[1], ss, len(pts)))
    print(json.dumps({"a": float(coef[0]), "b": float(coef[1]), "r2": float(ss), "n": len(pts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
