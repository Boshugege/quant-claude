# -*- coding: utf-8 -*-
"""第 3 级提速（任务书 §4.2）的离线预筛：用评测机同款中性化（engine.Core.residual）+
逐日 IC（与 result.json 完全同口径，已用 490 raw 复核 = 0.07716）预测合成版的平台 IC。

两种构造：
  raw  —— B = 头部原始值 + w·(rank(X)-0.5)   （不破坏头部原本的分布，实测基线不掉）
  rank —— B = (rank(头部)-0.5) + w·(rank(X)-0.5)（任务书 §4.2 字面口径，会自带约 -0.0028 的基线损失）

用法：predict_grid.py --x fm_1124,fm_1100 --ws=-0.5,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.5
"""
import argparse, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R

c = R.core()


def ranks(arr, dates, slot):
    k = pd.factorize(np.array([c.dates[int(x)] for x in dates], dtype="int64") * 1000 +
                     slot.astype("int64"))[0]
    df = pd.DataFrame({"k": k, "f": arr})
    return (df.groupby("k")["f"].rank(pct=True) - 0.5).to_numpy()


def ic_of_raw(B):
    return R.nw_t(R.ic_of(R.platform_res(B)))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="fm_490_indneu25")
    ap.add_argument("--x", required=True)
    ap.add_argument("--ws", required=True)
    ap.add_argument("--modes", default="raw")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2024-12-31")
    a = ap.parse_args()
    ok, did, slot = R.base_rows(a.start, a.end)
    fh, _, _ = R.load(a.h)
    idxh = np.flatnonzero(ok & np.isfinite(fh))
    ic_head_raw = ic_of_raw(fh)
    pr = np.full(c.n_rows, np.nan); pr[idxh] = ranks(fh[idxh], did[idxh], slot[idxh])
    ic_head_rank = ic_of_raw(pr)
    print("头部 %s：raw 口径 IC %.5f，rank 口径 IC %.5f（差 %+.5f）" %
          (a.h, ic_head_raw, ic_head_rank, ic_head_rank - ic_head_raw))
    ws = [float(x) for x in a.ws.split(",")]
    rows = []
    for name in a.x.split(","):
        fx, _, _ = R.load(name)
        ix = np.flatnonzero(ok & np.isfinite(fx))
        q = np.full(c.n_rows, np.nan); q[ix] = ranks(fx[ix], did[ix], slot[ix])
        for w in ws:
            for tag, head in [x for x in (("raw", fh), ("rank", pr)) if x[0] in a.modes]:
                B = np.where(np.isfinite(head) & np.isfinite(q), np.nan_to_num(head) + w * np.nan_to_num(q), np.nan)
                ic = ic_of_raw(B)
                rows.append((name, tag, w, ic, ic - (ic_head_raw if tag == "raw" else ic_head_rank)))
    df = pd.DataFrame(rows, columns=["x", "构造", "w", "预测IC", "Δ"])
    for name in a.x.split(","):
        s = df[df["x"] == name]
        best = s.loc[s["预测IC"].idxmax()] if len(s) else None
        print("== %s ==" % name)
        print(s.to_string(index=False, float_format=lambda v: "%.5f" % v))
        if best is not None:
            print("   最佳：%s 构造 %s w=%+.2f 预测IC %.5f Δ%+.5f" %
                  (name, best["构造"], best["w"], best["预测IC"], best["Δ"]))


if __name__ == "__main__":
    main()
