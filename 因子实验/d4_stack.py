# -*- coding: utf-8 -*-
"""D4 第三步：在底座 `1363 + s155×0.5` 上叠候选头部，本地快扫 + 提交。

底座 v_SB155_raw_wp0500（主区间）。候选是 D4 残差相关矩阵里与底座重叠最小的几个。
本地用 d2_eval 的残差快路径（与评测机 series() 同口径），扫 w 后再用 values 任务出正式五项。
09:30 截面按合成脚本口径置 0。
"""
import argparse
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_eval as D  # noqa: E402

URL = "http://127.0.0.1:46220/api/jobs"
OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
CAND = ["fm_390_ewma_slotw", "fm_s137", "fm_s139", "fm_233_win_upsh30",
        "fm_s160", "fm_s161", "fm_1083", "fm_1176", "fm_1187"]


def inst_codes(c):
    m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", usecols=["local_id", "instrument"])
    num = m["local_id"].str.replace("ID", "", regex=False).astype(int)
    tab = pd.Series(m["instrument"].to_numpy(), index=num.to_numpy())
    return tab.reindex(c.inst).to_numpy()


def rank_axis_zeroed(f, c):
    out = D.rank_axis(f)
    out[c.slot_of_date[c.date_id] == 930] = 0.0
    return out


def submit(parquet, name, start, end, baseline):
    cmd = ["curl", "-s", "-F", "files=@%s" % parquet, "-F", "kind=values",
           "-F", "start=%s" % start, "-F", "end=%s" % end,
           "-F", "name=%s" % name, "-F", "baseline=%s" % baseline, URL]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("提交 %s -> %s" % (name, r.stdout.strip() or r.stderr.strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v_SB155_raw_wp0500")
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--ws", default="0.05,0.1,0.2,0.3")
    ap.add_argument("--submit-w", default="", help="逗号分隔：只提交这些权重（默认不提交）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cands", default="")
    a = ap.parse_args()
    c = R.core()
    st = a.start + "-01" if len(a.start) == 7 else a.start
    en = a.end + "-31" if len(a.end) == 7 else a.end
    ws = [float(x) for x in a.ws.split(",")]
    subw = [float(x) for x in a.submit_w.split(",")] if a.submit_w else []
    cands = a.cands.split(",") if a.cands else CAND

    hf = D.aligned(a.base)
    base_res = D.resid_axis(hf, st, en)
    m0, _ = D.eval_mix(base_res, [], st, en)
    print("底座 %s：%s" % (a.base, D.show("base", m0)))
    rows = []
    for nm in cands:
        try:
            f = D.aligned(nm)
        except SystemExit:
            print("跳过（找不到）", nm)
            continue
        # 候选可能只有部分区间：缺失当 0
        f = np.where(np.isfinite(f), f, 0.0)
        Rj = D.rank_axis(f)
        Rj[c.slot_of_date[c.date_id] == 930] = 0.0
        for w in ws:
            m, _ = D.eval_mix(base_res, [(w, Rj)], st, en)
            d = {k: m[k] - m0[k] for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")}
            rows.append((nm, w, m, d))
            print("  %-18s w=%.2f  %s  Δ IC %+.5f IR %+.4f LS %+.2f 压力 %+.4f 换手 %+.4f"
                  % (nm, w, D.show("", m)[1:], d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"]))
        if subw:
            for w in subw:
                v = hf + w * np.nan_to_num(Rj, nan=0.0)
                v[c.slot_of_date[c.date_id] == 930] = 0.0
                o = pd.DataFrame({"date": pd.to_datetime(c.key_date), "instrument": inst_codes(c),
                                  "factor": np.nan_to_num(v, nan=0.0)})
                o = o[np.isfinite(np.asarray(hf, float))]
                nmout = "v_D4%s_%s_w%s" % (a.tag or nm.replace("fm_", ""), a.start.replace("-", ""),
                                           ("%+.3f" % w).replace("+", "p").replace("-", "m").replace(".", ""))
                fo = OUT / (nmout + ".parquet")
                o.to_parquet(fo, index=False)
                submit(str(fo), nmout, a.start, a.end, a.base)
                fo.unlink(missing_ok=True)
    rows.sort(key=lambda r: -(r[3]["ic_ir"] + r[3]["stress"] - 0.2 * r[3]["turnover"]))
    print("\n按 J = ΔIR + Δ压力 − 0.2·Δ换手 排序（前 8）：")
    for nm, w, m, d in rows[:8]:
        print("  %-18s w=%.2f  J %+.4f  ΔIC %+.5f ΔIR %+.4f Δ换手 %+.4f" %
              (nm, w, d["ic_ir"] + d["stress"] - 0.2 * d["turnover"], d["ic_mean"], d["ic_ir"], d["turnover"]))


if __name__ == "__main__":
    main()
