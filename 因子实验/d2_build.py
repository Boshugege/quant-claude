# -*- coding: utf-8 -*-
"""D2 全局最优解 → values 任务（用评测机出正式五项 + 公榜估计）。

δ 向量来自 `因子实验/d2_optimize.py`（拟合 2020-01～2022-12，验证 2023-24）的最优解：
 B = 1363原始值 + Σ_j δ_j·rank_j，其中 1363 已经含九根在 1363 里的原权重。
"""
import argparse
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_optimize as O  # noqa: E402
from build_values_composite import read_raw  # noqa: E402

URL = "http://127.0.0.1:46220/api/jobs"
OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
# 标识: (fm任务, o21任务, o20任务, EWMAα, δ)
DELTA = [
    ("1232", "fm_1232", "o21_1232", "o20_1232", 0.75, +0.0614),
    ("1236", "fm_1236", "o21_1236", "o20_1236", 0.0, +0.0139),
    ("1237", "fm_1237", "o21_1237", "o20_1237", 0.0, +0.0073),
    ("1260", "fm_1260", "o21_1260", "o20_1260", 0.0, -0.0288),
    ("1263", "fm_1263", "o21_1263", "o20_1263", 0.0, +0.0939),
    ("1323", "fm_1323", "o21_1323", "o20_1323", 0.0, +0.0085),
    ("1273", "fm_1273", "o21_1273", "o20_1273", 0.0, -0.0170),
    ("1334", "fm_1334", "o21_1334", "o20_1334", 0.0, -0.1583),
    ("1342", "fm_1342", "o21_1342", "o20_1342", 0.0, +0.0076),
    ("s155", "fm_s155", "o21_s155", "o20_s155", 0.0, -0.1825),
    ("s139", "fm_s139", "o21_s139", "o20_s139", 0.0, +0.3831),
    ("s161", "fm_s161", "o21_s161", "o20_s161", 0.0, +0.3000),
    ("1381", "fm_1381", "o21_1381", "o20_1381", 0.0, +0.0177),
    ("1385", "fm_1385", "o21_1385", "o20_1385", 0.0, +0.0275),
    ("1406", "fm_1406", "o21_1406", "o20_1406", 0.0, +0.0063),
]


def inst_codes(c):
    """core.inst 是内部数字 id，提交给平台的 values 文件要写股票代码字符串；对照表不随仓库发布。"""
    m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv",
                    usecols=["local_id", "instrument"])
    num = m["local_id"].str.replace("ID", "", regex=False).astype(int)
    tab = pd.Series(m["instrument"].to_numpy(), index=num.to_numpy())
    return tab.reindex(c.inst).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="1015-203626-f5f0", help="底座（主区间 1363 values 任务号）")
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--suffix", default="", help="o21 / o20 时给轴任务加的后缀")
    ap.add_argument("--name", default="v_D2OPT_raw")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    c = R.core()
    st = a.start + "-01" if len(a.start) == 7 else a.start
    en = a.end + "-31" if len(a.end) == 7 else a.end
    hf = R.load(a.h)[0]
    B = np.asarray(hf, float).copy()
    for lab, n1, n2, n3, al, d in DELTA:
        nm = n1 if not a.suffix else {"o21": n2, "o20": n3}[a.suffix]
        if abs(d) < 1e-9:
            continue
        ax = O.axis(c, nm, st, en, al)
        B = B + d * np.nan_to_num(ax, nan=0.0)
    B[c.slot_of_date[c.date_id] == 930] = 0.0
    m, _ = c.evaluate(B, start=st, end=en)
    print("合成 %s：IC %.5f IR %.4f 压力 %.4f 换手 %.4f 截面 %d"
          % (a.name, m["ic_mean"], m["ic_ir"], m["stress"], m["turnover"], m["n_xsec"]))
    o = pd.DataFrame({"date": pd.to_datetime(c.key_date), "instrument": inst_codes(c),
                      "factor": np.nan_to_num(B, nan=0.0)})
    o = o[np.isfinite(np.asarray(hf, float))]
    # id_map 里同一个 instrument 代码有多个 local_id（176 个），映射后会出现重复 (date, instrument)，
    # 平台对重复行会直接报错不给分；这里按 (date, instrument) 去重保留最后一条。
    o = o.drop_duplicates(["date", "instrument"], keep="last").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / (a.name + ".parquet")
    o.to_parquet(f, index=False)
    print("写出 %s rows=%d" % (f, len(o)))
    if a.submit:
        r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values",
                            "-F", "start=%s" % a.start, "-F", "end=%s" % a.end,
                            "-F", "name=%s" % a.name, "-F", "baseline=%s" % a.h, URL],
                           capture_output=True, text=True)
        print("提交 ->", r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()
