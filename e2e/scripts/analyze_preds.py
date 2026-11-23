# -*- coding: utf-8 -*-
"""诊断：按平台口径（官方标签 + 10 风格与 32 个行业剔除，见 工具/本地评分/localeval.py）评估端到端预测，并与传统因子对比。

输入是原始预测 DataFrame[date, instrument, factor]，覆盖区间内全部截面。
用法：
  python e2e/scripts/analyze_preds.py <json 或 预测 parquet> <start YYYY-MM> <end YYYY-MM> [tag]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
sys.path.insert(0, "工具/本地评分")
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend  # noqa: E402
import localeval as le  # noqa: E402
from factor_lib import FACTORS, add_components  # noqa: E402

OUT_DIR = os.path.join("e2e", "runs", "analysis")
_PANEL = {}


def month_end(m):
    return (pd.Timestamp(f"{m}-01") + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d 23:59:59")


def predict_json(path, start, end):
    ck = T.load_model(path)
    T.configure_from_ckpt(ck)
    dev = T.pick_device()
    models = T.build_models(ck, dev)
    stats = T.load_stats(ck)
    fb = FeatherBackend()
    parts = []
    for lo, hi in T.month_chunks(f"{start}-01", month_end(end)):
        parts.append(T.predict_range(models, fb, str(lo), str(hi), fb.pool(str(lo), str(hi)), stats, dev))
    return pd.concat(parts, ignore_index=True)


def panel_for(start, end):
    if (start, end) not in _PANEL:
        _PANEL[(start, end)] = add_components(le.build_panel(le.load_bars(start, end)))
    return _PANEL[(start, end)]


def analyze(pred, start, end, tag, refs=("90", "104"), raw=False):
    """平台口径五项指标；raw=True 时另报不剔除风格的结果（行名加 _raw，只作对照）。"""
    panel = panel_for(start, end)
    pred = pred.sort_values("date").reset_index(drop=True)

    def align(df):
        m = panel[["key_date", "instrument"]].merge(df.rename(columns={"date": "key_date"}), how="left",
                                                    on=["key_date", "instrument"])
        return pd.Series(m["factor"].to_numpy(), index=panel.index)

    cands = {"E2E": align(pred)}
    facs = {f[0]: f for f in FACTORS}
    for fid in refs:
        if fid in facs:
            try:
                cands[f"F{fid}"] = facs[fid][3](panel)
            except Exception as e:   # factor_lib 的同伴函数在 pandas 3 下会报错（需 .venv），跳过对照
                print(f"对照因子 {fid} 跳过：{e!r}")
    rows = []
    for name, f in cands.items():
        for suffix, neu in [("", True)] + ([("_raw", False)] if raw else []):
            r = le.evaluate(f, panel, neutralize=neu)
            rows.append({"name": name + suffix, **{k: r[k] for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")},
                         "cov": float(f[panel["is_eval"]].notna().mean()), **{f"s{k}": v for k, v in r["slots"].items()}})
    t = pd.DataFrame(rows).set_index("name")
    if "F90" in cands:
        ev = panel["is_eval"]
        c = pd.DataFrame({"m": cands["E2E"][ev], "f": cands["F90"][ev], "d": panel.loc[ev, "key_date"]}).dropna()
        t["corr_F90"] = np.nan
        t.loc["E2E", "corr_F90"] = c.groupby("d").apply(lambda g: g["m"].rank().corr(g["f"].rank())).mean()
    os.makedirs(OUT_DIR, exist_ok=True)
    t.to_csv(os.path.join(OUT_DIR, f"metrics_{tag}_{start}_{end}.csv"))
    pd.set_option("display.width", 250)
    print(f"== {tag} {start}..{end}")
    print(t.round(4).to_string())
    return t


if __name__ == "__main__":
    src, start, end = sys.argv[1:4]
    tag = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(os.path.dirname(src))
    if src.endswith(".parquet"):
        pred = pd.read_parquet(src)
    else:
        t0 = time.time()
        pred = predict_json(src, start, end)
        os.makedirs(OUT_DIR, exist_ok=True)
        pred.to_parquet(os.path.join(OUT_DIR, f"pred_{tag}_{start}_{end}.parquet"))
        print(f"predicted {len(pred):,} rows in {time.time() - t0:.0f}s")
    analyze(pred, start, end, tag)
