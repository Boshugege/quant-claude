#!/usr/bin/env python
"""重算候选因子与参照因子（默认 104）的截面秩相关，**剔除 09:30 截面**。

为什么需要它：`result.json -> correlations[fid].rank_corr_mean` 把 09:30 这类
"候选因子在该截面近似常数"的截面也算进平均。这种截面的逐截面相关系数在浮点上是
垃圾值（实测恒为 +1），会把所有因子的相关往 +1 拉，而且**不随因子取反变号**——
实测 468（−0.591）取反成 495 后不是 +0.591 而是 +0.841，两者数值上是精确取反的。
所以判"是否新信息源"要用剔除这些截面后的数值；本脚本按与评测机相同的
`per_date_rank_corr` 口径算，只把 09:30 以外的截面留下来。

用法：`/home/user/jyc_scorer/.venv/bin/python 因子实验/corr_excl0930.py <任务号> [<任务号> ...] [--ref 104]`
"""
import sys, pathlib
import numpy as np
import pandas as pd

SCORER = pathlib.Path("/home/user/jyc_scorer")
sys.path.insert(0, str(SCORER))
import engine, refs, config as C  # noqa: E402
sys.path.insert(0, str(SCORER))
sys.path.insert(0, str(SCORER / "build"))
import runner  # noqa: E402  依然可用：to_core_keys 逻辑在这里

RES = pathlib.Path("/home/user/quant-claude/工具/评分机/结果")


def to_keys(d):
    m = pd.read_csv(C.DATA / "exposure" / "id_map.csv", dtype=str)
    m = m[m["ok"].str.lower() == "true"]
    real2id = dict(zip(m["instrument"], m["local_id"]))
    inst = d["instrument"]
    is_pseudo = inst.str.match(r"^ID\d{4}$")
    pseudo = inst.where(is_pseudo, inst.map(real2id))
    ok = pseudo.notna()
    return (d.loc[ok, "date"].to_numpy("datetime64[ns]").astype(np.int64),
            pseudo[ok].str[2:].astype(np.int32).to_numpy(),
            d.loc[ok, "factor"].to_numpy(np.float64))


def main(argv):
    ref_fid = "104"
    if "--ref" in argv:
        i = argv.index("--ref"); ref_fid = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    core = engine.Core(str(C.CORE_DIR))
    pool = refs.RefPool()
    ref = np.asarray(pool.values[pool.fids.index(ref_fid)], np.float64)
    dm = core.date_range_mask("2023-01-01", "2024-12-31")
    for jid in argv:
        p = RES / jid / "factor_raw.parquet"
        if not p.exists():
            print("%s  缺 factor_raw.parquet" % jid); continue
        d = pd.read_parquet(p, columns=["date", "instrument", "factor"])
        kd, ii, v = to_keys(d)
        arr, hit = core.align(kd, ii, v)
        ok = np.isfinite(arr) & np.isfinite(ref) & dm[core.date_id]
        sel = np.flatnonzero(ok); g = core.date_id[sel]; D = core.n_dates
        ra, n = engine._group_rank_pct(g, arr[sel], D)
        rb, _ = engine._group_rank_pct(g, ref[sel], D)
        nz = np.maximum(n, 1)
        xa = ra - (np.bincount(g, ra, D) / nz)[g]
        xb = rb - (np.bincount(g, rb, D) / nz)[g]
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.bincount(g, xa * xb, D) / np.sqrt(np.bincount(g, xa * xa, D) * np.bincount(g, xb * xb, D))
        hms = pd.to_datetime(core.dates).strftime("%H%M").to_numpy()
        keep = (n >= 3) & np.isfinite(c)
        keep_all = keep.sum(); ex = keep & (hms != "0930")
        print("%s  含 09:30=%.4f（%d 截面）  剔除后=%.4f（%d 截面）  与 %s 的相关" % (
            jid, float(c[keep].mean()), int(keep_all), float(c[ex].mean()), int(ex.sum()), ref_fid))


if __name__ == "__main__":
    main(sys.argv[1:])
