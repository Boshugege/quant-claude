# -*- coding: utf-8 -*-
"""评分机验收（远端 ~/jyc_scorer 下运行：.venv/bin/python acceptance.py [--quick]）。结果写 logs/acceptance.json。

1. 评分引擎 = localeval.evaluate：2024 年的面板，因子 90、随机因子（含缺失、inf、缺行）、不剔除风格、半年子区间，
   五项指标逐项比较，要求差异 < 1e-9（换手、压力是同样的离散计算，应完全相同）。
2. 复现平台自测：平台处理后的因子值（探测 02）+ 官方标签，2024-01-02~03-01，与平台 jyc_eval 结果比较。
3. 模拟平台 = 平台：用模拟平台原样运行 因子代码/90、104 的提交 .py（2024-01~02），与平台导出的因子值（探测 01）
   逐截面比较秩相关；09:30 截面应与平台一样全为 0（平台 SQL 的 09:30 缺陷）。
4. 平台规则检查能抓出坏输入：重复键、inf、截面外时刻、某截面缺失 50%。
5. 真实代码与伪代码输入的结果相同。
6. 端到端提交（--e2e 指定 zip）：推理一个月，输出满足平台格式。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import engine  # noqa: E402
import panel_store  # noqa: E402

PROBE = C.DATA / "platform_probe"
PLAT = {"90": dict(ic_mean=0.0633322259124647, ic_ir=0.884819179988961, ls_sharpe=35.19718951107968, stress=0.7393099815983497, turnover=0.6078728070175439),
        "104": dict(ic_mean=0.06588155783363826, ic_ir=1.0881031767688831, ls_sharpe=35.77950285632168, stress=0.9374708446856378, turnover=0.6149122807017543)}
M5 = ["ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"]
OUT = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def maxdiff(a, b):
    return max(abs(a[k] - b[k]) for k in M5 if np.isfinite(a[k]) or np.isfinite(b[k]))


def test1(core):
    import localeval as le
    from factor_lib import FACTORS, add_components
    panel_store.configure_localeval()
    t0 = time.time()
    p = add_components(le.build_panel(le.load_bars("2024-01", "2024-12", data_dir=le.DATA_DIR)))
    log(f"[1] localeval 面板 2024：{time.time() - t0:.0f}s")
    rng = np.random.default_rng(0)
    fac = {str(x[0]): x for x in FACTORS}
    f90 = np.asarray(fac["90"][3](p), float)
    noisy = f90 + rng.normal(0, 0.5, len(p))
    noisy[rng.random(len(p)) < 0.05] = np.nan
    noisy[rng.random(len(p)) < 0.001] = np.inf
    cases = {"因子 90": (f90, True, None), "随机扰动（含缺失、inf）": (noisy, True, None), "不剔除风格": (f90, False, None),
             "2024H1 子区间": (f90, True, ("2024-01-01", "2024-06-30 23:59:59"))}
    res = {}
    kd = p["key_date"].to_numpy("datetime64[ns]").astype(np.int64)
    inst = p["instrument_id"].to_numpy()
    for name, (f, neu, rg) in cases.items():
        q = p
        if rg:
            q = p.copy()
            q["is_eval"] = q["is_eval"] & q["key_date"].between(*rg)
        t1 = time.time()
        ref = le.evaluate(f, q, neutralize=neu)
        t_le = time.time() - t1
        v, _ = core.align(kd, inst, f)
        t1 = time.time()
        got, _ = core.evaluate(v, rg[0] if rg else "2024-01-01", rg[1] if rg else "2024-12-31 23:59:59", neutralize=neu)
        t_en = time.time() - t1
        d = maxdiff(ref, got)
        res[name] = {"localeval": {k: ref[k] for k in M5}, "engine": {k: got[k] for k in M5}, "max_abs_diff": d,
                     "n_xsec": [int(ref["n_xsec"]), int(got["n_xsec"])], "seconds_localeval": round(t_le, 1), "seconds_engine": round(t_en, 2)}
        log(f"[1] {name}: 最大差 {d:.2e}，截面 {ref['n_xsec']}/{got['n_xsec']}，localeval {t_le:.1f}s → 引擎 {t_en:.2f}s")
    OUT["1_engine_equals_localeval"] = {"pass": all(r["max_abs_diff"] < 1e-9 and r["n_xsec"][0] == r["n_xsec"][1] for r in res.values()), "cases": res}


def test2(core):
    import scoring
    y = scoring.official_label(core)
    m = pd.read_csv(C.DATA / "exposure" / "id_map.csv", dtype=str)
    r2i = dict(zip(m.loc[m["ok"].str.lower() == "true", "instrument"], m.loc[m["ok"].str.lower() == "true", "local_id"]))
    res = {}
    for fid in ("90", "104"):
        d = pd.read_parquet(PROBE / f"probe2_process_factor_{fid}.parquet")
        ids = d["instrument"].map(r2i)
        ok = ids.notna()
        v, _ = core.align(d.loc[ok, "date"].to_numpy("datetime64[ns]").astype(np.int64), ids[ok].str[2:].astype(int).to_numpy(), d.loc[ok, "factor"].to_numpy())
        v = np.where(np.isfinite(v), v, np.nan)
        got, _ = core.evaluate(v, "2024-01-02", "2024-03-01 23:59:59", y=y)
        res[fid] = {"platform": PLAT[fid], "engine": {k: got[k] for k in M5}, "diff": {k: got[k] - PLAT[fid][k] for k in M5}}
        log(f"[2] {fid}: " + "  ".join(f"{k} {got[k]:.4f}/{PLAT[fid][k]:.4f}" for k in M5))
    OUT["2_reproduce_platform_selftest"] = {"pass": all(abs(r["diff"]["ic_mean"]) < 0.003 and abs(r["diff"]["ic_ir"]) < 0.05 for r in res.values()),
                                            "note": "本地缺少约 7% 股票的伪代码映射，误差主要来自这部分股票", "factors": res}


def run_job(kind, files, start, end, name, extra=None):
    jd = C.JOBS / f"accept-{name}-{int(time.time())}"
    (jd / "input").mkdir(parents=True)
    for f in files:
        shutil.copy(f, jd / "input" / Path(f).name)
    spec = {"kind": kind, "inputs": [Path(f).name for f in files], "start": start, "end": end, "label": "y_off", "chunk": "none", "name": name, **(extra or {})}
    (jd / "spec.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    t0 = time.time()
    subprocess.call([sys.executable, str(HERE / "runner.py"), "--job", str(jd)], stdout=open(jd / "log.txt", "w"), stderr=subprocess.STDOUT)
    prod = json.loads((jd / "produce.json").read_text(encoding="utf-8"))
    return jd, prod, time.time() - t0


def test3():
    res = {}
    for fid, fn in (("90", "90_多尺度反转阶梯.py"), ("104", "104_同伴残差阶梯过度中性.py")):
        src = HERE / "samples" / fn
        jd, prod, sec = run_job("platform", [src], "2024-01", "2024-02", f"plat{fid}")
        if not prod.get("ok"):
            res[fid] = {"error": prod.get("error")}
            log(f"[3] {fid} 运行失败：{prod.get('error')}")
            continue
        mine = pd.read_parquet(jd / "factor_raw.parquet")
        plat = pd.read_parquet(PROBE / f"probe_fac_{fid}.parquet")
        j = plat.merge(mine, on=["date", "instrument"], suffixes=("_plat", "_sim"))
        j["slot"] = j["date"].dt.strftime("%H:%M")
        rc = j.groupby("date").apply(lambda g: g["factor_plat"].rank().corr(g["factor_sim"].rank()), include_groups=False)
        by_slot = rc.groupby(rc.index.strftime("%H:%M")).mean()
        z930 = j[j["slot"] == "09:30"]
        res[fid] = {"rows_platform": len(plat), "rows_sim": len(mine), "rows_matched": len(j), "seconds": round(sec, 1),
                    "peak_gb": prod.get("produce_peak_gb"),
                    "rank_corr_mean_excl_0930": float(rc[rc.index.strftime("%H:%M") != "09:30"].mean()),
                    "rank_corr_by_slot": {k: (None if pd.isna(v) else float(v)) for k, v in by_slot.items()},
                    "abs_diff_median": float((j["factor_plat"] - j["factor_sim"]).abs().median()),
                    "share_exact_1e-9": float(((j["factor_plat"] - j["factor_sim"]).abs() < 1e-9).mean()),
                    "0930_all_zero_platform": bool((z930["factor_plat"].abs() < 1e-12).all()),
                    "0930_all_zero_sim": bool((z930["factor_sim"].abs() < 1e-12).all())}
        log(f"[3] {fid}: 匹配 {len(j):,}/{len(plat):,} 行，截面秩相关（不含 09:30）{res[fid]['rank_corr_mean_excl_0930']:.4f}，"
            f"完全相同比例 {res[fid]['share_exact_1e-9']:.3f}，09:30 全 0：平台 {res[fid]['0930_all_zero_platform']} / 模拟 {res[fid]['0930_all_zero_sim']}，{sec:.0f}s")
    OUT["3_platform_sim_equals_platform"] = {"pass": all(r.get("rank_corr_mean_excl_0930", 0) > 0.95 for r in res.values()), "factors": res}


def test4():
    import runner
    base = pd.read_parquet(PROBE / "probe_fac_90.parquet")
    base = base[base["date"] < "2024-01-10"]
    bad = {"重复键": pd.concat([base, base.head(50)]),
           "inf": base.assign(factor=np.where(np.arange(len(base)) < 10, np.inf, base["factor"])),
           "截面外时刻": pd.concat([base, base.head(20).assign(date=base.head(20)["date"] + pd.Timedelta("1min"))]),
           "某截面缺失 50%": base.drop(base[base["date"] == base["date"].iloc[5000]].sample(frac=0.5, random_state=0).index)}
    res = {}
    for k, df in bad.items():
        c, _ = runner.platform_checks(df, "2024-01", "2024-01")
        caught = {"重复键": c["duplicate_keys"] > 0, "inf": c["inf_count"] > 0, "截面外时刻": c["rows_outside_8_slots"] > 0,
                  "某截面缺失 50%": c.get("sections_over_40pct_missing", 0) > 0}[k]
        res[k] = {"caught": bool(caught), "pass_flag": c["pass"]}
        log(f"[4] {k}: {'抓到' if caught else '没抓到'}")
    OUT["4_bad_inputs"] = {"pass": all(r["caught"] and not r["pass_flag"] for r in res.values()), "cases": res}


def test5(core):
    import runner
    d = pd.read_parquet(PROBE / "probe_fac_104.parquet")
    c, dd = runner.platform_checks(d, "2024-01", "2024-03")
    k_real, _ = runner.to_core_keys(dd)
    m = pd.read_csv(C.DATA / "exposure" / "id_map.csv", dtype=str)
    m = m[m["ok"].str.lower() == "true"]
    dd2 = dd.assign(instrument=dd["instrument"].map(dict(zip(m["instrument"], m["local_id"])))).dropna(subset=["instrument"])
    k_ps, _ = runner.to_core_keys(dd2)
    r = []
    for k in (k_real, k_ps):
        v, _ = core.align(k["key_date"].to_numpy("datetime64[ns]").astype(np.int64), k["instrument_id"].to_numpy(), k["factor"].to_numpy())
        r.append(core.evaluate(v, "2024-01-02", "2024-03-01 23:59:59")[0])
    d5 = maxdiff(r[0], r[1])
    log(f"[5] 真实代码 vs 伪代码：最大差 {d5:.2e}")
    OUT["5_real_vs_pseudo_codes"] = {"pass": d5 == 0, "max_abs_diff": d5}


def test6(zip_path):
    jd, prod, sec = run_job("e2e", [zip_path], "2024-01", "2024-01", "e2e", {"chunk": "month"})
    c = prod.get("checks") or {}
    log(f"[6] 端到端推理 2024-01：{'成功' if prod.get('ok') else '失败 ' + str(prod.get('error'))}，{sec:.0f}s，平台规则 {'通过' if c.get('pass') else '未通过'}")
    OUT["6_e2e_inference"] = {"pass": bool(prod.get("ok") and c.get("pass")), "seconds": round(sec, 1), "peak_gb": prod.get("produce_peak_gb"), "checks": c,
                              "error": prod.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="跳过的测试编号，例如 1,3")
    ap.add_argument("--e2e", help="端到端 zip 路径（不给就跳过第 6 项）")
    a = ap.parse_args()
    skip = set(a.skip.split(","))
    core = engine.Core(C.CORE_DIR, workers=C.ENGINE_THREADS)
    for i, fn in (("1", lambda: test1(core)), ("2", lambda: test2(core)), ("3", test3), ("4", test4), ("5", lambda: test5(core))):
        if i not in skip:
            fn()
    if a.e2e and "6" not in skip:
        test6(a.e2e)
    OUT["all_pass"] = all(v["pass"] for k, v in OUT.items() if isinstance(v, dict) and "pass" in v)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "acceptance.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log("全部通过" if OUT["all_pass"] else "有未通过项", "→ logs/acceptance.json")


if __name__ == "__main__":
    main()
