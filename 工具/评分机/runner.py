# -*- coding: utf-8 -*-
"""任务执行器：把各种提交统一转成因子值 factor.parquet（date, instrument, factor），并按平台规则检查。

提交类型（kind）：
  values    因子值文件（parquet / csv，三列 date, instrument, factor）
  platform  传统因子提交（.py 或 .ipynb，定义 main(datasources, start_date, end_date)），在模拟平台上运行
  e2e       端到端提交（Transformer_modelsave_train.py + Transformer_modelsave_predict.ipynb + transformer_model.json，
            或三者的 zip），只运行推理 notebook 的 main
  factor_py 本地研究脚本（定义 factor(panel)，panel 为 localeval.build_panel 的结果）
  fid       factor_lib 已登记的编号
  sql       一条 DuckDB SQL（可引用平台表名），返回 date, instrument, factor
服务端以子进程运行本文件（内存上限与超时与平台一致），也可单独调试：
  python runner.py --job jobs/<id>
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import resource
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

SLOTS = [930, 1000, 1030, 1100, 1130, 1330, 1400, 1430]
COLS = ["date", "instrument", "factor"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def platform_env():
    """让 import dai / bigmodule 指向本地模拟。"""
    sim = str(HERE / "platform_sim")
    if sim not in sys.path:
        sys.path.insert(0, sim)
    os.environ.setdefault("JYC_PLATFORM_DIR", str(C.PLATFORM_DIR))
    os.environ.setdefault("JYC_EXPOSURE_DIR", str(C.DATA / "exposure"))
    os.environ.setdefault("JYC_DUCKDB_THREADS", str(C.DUCKDB_THREADS))
    os.environ.setdefault("JYC_DUCKDB_MEMORY", C.DUCKDB_MEMORY)


def month_bounds(start, end):
    s = pd.Period(start, "M").start_time
    e = pd.Period(end, "M").end_time.floor("s")
    return s, e


def chunks(start, end, mode):
    """按月 / 季 / 整段切分评估区间（平台一次调用整个测试区间；长区间分块以控制内存）。"""
    if mode == "none":
        return [month_bounds(start, end)]
    freq = {"month": "M", "quarter": "Q"}[mode]
    s, e = month_bounds(start, end)
    out = []
    for p in pd.period_range(s, e, freq=freq):
        a, b = max(s, p.start_time), min(e, p.end_time.floor("s"))
        out.append((a, b))
    return out


def notebook_code(path):
    """取出 notebook 的代码单元；跳过平台自测单元（含 jyc_eval / bigmodule）与 IPython 魔法命令。"""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if re.search(r"jyc_eval|bigmodule", src):
            log("跳过自测单元（含 jyc_eval / bigmodule）")
            continue
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        cells.append(src)
    return cells


def load_main(path):
    path = Path(path)
    ns = {"__name__": "jyc_submission", "__file__": str(path)}
    if path.suffix == ".ipynb":
        for i, src in enumerate(notebook_code(path)):
            exec(compile(src, f"{path.name}[cell {i}]", "exec"), ns)
    else:
        exec(compile(path.read_text(encoding="utf-8"), path.name, "exec"), ns)
    if not callable(ns.get("main")):
        raise ValueError("提交代码里没有找到 main(datasources, start_date, end_date)")
    return ns["main"]


def run_main(main, start, end, chunk_mode, datasources):
    outs = []
    cs = chunks(start, end, chunk_mode)
    fmt = lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
    try:
        import dai
    except ImportError:
        dai = None
    for i, (a, b) in enumerate(cs):
        t0 = time.time()
        sd, ed = fmt(a), fmt(b)
        if dai is not None and hasattr(dai, "set_chunk"):      # 让 DAI 模拟层预取下一块的 SQL 结果
            dai.set_chunk((sd, ed), (fmt(cs[i + 1][0]), fmt(cs[i + 1][1])) if i + 1 < len(cs) else None)
        log(f"调用 main(datasources, {sd!r}, {ed!r})")
        df = main(dict(datasources), sd, ed)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"main 返回的是 {type(df).__name__}，应为 DataFrame")
        outs.append(df)
        log(f"  返回 {len(df):,} 行，{time.time() - t0:.0f}s，内存峰值 {peak_gb():.1f} GB")
    return pd.concat(outs, ignore_index=True)


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def find_e2e(job_dir, inputs):
    """在输入里找到端到端三件套（可以是 zip），解压到 job_dir/e2e/，返回目录。"""
    dst = job_dir / "e2e"
    dst.mkdir(exist_ok=True)
    for f in inputs:
        f = Path(f)
        if f.suffix == ".zip":
            with zipfile.ZipFile(f) as z:
                for n in z.namelist():
                    if n.endswith("/") or "__MACOSX" in n:
                        continue
                    base = os.path.basename(n)
                    if base in ("Transformer_modelsave_train.py", "Transformer_modelsave_predict.ipynb", "transformer_model.json"):
                        with z.open(n) as s, open(dst / base, "wb") as t:
                            shutil.copyfileobj(s, t)
        elif f.name in ("Transformer_modelsave_train.py", "Transformer_modelsave_predict.ipynb", "transformer_model.json"):
            shutil.copy(f, dst / f.name)
    need = ["Transformer_modelsave_train.py", "Transformer_modelsave_predict.ipynb", "transformer_model.json"]
    miss = [n for n in need if not (dst / n).exists()]
    if miss:
        raise FileNotFoundError(f"端到端提交缺少文件：{miss}（三个文件名必须与平台要求一致）")
    return dst


# ---------------------------------------------------------------- 平台规则检查
def platform_checks(raw: pd.DataFrame, start, end):
    """按平台规则检查输出（提交须知：三列、8 个截面、无重复键、无 inf、每个截面缺失率 ≤ 40%）。"""
    c = {"rows": int(len(raw)), "columns": [str(x) for x in raw.columns]}
    c["exactly_three_columns"] = list(raw.columns) == COLS
    if not set(COLS).issubset(raw.columns):
        c["fatal"] = f"缺少列：{sorted(set(COLS) - set(raw.columns))}"
        return c, None
    d = raw[COLS].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    c["date_parse_fail"] = int(d["date"].isna().sum())
    hhmm = d["date"].dt.hour * 100 + d["date"].dt.minute
    d.loc[hhmm == 925, "date"] = d.loc[hhmm == 925, "date"].dt.normalize() + pd.Timedelta("09:30:00")
    hhmm = d["date"].dt.hour * 100 + d["date"].dt.minute
    bad_slot = ~hhmm.isin(SLOTS) | (d["date"].dt.second != 0)
    c["rows_outside_8_slots"] = int(bad_slot.sum())
    c["outside_slot_examples"] = sorted({str(x) for x in d.loc[bad_slot, "date"].dt.strftime("%H:%M:%S").dropna().unique()})[:10]
    d["instrument"] = d["instrument"].astype(str).str.strip()
    num = pd.to_numeric(d["factor"], errors="coerce")
    c["non_numeric_factor"] = int(num.isna().sum() - d["factor"].isna().sum())
    d["factor"] = num.astype("float64")
    c["inf_count"] = int(np.isinf(d["factor"].to_numpy()).sum())
    c["nan_count"] = int(d["factor"].isna().sum())
    c["duplicate_keys"] = int(d.duplicated(["date", "instrument"]).sum())
    s, e = month_bounds(start, end)
    c["rows_outside_range"] = int(((d["date"] < s) | (d["date"] > e)).sum())
    # 每个截面的缺失率：相对平台股票池（cpt_jyc_2026_instruments）
    pool_f = C.PLATFORM_DIR / "instruments.parquet"
    if pool_f.exists():
        pool = pd.read_parquet(pool_f)
        pool = pool[(pool["date"] >= s) & (pool["date"] <= e)]
        ok = d[np.isfinite(d["factor"])][["date", "instrument"]].drop_duplicates()
        got = pool.merge(ok, on=["date", "instrument"], how="inner").groupby("date").size()
        tot = pool.groupby("date").size()
        miss = (1 - got.reindex(tot.index).fillna(0) / tot)
        bys = miss.groupby(miss.index.hour * 100 + miss.index.minute)
        c["missing_rate_by_slot"] = {str(k): {"mean": float(v.mean()), "max": float(v.max())} for k, v in bys}
        c["sections_over_40pct_missing"] = int((miss > 0.4).sum())
        c["sections_total"] = int(len(miss))
        c["worst_sections"] = {str(k): float(v) for k, v in miss.sort_values(ascending=False).head(5).items()}
    c["pass"] = bool(c["exactly_three_columns"] and c["date_parse_fail"] == 0 and c["rows_outside_8_slots"] == 0
                     and c["inf_count"] == 0 and c["duplicate_keys"] == 0 and c.get("sections_over_40pct_missing", 0) == 0)
    return c, d


def to_core_keys(d: pd.DataFrame):
    """date/instrument（真实代码或 ID0006 伪代码）→ (截面时间 ns, 伪代码数字)。返回映射统计。"""
    m = pd.read_csv(C.DATA / "exposure" / "id_map.csv", dtype=str)
    m = m[m["ok"].str.lower() == "true"]
    real2id = dict(zip(m["instrument"], m["local_id"]))
    inst = d["instrument"]
    is_pseudo = inst.str.match(r"^ID\d{4}$")
    pseudo = inst.where(is_pseudo, inst.map(real2id))
    stat = {"pseudo_code_rows": int(is_pseudo.sum()), "real_code_rows": int((~is_pseudo).sum()),
            "unmapped_rows": int(pseudo.isna().sum()),
            "unmapped_examples": sorted(inst[pseudo.isna()].unique().tolist())[:10]}
    ok = pseudo.notna()
    out = pd.DataFrame({"key_date": d.loc[ok, "date"].to_numpy("datetime64[ns]"),
                        "instrument_id": pseudo[ok].str[2:].astype(np.int32).to_numpy(),
                        "factor": d.loc[ok, "factor"].to_numpy()})
    return out, stat


# ---------------------------------------------------------------- 主流程
def produce(job_dir: Path, spec: dict):
    kind, start, end = spec["kind"], spec["start"], spec["end"]
    inputs = [job_dir / "input" / x for x in spec.get("inputs", [])]
    warnings = []
    if kind == "values":
        f = inputs[0]
        raw = pd.read_parquet(f) if f.suffix in (".parquet", ".pq") else pd.read_csv(f)
    elif kind in ("platform", "e2e", "sql"):
        platform_env()
        import dai
        datasources = {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments"}
        if kind == "platform":
            main = load_main(inputs[0])
            raw = run_main(main, start, end, spec.get("chunk") or "quarter", datasources)
        elif kind == "e2e":
            d = find_e2e(job_dir, inputs)
            os.chdir(d)
            sys.path.insert(0, str(d))
            try:
                import torch
                torch.set_num_threads(int(spec.get("torch_threads", C.ENGINE_THREADS)))
                log(f"torch {torch.__version__}，CPU 线程 {torch.get_num_threads()}，CUDA 可用 {torch.cuda.is_available()}")
            except ImportError:
                raise RuntimeError("端到端推理需要 torch，远端未安装")
            main = load_main(d / "Transformer_modelsave_predict.ipynb")
            raw = run_main(main, start, end, spec.get("chunk") or "month", datasources)
        else:
            s, e = month_bounds(start, end)
            sql = inputs[0].read_text(encoding="utf-8")
            raw = dai.query(sql, filters={"date": [str(s), str(e)]}).df()
        warnings += sorted(dai.WARNINGS)
        (job_dir / "dai_queries.json").write_text(json.dumps(dai.LOG, ensure_ascii=False, indent=1), encoding="utf-8")
    elif kind in ("factor_py", "fid"):
        import panel_store
        panel_store.configure_localeval()
        if kind == "fid":
            import factor_lib
            hit = [x for x in factor_lib.FACTORS if str(x[0]) == str(spec["fid"])]
            if not hit:
                raise ValueError(f"factor_lib.FACTORS 中没有编号 {spec['fid']}")
            fn = hit[0][3]
        else:
            import importlib.util
            sp = importlib.util.spec_from_file_location("jyc_user_factor", inputs[0])
            mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(mod)
            if not callable(getattr(mod, "factor", None)):
                raise ValueError("研究脚本必须定义 factor(panel) -> 与 panel 等长的数组")
            fn = mod.factor
        keys = panel_store.factor_on_panel(fn, start, end, log=log)
        keys.to_parquet(job_dir / "factor_keys.parquet", index=False)
        c = {"rows": int(len(keys)), "pass": True, "note": "面板模式：因子按面板行计算，平台输出格式检查不适用"}
        return c, {"pseudo_code_rows": int(len(keys)), "real_code_rows": 0, "unmapped_rows": 0}, warnings
    else:
        raise ValueError(f"未知的提交类型 {kind}")

    raw.to_parquet(job_dir / "factor_raw.parquet", index=False) if len(raw) else None
    checks, d = platform_checks(raw, start, end)
    if d is None:
        raise ValueError(checks["fatal"])
    keys, mstat = to_core_keys(d.dropna(subset=["date"]).drop_duplicates(["date", "instrument"]))
    keys.to_parquet(job_dir / "factor_keys.parquet", index=False)
    return checks, mstat, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    job_dir = Path(a.job).resolve()
    spec = json.loads((job_dir / "spec.json").read_text(encoding="utf-8"))
    t0 = time.time()
    try:
        checks, mstat, warnings = produce(job_dir, spec)
        out = {"ok": True, "checks": checks, "mapping": mstat, "warnings": warnings,
               "produce_seconds": round(time.time() - t0, 1), "produce_peak_gb": round(peak_gb(), 2)}
    except Exception as e:
        traceback.print_exc()
        out = {"ok": False, "error": f"{type(e).__name__}: {e}", "produce_seconds": round(time.time() - t0, 1)}
    (job_dir / "produce.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log("完成" if out["ok"] else f"失败：{out['error']}")
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
