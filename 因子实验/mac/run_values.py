# -*- coding: utf-8 -*-
"""批量算因子值：python run_values.py <因子文件...>；按季度、2023-01～2024-12，输出 out/<编号>.parquet（date, instrument, factor）。"""
import gc, importlib.util, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dai  # noqa: E402  替身
import pandas as pd  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
files = sys.argv[1:]
mods = []
for f in files:
    spec = importlib.util.spec_from_file_location("m" + os.path.basename(f)[:4], f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    mods.append((os.path.basename(f)[:4], m))
ds = {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments"}
res = {n: [] for n, _ in mods}
for q in pd.period_range("2023Q1", "2024Q4", freq="Q"):
    sd, ed = q.start_time, q.end_time.normalize()
    t0 = time.time()
    dai.load_span(sd - pd.Timedelta(days=135), ed + pd.Timedelta(hours=23, minutes=59, seconds=59))
    for n, m in mods:
        out = m.main(ds, sd.strftime("%Y-%m-%d 00:00:00"), ed.strftime("%Y-%m-%d 23:59:59"))
        res[n].append(out)
    print(f"{q} done {time.time() - t0:.0f}s rows={dai._TBL['tbl'].num_rows:,}", flush=True)
    dai._TBL["tbl"] = None
    gc.collect()
for n, parts in res.items():
    pd.concat(parts, ignore_index=True).to_parquet(os.path.join(OUT, f"{n}.parquet"), index=False)
print("saved", list(res))
