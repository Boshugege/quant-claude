import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "工具/本地评分")
import localeval as le
from factor_lib import FACTORS, add_components
facs = {f[0]: f for f in FACTORS}
for y in ("2023", "2024"):
    p = add_components(le.build_panel(le.load_bars(f"{int(y)-1}-07", f"{y}-12")))
    hh = (p["slot"] // 100).astype(int).astype(str).str.zfill(2) + ":" + (p["slot"] % 100).astype(int).astype(str).str.zfill(2)
    p["key_date"] = p["day"] + pd.to_timedelta(hh + ":00")
    p["instrument"] = "ID" + p["instrument_id"].astype(int).astype(str).str.zfill(4)
    keep = p["day"] >= pd.Timestamp(f"{y}-01-01")
    for fid in ("90", "104"):
        f = facs[fid][3](p)
        pd.DataFrame({"date": p.loc[keep, "key_date"], "instrument": p.loc[keep, "instrument"], "factor": f[keep]}).to_parquet(f"e2e/runs/analysis/F{fid}_{y}.parquet")
    print(y, "done")
