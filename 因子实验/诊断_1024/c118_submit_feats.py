# c118（10-24）：把 c115 的 14:30 候选（只有 14:30 有值，其它截面填 0）作为 values 任务交评分机，供 c105i 的 ADD=任务@1430:权重 读取。
import subprocess, numpy as np, pandas as pd
d = pd.read_parquet("/srv/media/jyc_scorer/incoming/e2eslot/c115_feats.parquet")
keys = pd.read_parquet("/srv/media/jyc_scorer/incoming/e2eslot/e2e_R22a_2023.parquet", columns=["date", "instrument"])
keys = pd.concat([keys, pd.read_parquet("/srv/media/jyc_scorer/incoming/e2eslot/e2e_R22a_2024.parquet", columns=["date", "instrument"])])
_m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", dtype=str); _m = _m[_m["ok"].str.lower() == "true"]
I2R = dict(zip(_m["local_id"], _m["instrument"]))          # 底座任务用真实代码（映射不上的才用 ID 伪代码），附加项必须同一口径
for f in ["y_20", "cvx_20", "r_day", "drift_20", "hl_pos", "y_5", "auc_20"]:
    v = keys.merge(d[["date", "instrument", "f_" + f]], on=["date", "instrument"], how="left").rename(columns={"f_" + f: "factor"})
    v["factor"] = v["factor"].fillna(0.0)
    v["instrument"] = v["instrument"].map(lambda s: I2R.get(s, s))
    out = f"/srv/media/jyc_scorer/incoming/e2eslot/c115_{f}.parquet"
    v.to_parquet(out, index=False)
    r = subprocess.run(["curl", "-s", "-F", f"files=@{out}", "-F", "kind=values", "-F", "start=2023-01", "-F", "end=2024-12",
                        "-F", f"name=c115_{f}", "http://127.0.0.1:46220/api/jobs"], capture_output=True, text=True)
    print(f, r.stdout[:120], flush=True)
