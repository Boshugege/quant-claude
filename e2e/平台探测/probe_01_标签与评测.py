# -*- coding: utf-8 -*-
# 平台探测 01（2026-10-12）：在平台 notebook 里按 "# %%" 分单元依次运行，每个单元互相独立，可以单独重跑。
# 产出文件都写在当前工作目录，用于核对本地标签和评测口径，不随仓库发布：
#   probe_label_*.parquet          官方标签表（确定 14:30 截面标签的定义，核对本地标签）
#   probe_jyc_eval_source.txt      评测函数源码（如果平台允许查看）
#   probe_fac_<编号>.parquet        因子在平台上的取值（核对平台实现与本地是否一致）
#   probe_eval_results.txt         各因子自测的五项原始指标与日志
# 只读查询与自测，不提交任何东西，不影响榜单。

# %% 单元 1：官方标签表 cpt_jyc_2026_vwap（结构 + 2024 年 1–3 月 + 2021 年 6 月）
import dai
import pandas as pd
import numpy as np

for tag, (s, e) in {"2024Q1": ("2024-01-01 00:00:00", "2024-03-31 23:59:59"),
                    "202106": ("2021-06-01 00:00:00", "2021-06-30 23:59:59")}.items():
    t = dai.query("SELECT * FROM cpt_jyc_2026_vwap", filters={"date": [s, e]}).df()
    print(f"== {tag} 行数 {len(t):,}")
    print(t.dtypes)
    print(t.head(10).to_string())
    hm = pd.to_datetime(t["date"]).dt.strftime("%H:%M")
    print("各时点行数：", hm.value_counts().sort_index().to_dict())
    print("各时点标签缺失率：", t.drop(columns=["date", "instrument"]).isna().groupby(hm).mean().round(4).to_dict())
    t.to_parquet(f"probe_label_{tag}.parquet")

# %% 单元 2：评测函数源码（看不到也没关系，会打印原因）
import inspect
from bigmodule import M

out = []
ev = M.jyc_eval
out.append(f"type: {type(ev)}")
out.append("dir: " + ", ".join(n for n in dir(ev) if not n.startswith("__")))
for name in ["_latest", "run", "v6", "_run"]:
    obj = getattr(ev, name, None)
    if obj is None:
        continue
    try:
        out.append(f"\n==== {name}  file={inspect.getfile(obj)}\n" + inspect.getsource(obj))
    except Exception as ex:
        out.append(f"\n==== {name}: 无法取得源码 {ex!r}")
try:
    import sys
    mod = sys.modules.get(type(ev).__module__) or inspect.getmodule(ev._latest)
    out.append(f"\n==== 模块 {mod}\n" + inspect.getsource(mod))
except Exception as ex:
    out.append(f"\n==== 模块源码不可用 {ex!r}")
open("probe_jyc_eval_source.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out)[:3000])

# %% 单元 3：同一区间自测 5 个因子（需要先把 因子代码/ 下这 5 个 .py 上传到当前目录）
#   90_多尺度反转阶梯.py  98_四尺度阶梯加四特征.py  100_四尺度阶梯加特质波动.py
#   102_纯股票特征合成.py  104_同伴残差阶梯过度中性.py
# 目的：本地认为 98/100 好于 90，公榜相反。看平台评测在同一区间给出的原始指标是否也相反。
import contextlib
import glob
import importlib.util
import io
import json
import time
from bigmodule import M

START, END = "2024-01-02 00:00:00", "2024-03-01 23:59:59"   # 与 E01 平台自测同一区间
DS = {"bar1m": "cpt_jyc_2026_stock_bar1m"}


def load_main(path):
    spec = importlib.util.spec_from_file_location("fac_" + path.split("_")[0], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


log = open("probe_eval_results.txt", "a", encoding="utf-8")
for fid in ["90", "98", "100", "102", "104"]:
    paths = sorted(glob.glob(f"{fid}_*.py"))
    if not paths:
        print(f"{fid}: 没找到文件，跳过")
        continue
    t0 = time.time()
    df = load_main(paths[0])(DS, START, END)
    df.to_parquet(f"probe_fac_{fid}.parquet")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        res = M.jyc_eval._latest(factor_data=df, show=False)
    try:
        rtxt = json.dumps(res, ensure_ascii=False, default=str, indent=1)
    except Exception:
        rtxt = repr(res)
    msg = f"\n######## {fid} {paths[0]} rows={len(df):,} 用时 {time.time() - t0:.0f}s\n结果：{rtxt[:20000]}\n日志：\n{buf.getvalue()[-20000:]}"
    log.write(msg)
    log.flush()
    print(msg[:1500])
log.close()
