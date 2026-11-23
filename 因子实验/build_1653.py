# -*- coding: utf-8 -*-
"""1617 + 一档委托规模比 + 附加项 + 最小抬升缓冲带（10-22）。
在 build_1649 的输出上：把附加项的平台脚本（s1_sqlgen 生成、评分机五年任务跑过的同一份代码）改名为 _addon_<名字> 嵌进来，
_post_turnover 里在规模比之后、缓冲带之前逐个加：z ← z(z + w·z(附加项))，顺序、口径与评分机 c105h 的 ADD 相同。
用法：python3 因子实验/build_1653.py <编号> <名字> <buf> <m> <附加项1:权重,附加项2:权重,…> [<q_w>]"""
import importlib.util
import os
import subprocess
import sys

num, title, buf, lm, adds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
qw = sys.argv[6] if len(sys.argv) > 6 else "0.3"
subprocess.run([sys.executable, "因子实验/build_1649.py", num, title, buf, lm, qw], check=True)
DST = "因子代码/%s_%s.py" % (num, title)
s = open(DST, encoding="utf-8").read()

spec = importlib.util.spec_from_file_location("sqlgen", "因子实验/s1_sqlgen.py")
G = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = ["s1_sqlgen"]; spec.loader.exec_module(G); sys.argv = _argv
AD = []                                                                         # (任务名, 权重, 只生效的时段 HHMM 或 None)；任务@1430:w 只在 14:30 生效（10-24）
for kv in adds.split(","):
    if not kv:
        continue
    job, w = kv.rsplit(":", 1)
    sl = None
    if "@" in job:
        job, sl = job.split("@")
    AD.append((job, float(w), sl))
funcs = []
for full, w, _sl in sorted({(f, 0.0, None) for f, _, _ in AD}):
    batch, name = full.split("_", 1)
    doc, formula, smooth, lookback = G.B[batch][name][:4]
    fam = G.B[batch][name][4] if len(G.B[batch][name]) > 4 else "base"
    src = G.TEMPLATE.format(doc=doc, batch=batch, smooth_doc=("，按股票跨截面指数平滑 α=%.1f（过去的权重）" % smooth) if smooth else "",
                            lookback=lookback, sql=G.SQLS[fam], formula=formula, smooth=smooth)
    assert src.count("def main(datasources, start_date, end_date):") == 1
    funcs.append(src.replace("def main(datasources, start_date, end_date):", "def _addon_%s(datasources, start_date, end_date):" % full))


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)


rep("""        d = d.drop(columns=["q"])
""", """        d = d.drop(columns=["q"])
    for _fn, _w, _sl in [%s]:                                                  # 附加项（评分机五年任务的同一份代码）；_sl 非空时只在该时段生效
        _a = _fn(datasources, start_date, end_date)[["date", "instrument", "factor"]].rename(columns={"factor": "_a"})
        _a["date"] = pd.to_datetime(_a["date"])
        _a["instrument"] = _a["instrument"].astype(str)
        _a = _a.drop_duplicates(["date", "instrument"], keep="last")
        d = d.merge(_a, on=["date", "instrument"], how="left")
        _za = _z(d["_a"].astype(float).fillna(0.0))
        if _sl is not None:
            _za[(d["date"].dt.strftime("%%H%%M") != _sl).to_numpy()] = 0.0
        d["z"] = _z(d["z"] + _w * _za)
        d = d.drop(columns=["_a"])
""" % ", ".join("(_addon_%s, %r, %r)" % (f, w, sl) for f, w, sl in AD))
k = s.index('\nif __name__ == "__main__":')
s = s[:k] + "\n\n# ===== 附加项：评分机五年任务的平台脚本原样嵌入（main 改名）=====\n" + "\n\n".join(funcs) + "\n" + s[k:]
rep("腿边界缓冲带（最小抬升", "加附加项（%s，各自逐截面 z 分数）；腿边界缓冲带（最小抬升" % "、".join("%s%s×%.2f" % (f, "@" + sl if sl else "", w) for f, w, sl in AD))
open(DST, "w", encoding="utf-8").write(s)
print("加附加项", DST, AD)
