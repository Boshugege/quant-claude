# -*- coding: utf-8 -*-
"""两个头部的合成代码版：A 原始值 + w·(rank(B) − 0.5)，与 build_values_composite.py 的 raw 模式逐值一致（H = A、X = B）。
A 先算完（只留结果），再按自然季度循环算 B（控制平台整段调用的内存）。
用法：python3 因子实验/build_two_head.py <编号> <标题> <A 文件> <B 文件> <w> [B 的说明]
例：python3 因子实验/build_two_head.py 1392 合成1363加156排名零点五 因子代码/1363_….py /tmp/s156.py 0.5 "sushirui 156（盘口锚点系列）"
"""
import io, os, re, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
num, title, fa, fb, w = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
bdesc = sys.argv[6] if len(sys.argv) > 6 else os.path.basename(fb)


def body(path, newname):
    s = io.open(path, encoding="utf-8").read()
    parts = s.split('\nif __name__ == "__main__":')
    b, k = re.subn(r"^def main\(datasources, start_date, end_date\):", f"def {newname}(datasources, start_date, end_date):",
                   parts[0].rstrip() + "\n", count=1, flags=re.M)
    assert k == 1, path
    return b, (parts[1] if len(parts) > 1 else None)


ba, tail = body(fa, "_main_a")
bb, _ = body(fb, "_main_b")
names_a = set(re.findall(r"^def (\w+)\(", ba, flags=re.M))
names_b = set(re.findall(r"^def (\w+)\(", bb, flags=re.M))
assert not (names_a & names_b), names_a & names_b
main = f'''def main(datasources, start_date, end_date):
    """【{num}：{title}】两头部合成：A = {os.path.basename(fa)}（原始值）＋ {w:g} × B（截面百分位排名 − 0.5）。
    B = {bdesc}。
    口径与评测机 values 合成（build_values_composite.py 的 raw 模式）一致：
      B 先在自身全部行上逐截面排名；A 有、B 没有的行 B 项记 0；09:30 置 0；缺失记 0。
    A 先算完只留结果；B 按自然季度循环计算，控制平台整段调用时的内存。
    """
    import gc
    import numpy as np
    import pandas as pd

    W = {w}
    a = _main_a(datasources, start_date, end_date)[["date", "instrument", "factor"]].copy()
    a["date"] = pd.to_datetime(a["date"])
    a["instrument"] = a["instrument"].astype(str)
    gc.collect()
    parts = []
    for _q in pd.period_range(pd.Timestamp(start_date).to_period("Q"), pd.Timestamp(end_date).to_period("Q"), freq="Q"):
        _s = max(pd.Timestamp(start_date), _q.start_time)
        _e = min(pd.Timestamp(end_date), _q.end_time)
        if _s > _e:
            continue
        parts.append(_main_b(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S"))[["date", "instrument", "factor"]])
        gc.collect()
    b = pd.concat(parts, ignore_index=True)
    del parts
    b["date"] = pd.to_datetime(b["date"])
    b["instrument"] = b["instrument"].astype(str)
    b = b.drop_duplicates(["date", "instrument"])
    b["factor"] = pd.to_numeric(b["factor"], errors="coerce")
    b["rx"] = b.groupby("date")["factor"].rank(pct=True) - 0.5
    m = a.merge(b[["date", "instrument", "rx"]], on=["date", "instrument"], how="left")
    del a, b
    m["rx"] = m["rx"].fillna(0.0)
    v = pd.to_numeric(m["factor"], errors="coerce").to_numpy(float) + W * m["rx"].to_numpy(float)
    v[m["date"].dt.strftime("%H%M").eq("0930").to_numpy()] = 0.0
    out = pd.DataFrame({{"date": m["date"].to_numpy(), "instrument": m["instrument"].to_numpy(), "factor": np.nan_to_num(v, nan=0.0)}})
    return out.sort_values(["date", "instrument"]).reset_index(drop=True)


'''
out = os.path.join(D, f"{num}_{title}.py")
src = main + ba + "\n\n" + bb
if tail is not None:
    src += '\n\nif __name__ == "__main__":' + tail
io.open(out, "w", encoding="utf-8").write(src)
print("wrote", out)
