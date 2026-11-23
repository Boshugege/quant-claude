# -*- coding: utf-8 -*-
"""多头部 / 多根合成的代码版：A 原始值 + Σ w_i·(rank(B_i) − 0.5)。
与评测机 values 合成（build_values_composite.py 的 raw 模式，逐步叠加）逐值同口径：
  每个 B_i 在自身全部行上逐截面百分位排名；A 有、B_i 没有的行该项记 0；09:30 置 0；缺失记 0。
A 先算完只留结果；每个 B_i 按自然季度循环计算、算完只留排名，控制平台整段调用时的内存。
每个 B_i 文件里的顶层函数都加后缀 _b<i>，避免与 A 及其他 B 重名（如多个 _root_impl）。
用法：python3 因子实验/build_multi_head.py [--aq] <编号> <标题> <A 文件> <B1 文件>=<w1>[@<EWMA α>][:说明] ...
  --aq：A 也按自然季度循环计算再拼接（A 自身是整段一次性算、内存峰值高时用，如队友 161 头部；A 只用过去数据时结果不变）。
  w 后面可跟 @α：先对 B 在同一股票的连续截面上做 EWMA（新值权重 1 − α，与 build_values_composite.py --ewma α 相同），再排名。
  再往后可跟 ~HHMM+额外权重（可多个）：该截面时刻的权重 = w + 额外权重，如 0.05~1430+0.10（14:30 截面 0.15，其余 0.05）。
"""
import io, os, re, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
AQ = "--aq" in sys.argv
BOFF = int(os.environ.get("BOFF", "0"))                 # B 的编号偏移：A 自身已是多根合成（含 main_b1…）时用，避免重名
argv = [a for a in sys.argv if a != "--aq"]
num, title, fa = argv[1], argv[2], argv[3]
specs = []
for a in argv[4:]:
    path, rest = a.split("=", 1)
    w, _, desc = rest.partition(":")
    w, *slot_ex = w.split("~")
    w, _, ew = w.partition("@")
    ex = {a.split("+")[0]: float(a.split("+")[1]) for a in slot_ex}
    specs.append((path, float(w), float(ew) if ew else 0.0, desc or os.path.basename(path), ex))


def load(path):
    s = io.open(path, encoding="utf-8").read()
    return s.split('\nif __name__ == "__main__":')


def rename_defs(src, suffix):
    names = re.findall(r"^def (\w+)\(", src, flags=re.M)
    for n in sorted(set(names), key=len, reverse=True):
        src = re.sub(r"\b%s\(" % re.escape(n), "%s%s(" % (n, suffix), src)
    return src


pa = load(fa)
pa[0] = re.sub(r"\b_main_a\(", "_main_a0(", pa[0])                     # A 自身是 build_multi_head 的产物时，它内部的 _main_a 先改名
ba = re.sub(r"^def main\(datasources, start_date, end_date\):", "def _main_a(datasources, start_date, end_date):",
            pa[0].rstrip() + "\n", count=1, flags=re.M)
tail = pa[1] if len(pa) > 1 else None
bodies, calls, docs = [], [], []
for i, (path, w, ew, desc, ex) in enumerate(specs, 1 + BOFF):
    b = rename_defs(load(path)[0].rstrip() + "\n", f"_b{i}")
    assert re.search(r"^def main_b%d\(datasources, start_date, end_date\):" % i, b, flags=re.M), path
    bodies.append(b)
    calls.append(f"        (main_b{i}, {w}, {ew}, {ex!r}),")
    docs.append(f"      B{i} × {w:g}{'（先做 EWMA α=%g）' % ew if ew else ''}{''.join('，%s:%s 截面再加 %g' % (k[:2], k[2:], x) for k, x in ex.items())}：{desc}")
names_a = set(re.findall(r"^def (\w+)\(", ba, flags=re.M))
for b in bodies:
    assert not (names_a & set(re.findall(r"^def (\w+)\(", b, flags=re.M)))
doc_txt = "\n".join(docs)
if AQ:
    a_call = """    _pa = []
    for _q in pd.period_range(pd.Timestamp(start_date).to_period("Q"), pd.Timestamp(end_date).to_period("Q"), freq="Q"):
        _s = max(pd.Timestamp(start_date), _q.start_time)
        _e = min(pd.Timestamp(end_date), _q.end_time)
        if _s > _e:
            continue
        _pa.append(_main_a(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S"))[["date", "instrument", "factor"]])
        gc.collect()
    a = pd.concat(_pa, ignore_index=True).drop_duplicates(["date", "instrument"])            # A 也按季度算（--aq），控制内存
    del _pa"""
else:
    a_call = """    a = _main_a(datasources, start_date, end_date)[["date", "instrument", "factor"]].copy()"""
call_txt = "\n".join(calls)
main = f'''def main(datasources, start_date, end_date):
    """【{num}：{title}】多头部合成：A = {os.path.basename(fa)}（原始值）＋ Σ w × (B 的截面百分位排名 − 0.5)。
{doc_txt}
    口径与评测机 values 合成（build_values_composite.py 的 raw 模式逐步叠加）一致：每个 B 在自身全部行上逐截面排名，
    A 有、B 没有的行该项记 0，09:30 置 0，缺失记 0。A 先算完只留结果；每个 B 按自然季度循环计算，控制内存。
    """
    import gc
    import numpy as np
    import pandas as pd

{a_call}
    a["date"] = pd.to_datetime(a["date"])
    a["instrument"] = a["instrument"].astype(str)
    v = pd.to_numeric(a["factor"], errors="coerce").to_numpy(float)
    key = a[["date", "instrument"]]
    _hm = a["date"].dt.strftime("%H%M").to_numpy()
    gc.collect()
    for _fn, _w, _ew, _ex in [
{call_txt}
    ]:
        parts = []
        for _q in pd.period_range(pd.Timestamp(start_date).to_period("Q"), pd.Timestamp(end_date).to_period("Q"), freq="Q"):
            _s = max(pd.Timestamp(start_date), _q.start_time)
            _e = min(pd.Timestamp(end_date), _q.end_time)
            if _s > _e:
                continue
            parts.append(_fn(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S"))[["date", "instrument", "factor"]])
            gc.collect()
        b = pd.concat(parts, ignore_index=True)
        del parts
        b["date"] = pd.to_datetime(b["date"])
        b["instrument"] = b["instrument"].astype(str)
        b = b.drop_duplicates(["date", "instrument"])
        b["factor"] = pd.to_numeric(b["factor"], errors="coerce")
        if _ew > 0:                                                       # 同一股票按时间连续的 EWMA（跨日），只用过去，与 values 合成的 --ewma 相同
            b = b.sort_values(["instrument", "date"]).reset_index(drop=True)
            b["factor"] = b.groupby("instrument", sort=False)["factor"].transform(lambda x: x.ewm(alpha=1 - _ew, adjust=False, ignore_na=True).mean())
        b["rx"] = b.groupby("date")["factor"].rank(pct=True) - 0.5
        rx = key.merge(b[["date", "instrument", "rx"]], on=["date", "instrument"], how="left")["rx"].fillna(0.0).to_numpy(float)
        del b
        v = v + _w * rx
        for _s, _x in _ex.items():                                        # 分时段的额外权重
            v = v + _x * rx * (_hm == _s)
        gc.collect()
    v[a["date"].dt.strftime("%H%M").eq("0930").to_numpy()] = 0.0
    out = pd.DataFrame({{"date": a["date"].to_numpy(), "instrument": a["instrument"].to_numpy(), "factor": np.nan_to_num(v, nan=0.0)}})
    return out.sort_values(["date", "instrument"]).reset_index(drop=True)


'''
out = os.path.join(D, f"{num}_{title}.py")
src = main + ba + "\n\n" + "\n\n".join(bodies)
if tail is not None:
    src += '\n\nif __name__ == "__main__":' + tail
io.open(out, "w", encoding="utf-8").write(src)
print("wrote", out)
