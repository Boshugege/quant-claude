# -*- coding: utf-8 -*-
"""生成"按月分批、方向正确"的新根单独版（公榜探针用，控制平台整段调用的内存）。
根 = 多统计量框架原样复用（只把 STAT 变成参数）；main 按自然月分批调用，拼接后乘以方向 SIGN（−1 = 反向交付）。
用法：python3 因子实验/build_standalone_monthly.py <编号> <标题> <框架文件名> <STAT> <SIGN: 1 或 -1> [--lean]
--lean：分钟 SQL 只保留该 STAT 用到的列（与 build_multiroot.py --lean 同一套裁剪，算法逐字不变）
例：python3 因子实验/build_standalone_monthly.py 1352 大单方向反向按月 1260_量价同向.py N_dir -1"""
import io, os, re, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
num, title, fw, stat, sign = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
LEAN = "--lean" in sys.argv[6:]
assert sign in (1, -1)
src = io.open(os.path.join(D, fw), encoding="utf-8").read()
body, tail = src.split('\nif __name__ == "__main__":')
body = re.sub(r"^def main\(datasources, start_date, end_date\):", "def _root_impl(datasources, start_date, end_date, STAT):", body.rstrip() + "\n", count=1, flags=re.M)
body, k = re.subn(r'^    STAT = "[^"]*"\n', "", body, count=1, flags=re.M)
assert k == 1, "框架里找不到 STAT 行"
if LEAN:
    sys.argv = [sys.argv[0]]                                                    # 只借用 build_multiroot 里的裁剪函数
    import importlib.util
    _sp = importlib.util.spec_from_file_location("_bm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_lean_sql.py"))
    _bm = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_bm)
    body = _bm.lean_root(body, [stat])
flip = '    out["factor"] = -out["factor"]                                           # 反向交付（09:30 仍为 0）\n' if sign < 0 else ""
main = f'''def main(datasources, start_date, end_date):
    """【{num}：{title}】新根单独版（公榜探针），STAT = {stat}，方向 {"反向" if sign < 0 else "正向"}。
    根 = {fw} 原样复用；按自然月分批调用，控制平台整段调用时的内存。
    """
    import pandas as pd
    parts = []
    for _ms in pd.period_range(pd.Timestamp(start_date).to_period("M"), pd.Timestamp(end_date).to_period("M"), freq="M"):
        _s = max(pd.Timestamp(start_date), _ms.start_time)
        _e = min(pd.Timestamp(end_date), _ms.end_time)
        if _s > _e:
            continue
        parts.append(_root_impl(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S"), "{stat}"))
    out = pd.concat(parts, ignore_index=True)
{flip}    return out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


'''
out = os.path.join(D, f"{num}_{title}.py")
io.open(out, "w", encoding="utf-8").write(main + body + '\n\nif __name__ == "__main__":' + tail)
print("wrote", out)
