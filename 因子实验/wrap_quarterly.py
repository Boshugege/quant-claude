# -*- coding: utf-8 -*-
"""把任意因子文件包成"按自然季度循环调用原 main"的版本（控制平台整段调用的内存）。
原 main 原样改名 _main_q；新 main 按季度切分 [start, end]，逐季调用、拼接，季度之间 gc。
评测机的 platform 任务本来就按季度调用 main，所以包装后的输出与评测机已有分数逐位一致。
用法：python3 因子实验/wrap_quarterly.py <源文件> <编号> <标题>"""
import io, os, re, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
src, num, title = sys.argv[1], sys.argv[2], sys.argv[3]
s = io.open(src, encoding="utf-8").read()
body, tail = s.split('\nif __name__ == "__main__":')
body, k = re.subn(r"^def main\(datasources, start_date, end_date\):", "def _main_q(datasources, start_date, end_date):", body, count=1, flags=re.M)
assert k == 1
head = f'''def main(datasources, start_date, end_date):
    """【{num}：{title}】= {os.path.basename(src)} 按自然季度循环调用（原 main 原样保留为 _main_q）。
    平台整段调用时，每次只处理一个季度（另加原代码自带的回看），内存与评测机按季度评分时相同；输出与评测机已有分数逐位一致。
    """
    import gc
    import pandas as pd
    parts = []
    for _q in pd.period_range(pd.Timestamp(start_date).to_period("Q"), pd.Timestamp(end_date).to_period("Q"), freq="Q"):
        _s = max(pd.Timestamp(start_date), _q.start_time)
        _e = min(pd.Timestamp(end_date), _q.end_time)
        if _s > _e:
            continue
        parts.append(_main_q(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S")))
        gc.collect()
    out = pd.concat(parts, ignore_index=True)
    return out[["date", "instrument", "factor"]].drop_duplicates(["date", "instrument"]).sort_values(["date", "instrument"]).reset_index(drop=True)


'''
out = os.path.join(D, f"{num}_{title}.py")
io.open(out, "w", encoding="utf-8").write(head + body.rstrip() + "\n" + '\n\nif __name__ == "__main__":' + tail)
print("wrote", out)
