# -*- coding: utf-8 -*-
"""任务 Q：逐年稳定性表（2020–2024）。

对每个根，取
  2020        -> o20_<编号> 的 result.json["by_year"]["2020"]
  2021 / 2022 -> o21_<编号> 的 by_year
  2023 / 2024 -> fm_<编号>  的 by_year
列：编号、五年 IC、五年 IC 线性趋势（IC/年）、2024 IC / 五年均值、IC 符号是否五年一致。
任务目录查找逻辑照抄 read_vs_base.py（jobs + jobs_archive，spec.json 匹配 name，取有 result.json 的最新目录）。
用法：agent-run $PY -u 因子实验/stability_table.py
"""
import glob, json, os

J = ["/home/user/jyc_scorer/jobs", "/srv/media/jyc_scorer/jobs_archive"]

ROOTS = {
    "新根（1291／1292 里）": [1232, 1236, 1237, 1260, 1263, 1323, 1273, 1334, 1342],
    "公榜未兑现对照": [1133, 1131],
}
YEARS = [2020, 2021, 2022, 2023, 2024]
PREFIX = {2020: "o20_", 2021: "o21_", 2022: "o21_", 2023: "fm_", 2024: "fm_"}


def find(spec):
    """返回 result.json 所在目录；找不到返回 None。"""
    hits = []
    for b in J:
        for sp in glob.glob(os.path.join(b, "*", "spec.json")):
            try:
                if json.load(open(sp)).get("name") == spec:
                    hits.append(os.path.dirname(sp))
            except Exception:
                pass
    with_res = [d for d in hits if os.path.exists(os.path.join(d, "result.json"))]
    if with_res:
        return sorted(with_res, key=os.path.basename)[-1]
    return None


def yearly_ic(num):
    """返回 {年: IC 或 None} 以及缺哪年。"""
    out, missing = {}, []
    for y in YEARS:
        d = find(PREFIX[y] + str(num))
        if d is None:
            out[y] = None
            missing.append(f"{PREFIX[y]}{num}")
            continue
        by = json.load(open(os.path.join(d, "result.json"))).get("by_year", {})
        cell = by.get(str(y))
        out[y] = None if cell is None else cell.get("ic_mean")
        if out[y] is None:
            missing.append(f"{PREFIX[y]}{num}")
    return out, missing


def slope_ols(ys, xs):
    """最小二乘斜率（IC/年）。"""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def fmt(v, w=7, p=4):
    return "  " + "-" * w if v is None else f"{v:+.{p}f}"


print("任务 Q：逐年稳定性表（IC_mean，2020–2024）")
print("数据：2020=o20_ / 2021,2022=o21_ / 2023,2024=fm_；趋势=五年 IC 对年份的最小二乘斜率（IC/年）")
print()

hdr = f"{'编号':>6} {'组别':<18} " + " ".join(f"{y:>8}" for y in YEARS) + f" {'趋势/年':>9} {'2024/均值':>10} {'符号一致':>8}"
print(hdr)
print("-" * len(hdr))

rows = []
for grp, nums in ROOTS.items():
    for n in nums:
        ic, missing = yearly_ic(n)
        xs = [y for y in YEARS if ic[y] is not None]
        ys = [ic[y] for y in xs]
        slope = slope_ols(ys, xs)
        mean = sum(ys) / len(ys) if ys else None
        ratio = (ic[2024] / mean) if (ic[2024] is not None and mean) else None
        signs = set(1 if v > 0 else (-1 if v < 0 else 0) for v in ys)
        if not xs:
            consistent = "—"
        elif len(xs) < 5:
            consistent = "—(缺年)"
        else:
            consistent = "是" if len(signs) == 1 else "否"
        rows.append((grp, n, ic, slope, ratio, consistent, missing))
        cells = " ".join(f"{fmt(ic[y], 8):>8}" for y in YEARS)
        sl = "    —    " if slope is None else f"{slope:+.5f}"
        rt = "    —     " if ratio is None else f"{ratio:+.2f}"
        print(f"{n:>6} {grp:<18} {cells} {sl:>9} {rt:>10} {consistent:>8}")

print()
print("符号不一致 / 有缺口的根：")
for grp, n, ic, slope, ratio, consistent, missing in rows:
    flags = []
    if consistent.startswith("否"):
        flags.append("符号不一致")
    if missing:
        flags.append("缺任务: " + ",".join(missing))
    if flags:
        print(f"  {n}: " + "；".join(flags))
print()
print("解读提示：")
print("  - 趋势 < 0 且 |趋势| 明显 > 0 的根，2020→2024 在衰减；")
print("  - 1133、1131 若趋势也明显为负，说明逐年稳定性本可预警公榜不兑现；")
print("    若趋势接近 0，则逐年稳定性预测不了公榜，需照实写明。")
