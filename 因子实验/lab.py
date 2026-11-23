# -*- coding: utf-8 -*-
"""本地因子发现台：加载 30 分钟面板（磁盘缓存）、批量评估候选、与本地参照因子对照。

用法：
    python3 因子实验/lab.py --start 2023-01 --end 2023-12 --script 因子实验/cand_r1.py
脚本文件需定义 CANDS = {名称: 函数}；函数签名 fn(p) -> 与 p 等长的因子值。
p 已包含 localeval.build_panel 与 factor_lib.add_components 的中间量。

指标口径与 工具/本地评分/localeval.py 一致（官方标签缓存 + 风格/行业剔除后的五项指标）。
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "工具", "本地评分"))
import localeval as le          # noqa: E402
import factor_lib as fl         # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "结果")
os.makedirs(CACHE, exist_ok=True)
METRICS = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")


def get_panel(start, end, rebuild=False):
    key = "%s_%s" % (start, end)
    path = os.path.join(CACHE, "panel_%s.feather" % key.replace("-", ""))
    if os.path.exists(path) and not rebuild:
        t0 = time.time()
        p = pd.read_feather(path)
        print("[lab] 面板缓存 %s（%d 行，%.0fs）" % (os.path.basename(path), len(p), time.time() - t0))
        return p
    t0 = time.time()
    p = le.build_panel(le.load_bars(start, end))
    p = fl.add_components(p)
    print("[lab] 构建面板 %s~%s：%d 行，%.0fs" % (start, end, len(p), time.time() - t0))
    try:
        p.reset_index(drop=True).to_feather(path)
        print("[lab] 已缓存 %s" % os.path.basename(path))
    except Exception as e:
        print("[lab] 缓存写入失败：%s" % e)
    return p


def references(p):
    """本地参照因子（与公榜编号同名），用于给出可解释的相对排序。"""
    return {
        "ref_56日内反转": lambda q: q["r_day"],
        "ref_69三尺度": lambda q: fl.rank_blend(q, ["r_dvwap", "r_day", "r_30"]),
        "ref_90多尺度阶梯": lambda q: fl.ladder(q, ["30", "rv4", "rv8", "day"], False),
        "ref_104同伴残差1.5倍中性": lambda q: fl.neutral4(q, fl.peer_res_ladder(q), 1.5),
        "ref_107同伴残差": lambda q: fl.peer_res_ladder(q),
    }


def load_cands(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cand_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.CANDS)


def score(p, cands, tag=""):
    rows = {}
    for name, fn in cands.items():
        t0 = time.time()
        try:
            v = fn(p)
            v = pd.Series(np.asarray(v), index=p.index) if not isinstance(v, pd.Series) else v
            m = le.evaluate(v, p, label="y_off")
        except Exception as e:
            print("  [%s] 评估失败：%s: %s" % (name, type(e).__name__, e))
            continue
        rows[name] = {k: float(m[k]) for k in METRICS}
        rows[name]["秒"] = round(time.time() - t0, 1)
        rows[name]["覆盖"] = float(pd.Series(v).notna().mean())
    t = pd.DataFrame(rows).T
    if t.empty:
        return t
    t["本地综合分"] = le.composite(t)
    t = t.sort_values("本地综合分", ascending=False)
    print("\n[%s] %d 个候选，按本地综合分排序（含参照池）" % (tag, len(cands)))
    print(t.round(4).to_string())
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    p = get_panel(args.start, args.end, rebuild=args.rebuild)
    print("[lab] 评估截面 %d，y_off 非空 %.3f" % (p["is_eval"].sum(), float(p.loc[p["is_eval"], "y_off"].notna().mean())))
    cands = {} if args.no_ref else references(p)
    cands.update(load_cands(args.script))
    t = score(p, cands, tag=args.tag or "%s~%s" % (args.start, args.end))
    if args.out and not t.empty:
        t.round(6).to_csv(args.out)
        print("[lab] 已写出 %s" % args.out)


if __name__ == "__main__":
    main()
