# -*- coding: utf-8 -*-
"""从 工具/评分机/结果/*/result.json 生成批次因子说明文档（表格自动生成，避免手抄）。"""
import glob
import json
import os
import sys

WANT = {
    "base90": ("90 号（对照）", "—"),
    "base104": ("104 号（对照）", "—"),
    "104_fix930": ("104+09:30 修复（对照）", "—"),
    "113_dsd": ("113", "104 + 0.15 下行波动占比附加"),
    "114_softpeer": ("114", "软同伴（相关性 2 次方加权，k=100）"),
    "115_amp": ("115", "104 + 0.20 低振幅附加"),
    "116_soft_amp_dsd": ("116", "软同伴 + 振幅 0.15 + 下行波动 0.15"),
    "117_soft1": ("117", "软同伴（相关性 1 次方加权）"),
    "118_soft_frac2": ("118", "软同伴 + 四特征 2.0 倍中性"),
    "119_rv16": ("119", "软同伴 + 约 2 日尺度分量（六尺度）"),
    "120_soft_nofix": ("120", "同 114，但不补 09:30（口径对照）"),
}
rows = {}
for d in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "工具", "评分机", "结果", "*"))):
    f = os.path.join(d, "result.json")
    if not os.path.exists(f):
        continue
    j = json.load(open(f))
    nm = (j.get("meta") or {}).get("name")
    if nm in WANT and (j.get("metrics") or {}):
        rows[nm] = (os.path.basename(d), j)

out = ["# 第十七批因子说明（113–120，factor/discover 第一波）", "",
       "> 工作分支 `factor/discover`。本批共 8 个候选，全部为新方向（软同伴定义、下行波动占比、振幅、更长尺度），",
       "> 且**全部带 09:30 截面修复**（1 分钟数据没有 09:30:00 bar，09:30 窗口取 09:25 集合竞价 bar 的价格）。", ""]

out += ["## 0. 本轮的硬规则与口径", "",
        "- **必须补 09:30**：否则自评口径与真实榜单不一致（用户判定）。评分机会把“常数/全 0 截面”剔除，",
        "  因此它对 09:30 全 0 的因子只按 7 个截面求平均，而补齐后是 8 个截面——**同一候选必须与同为补齐的版本比较**。",
        "- 本地 `localeval` 口径只用于粗筛（R1–R5 共 60 余个方向），交付与否由评分机裁决。",
        "- 120 号是 114 号的“不补 09:30”对照，用于用真实公榜分数确认平台对常数截面的处理。", ""]

out += ["## 1. 平台口径结果（评分机，2023-01~2024-12，8 截面）", "",
        "| 编号 | 因子 | IC | IC_IR | 多空夏普 | 压力 | 换手 | 参照池百分位（IC/IR/LS/压力/换手） | 代理分 | 与104相关 | 耗时 | 任务号 |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|"]
for nm, (jid, j) in rows.items():
    m, q = j.get("metrics"), ((j.get("reference_pool") or {}).get("percentile") or {})
    label, desc = WANT[nm]
    proxy = sum(q.get(k, 0) for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")) / 5 if q else float("nan")
    corr = (j.get("correlations") or {}).get("104", {}).get("rank_corr_mean", float("nan"))
    out.append("| %s | %s | %.4f | %.3f | %.2f | %.3f | %.3f | %.2f / %.2f / %.2f / %.2f / %.2f | %.3f | %.3f | %.0fs | %s |" % (
        label, desc, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"],
        q.get("ic_mean", 0), q.get("ic_ir", 0), q.get("ls_sharpe", 0), q.get("stress", 0), q.get("turnover", 0),
        proxy, corr, (j.get("meta") or {}).get("produce_seconds", 0), jid))
out += ["", "分年（2023 为验证段，2024 为保留段）：", "",
        "| 编号 | 2023 IC / IC_IR / 压力 / 换手 | 2024 IC / IC_IR / 压力 / 换手 |", "|---|---|---|"]
for nm, (jid, j) in rows.items():
    by = j.get("by_year") or {}
    cells = []
    for y in ("2023", "2024"):
        v = by.get(y)
        cells.append("%.4f / %.3f / %.3f / %.3f" % (v["ic_mean"], v["ic_ir"], v["stress"], v["turnover"]) if v else "—")
    out.append("| %s | %s | %s |" % (WANT[nm][0], cells[0], cells[1]))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "文档/因子说明/第十七批因子说明_113-120.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("生成成功，%d 行" % len(out))
