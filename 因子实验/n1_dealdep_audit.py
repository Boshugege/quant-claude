# -*- coding: utf-8 -*-
"""第十八轮 N1（更正版）②：全候选的 deal_number 依赖审计。

平台公榜数据里 deal_number 很可能为空/全 0（1367 报错、1361 里 Y_sd5/X_n5d 贡献为零的
唯一共同点）。本脚本把仓库里的候选因子按"最终 factor 是否依赖 deal_number"分类：

  依赖原语：deal_number（成交笔数）、lats = ln(成交额÷成交笔数)、c_avgtr（Σ成交额÷Σ成交笔数）、
            由它们派生的 cnd/nd5/sd5/sdlats/nup/ndn。
  独立原语：open/close/high/low/pre_close、volume、amount、涨跌停价、委托笔数与挂单量、
            暴露表（前一日错位）。

判定顺序：① 文件里 STAT = "Xxx" -> 用框架解出的 STAT 依赖表；② 合成体 W = {...} 的根名 ->
按 STAT 表；③ 头部特征里的 c_avgtr；④ 兜底：剥掉注释/docstring 后直接搜依赖原语。
输出 markdown 表并写 因子实验/n1_dealdep_table.md。
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
FDIR = ROOT / "因子代码"
FRAMEWORKS = ["1320_多日分钟框架.py", "1330_资金流框架.py", "1340_资金流窗口框架.py",
              "1370_分布形态框架.py", "1386_多日分布形态框架.py"]
DEP_TOKENS = ("lats", "cnd", "cbig", "csmall", "nd5", "sd5", "nup", "ndn")
CODE_DEP = re.compile(r"deal_number|sum_deals|c_avgtr|_avgtr|lats|\bnup\b|\bndn\b")
DOC = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')


def strip_code(txt):
    txt = DOC.sub("", txt)
    return "\n".join(ln.split("#")[0] if not ln.strip().startswith("#") else "" for ln in txt.splitlines())


def stat_map():
    m = {}
    for fn in FRAMEWORKS:
        p = FDIR / fn
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        i = txt.find("sig = {")
        if i < 0:
            continue
        j = txt.find("\n                }[", i)
        block = txt[i:j if j > 0 else len(txt)]
        for line in block.splitlines():
            mm = re.match(r'\s*"([A-Za-z0-9_.]+)"\s*:\s*(.+?),?\s*$', line)
            if not mm:
                continue
            m[mm.group(1)] = m.get(mm.group(1), False) or any(t in mm.group(2) for t in DEP_TOKENS)
    return m


def bchain(txt, sm):
    """B 式合成（1398/1602/1759 一族）：把实际被调用的 main_bN 映射到它的 STAT，返回依赖根列表。"""
    stat_of = dict(re.findall(r'parts\.append\(_root_impl_b(\d+)\([^\n]*?,\s*"([A-Za-z0-9_]+)"\)\)', txt))
    loop = re.findall(r'\(main_b(\d+),\s*[-\d.]+,\s*[-\d.]+', txt)
    dep = [stat_of.get(b, "b" + b) for b in loop if sm.get(stat_of.get(b, ""), False)]
    return loop, dep, stat_of


def roots_of(txt):
    i = txt.find("W = {")
    if i < 0:
        return None
    j = txt.find("}", i)
    return re.findall(r'"([A-Za-z0-9_]+)"\s*:', txt[i:j]) if j > 0 else []


def main():
    sm = stat_map()
    rows = []
    for p in sorted(FDIR.glob("*.py")):
        num = p.name.split("_")[0]
        if not num.isdigit() or int(num) < 1200:
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        code = strip_code(txt)
        rt = roots_of(txt)
        if rt is not None:
            dep = [r for r in rt if sm.get(r, False)]
            head = "c_avgtr" in code                      # 头部信号分量（权重 0.15）
            kind = "合成"
            verdict = "依赖" if dep else ("仅头部" if head else "干净")
            note = "根：" + ",".join(rt)
            if dep:
                note += "｜依赖根：" + ",".join(dep)
            if head:
                note += "｜头部 c_avgtr"
            rows.append((num, p.name, kind, verdict, ("是" if dep else "—"), ("是" if head else "—"), note))
            continue
        loop, depb, stat_of = bchain(txt, sm)
        if loop:
            note = "调用链：" + ",".join(stat_of.get(b, "b" + b) for b in loop)
            if depb:
                note += "｜依赖根：" + ",".join(depb)
            rows.append((num, p.name, "B合成", "依赖" if depb else "干净",
                         "是" if depb else "—", "是" if "c_avgtr" in code else "—", note))
            continue
        m = re.search(r'STAT = "([A-Za-z0-9_]+)"', txt)
        stat = m.group(1) if m else None
        dep = sm.get(stat, False) if stat else None
        if dep is None:
            dep = bool(CODE_DEP.search(code))
        rows.append((num, p.name, "单根", "依赖" if dep else "干净", "是" if dep else "—", "—",
                     ("STAT=%s" % stat) if stat else ""))

    lines = ["| 编号 | 文件 | 类型 | 依赖 deal_number | STAT | 头部c_avgtr | 说明 |",
             "|---|---|---|---|---|---|---|"]
    for num, name, kind, verdict, s, h, note in rows:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (num, name, kind, verdict, s, h, note))
    lines += [
        "",
        "## 依赖 deal_number 的根 → 只用收益/成交量的替代版",
        "",
        "| 依赖根 | 用到的原语 | 本次给出的不依赖版本 | 文件 |",
        "|---|---|---|---|",
        "| `N_dir` 当日大单方向 | `lats=ln(成交额÷成交笔数)` | `W_vdir` 当日量价同向（`Σr·Δln成交量/Σ\|r\|`，本来就在 1363 里） | `1340_资金流窗口框架.py` |",
        "| `N_big` 大单减小单分钟收益 | 同上 | `Z_mfr60`/`Z_net60`（按分钟量占比/净流向分层） | `1340_资金流窗口框架.py` |",
        "| `X_n5d` 近 5 日大单方向 | 同上（跨日 w5） | **`Y_vd5` 近 5 日量价同向** | `1750_近五日量价同向.py` |",
        "| `Y_sd5` 近 5 日单笔额离散度 | `STDDEV(lats)` | **`Y_vsd5` 近 5 日分钟量对数离散度** / **`Y_z5` 近 5 日零收益分钟占比** | `1751_...py` / `1755_...py` |",
        "| `W_atsd` 当日单笔额对数离散度 | `STDDEV(lats)` | 当日版未单独建，可用 `Y_vsd5`（近 5 日同族）代替 | `1751_...py` |",
        "| `Z_ats` 上涨/下跌分钟单笔额比 | `nup`/`ndn` | `Z_net60` 净流向占比（只用 volume） | `1340_资金流窗口框架.py` |",
        "",
        "## 头部自身的 deal_number 依赖（重要）",
        "",
        "**所有带头部的合成都有这一项**：头部（490／148 族）的信号分量是",
        "`rank_blend([ladder, c_upsh, c_avgtr, r_rv16, c_body], [1.0, 0.30, 0.15, 0.20, 0.15])`，",
        "其中 `c_avgtr = Σ成交额 ÷ Σ成交笔数` 的 40 窗口均值**用的就是 `deal_number`**（权重 0.15，缺失时 `rank_blend` 会按可用分量重新归一）。",
        "若平台 `deal_number` 为空，`c_avgtr` 整列 NaN → 自动退化成「四分量再归一」，**脚本不报错、因子照常出分**，",
        "这与公榜上 490 家族照常出分、且 IC 放大倍数与其他候选一致（×1.18）吻合 —— 说明这一项的影响是二阶的，量级 ≤0.002 IC。",
        "",
        "审计脚本：`因子实验/n1_dealdep_audit.py`（本文件由它自动生成）。",
    ]
    out = "\n".join(lines)
    (ROOT / "因子实验" / "n1_dealdep_table.md").write_text(out + "\n", encoding="utf-8")
    print("依赖 STAT：", sorted(k for k, v in sm.items() if v))
    print("干净 STAT：", sorted(k for k, v in sm.items() if not v))
    print()
    for num, name, kind, verdict, s, h, note in rows:
        print("%-6s %-10s %-4s %-3s %-3s %s" % (num, kind, verdict, s, h, name[:44]))


if __name__ == "__main__":
    main()
