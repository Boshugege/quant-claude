#!/usr/bin/env python3
"""因子赛道决赛材料的数据图（报告版 *_r，幻灯片版 *_s）。

只读仓库数据，图输出到本脚本所在目录（PDF 矢量 + 150 dpi PNG）。
运行：python3 make_figures.py            # 全部重画（输出到本目录）并打印图中用到的全部数字
      python3 make_figures.py --report-palette slides --out 某目录   # 报告版改用幻灯片配色（橙色强调）

依赖：python3、matplotlib 3.9、numpy、fontTools（macOS 系统字体 Avenir Next 与
Hiragino Sans GB 均为字体集 .ttc，脚本用 fontTools 把需要的字重抽到临时目录再注册）。

数据来源（全部只读）：
  工具/评分机/任务索引/评分机全部任务索引_2026-10-31.csv  本地评测五项与 by_year（按 job_id 去重，取首次出现）
  因子实验/public_rows.json                               公榜五项
  因子实验/models/h2_avg16.json                           基础组合（16 个浅层树平均）的模型文件
以下数字仓库里没有可直接读取的原始表，按已核对文档写入常量（出处见各常量注释）：
  1669 私榜五项（用户提供，〔00〕§1.1）；14:30 截面 RankIC 阶梯（〔因03〕§7.4、〔因02〕§4.7–4.9）；
  研究各阶段首次提交公榜的日期（〔00〕§2.2、〔因06〕§0）。
图中不出现排行榜综合得分的数值：纵轴原为得分的两处（换手交换图右栏、研究过程图上栏）
分别改为公榜 ICIR 与压力稳定性、研究阶段时间线。
"""
import argparse
import collections
import csv
import json
import math
import tempfile
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

ROOT = Path("/Users/user/Desktop/quant-claude")
IDX = ROOT / "工具/评分机/任务索引/评分机全部任务索引_2026-10-31.csv"
PUB = ROOT / "因子实验/public_rows.json"
TREE = ROOT / "因子实验/models/h2_avg16.json"
OUT = Path(__file__).resolve().parent

MINUS = "−"
CM = 1 / 2.54

# ---------------------------------------------------------------- 字体与配色

def setup_fonts():
    """把 Avenir Next Regular／Demi Bold、Hiragino Sans GB W3／W6 从系统字体集抽出并注册。"""
    from fontTools.ttLib import TTCollection

    cache = Path(tempfile.gettempdir()) / "fx_fig_fonts"
    cache.mkdir(exist_ok=True)
    want = {
        "Avenir Next Regular": "AvenirNext-Regular.ttf",
        "Avenir Next Demi Bold": "AvenirNext-DemiBold.ttf",
        "Hiragino Sans GB W3": "HiraginoSansGB-W3.otf",
        "Hiragino Sans GB W6": "HiraginoSansGB-W6.otf",
    }
    for ttc in ("/System/Library/Fonts/Avenir Next.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"):
        for face in TTCollection(ttc).fonts:
            full = face["name"].getDebugName(4)
            if full in want:
                p = cache / want[full]
                if not p.exists():
                    face.save(str(p))
                fm.fontManager.addfont(str(p))
    plt.rcParams.update({
        "font.family": ["Avenir Next", "Hiragino Sans GB"],  # 西文与数字 Avenir Next，中文回退到冬青黑体
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.unicode_minus": True,
        "pdf.fonttype": 3,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.solid_capstyle": "round",
    })


PALETTES = {
    # 幻灯片模板 slides_preamble.tex：snavy / sorange / sline / stint / sgrey / sink
    "slides": dict(navy="#0F2E5A", accent="#E8772E", light="#C9D3E0", mid="#7C8FAD",
                   tint="#E8EFF8", ink="#1E2329", sub="#5E6672", grid="#E4E9F0"),
    # 报告模板 report_preamble.tex：rnavy / raccent / rline / rgrey
    "report": dict(navy="#1B365D", accent="#C0392B", light="#BFC5CC", mid="#7A828C",
                   tint="#EEF2F7", ink="#1E2329", sub="#6B737D", grid="#E6E9ED"),
}
REPORT_PALETTE = "report"
C = dict(PALETTES["report"])


def use_palette(mode):
    C.clear()
    C.update(PALETTES["slides" if mode == "s" else REPORT_PALETTE])
    for k in ("text.color", "axes.labelcolor"):
        plt.rcParams[k] = C["ink"]
    for k in ("xtick.color", "ytick.color", "axes.edgecolor"):
        plt.rcParams[k] = C["sub"]
    plt.rcParams["xtick.labelcolor"] = C["ink"]
    plt.rcParams["ytick.labelcolor"] = C["ink"]


# ---------------------------------------------------------------- 小工具

def num(v, d, sign=False):
    s = f"{abs(v):.{d}f}"
    if v < 0 and float(s) != 0:
        return MINUS + s
    return ("+" + s) if sign else s


def pct(v, sign=True):
    s = f"{abs(v) * 100:.0f}%"
    if v < 0 and s != "0%":
        return MINUS + s
    return ("+" + s) if sign else s


def new_fig(w_cm, h_cm, **kw):
    return plt.figure(figsize=(w_cm * CM, h_cm * CM), layout="constrained", **kw)


def clean(ax, left=True, grid=True, bottom=True):
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if grid:
        ax.yaxis.grid(True, color=C["grid"], lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    if not left:
        ax.tick_params(axis="y", length=0)


def hide_y(ax):
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])


def panel_label(ax, text, x=0.0, y=1.02):
    """面板说明（坐标轴含义），不是图标题。"""
    return ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", color=C["sub"], fontsize=10)


def label(ax, x, y, text, color=None, weight="normal", dy=3, dx=0, ha="center", va="bottom", rotation=0,
          box=False):
    """在数据点 (x, y) 旁按“点”为单位偏移写数值。"""
    kw = {}
    if box:
        kw["bbox"] = dict(fc="white", ec="none", pad=0.6)
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                       color=color or C["ink"], fontweight=weight, rotation=rotation, annotation_clip=False,
                       zorder=6, **kw)


def arrow(ax, x0, y0, x1, y1, color, lw=1.1, style="-|>", ms=8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=ms))


def align_panel_labels(fig, pairs):
    """把面板说明的左端对齐到该坐标轴 y 刻度标签的左端。"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax, t in pairs:
        x0 = min(lb.get_window_extent(r).x0 for lb in ax.get_yticklabels() if lb.get_text())
        t.set_x(ax.transAxes.inverted().transform((x0, 0))[0])


def save(fig, name):
    # 数据坐标里的文字不参与版面计算（都放在坐标轴内），避免压缩绘图区
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_transform() != ax.transAxes:
                t.set_in_layout(False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  -> {name}.pdf / .png")


# ---------------------------------------------------------------- 数据

def load_index():
    rows = list(csv.DictReader(open(IDX, encoding="utf-8")))
    by = {}
    for x in rows:
        by.setdefault(x["job_id"], x)  # 在线与归档各列一次，按 job_id 去重
    names = collections.defaultdict(list)
    for x in by.values():
        names[x["name"]].append(x)
    return rows, by, names


ROWS, BY, NAMES = load_index()
PUBROWS = json.load(open(PUB, encoding="utf-8"))
KEYS = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")


def job(name):
    xs = NAMES.get(name)
    assert xs, f"任务索引里没有 {name}"
    assert len(xs) == 1, f"{name} 有 {len(xs)} 个任务，需要指定"
    return xs[0]


def five(name):
    x = job(name)
    return {k: float(x[k]) for k in KEYS}


def by_year(name):
    return json.loads(job(name)["by_year"])


# 链 A：基础组合（拟合 2020–2022 的浅层树）逐步加入增量信号与后处理，本地 2023–2024 合并口径
CHAIN_A = [
    ("基础组合", "t17_base"),
    ("+缓冲带（整体加分）", "y17_buf70"),
    ("+委托规模比", "qs_q30buf70"),
    ("改为最小幅度调整", "lm_q30L60buf100"),
    ("+订单流等三项", "cd_relce_q30L60buf115"),
    ("+收盘竞价惯性", "c4_a_q30L60buf100"),
    ("+14:30 反向调整", "t14_a_q30L60buf100"),
    ("+14:30 放量项", "w14_ad_q30L60buf120"),
    ("缓冲前中性化", "rslag_w14_q30L60buf70"),
    ("缓冲强度改为 1.0", "rslag_w14_q30L60buf100"),
    ("+尾盘与关注度四项", "rd_c6_q30L60buf110"),
]
FINAL = "rd_c6_q30L60buf110"          # 最终因子配方（1669）
LIN_BASE = "f10_base"                  # 线性合成基准
LIN_FULL = "rsrd_c6_q30L60buf80"       # 线性合成基准 + 同一套增量信号与后处理
TRADE_ADD = ["qs_q30buf70", "qc_q30buf100", "qc_q30buf150", "qx_q30buf200", "qx_q30buf250"]
TRADE_NEU = ["rslag_w14_q30L60buf70", "rslag_w14_q30L60buf100", "rslag_w14_q30L60buf130",
             "rslag_w14_q30L60buf150", "rslag_w14_q30L60buf170"]
TRADE_C6 = ["rd_c6_q30L60buf70", "rd_c6_q30L60buf80", "rc_c6_q30L60buf100", "rd_c6_q30L60buf110",
            "rd_c6_q30L60buf115"]
TRADE_BETA = [0.7, 0.8, 1.0, 1.1, 1.15]
PUB_TURN = ["1642", "1644", "1643", "1645", "1646", "1647", "1648"]  # 同一底座、加性缓冲强度不同（10-22 同日回报）
PLACEBO_BASE = "t14_a_q30L60buf100"

# 1669 私榜五项（用户提供，〔00〕§1.1）：RankIC／ICIR／多空夏普／压力稳定性／换手率
PRIVATE_1669 = dict(ic_mean=0.0790, ic_ir=1.5593, ls_sharpe=45.6269, stress=1.4086, turnover=0.2662)
# 14:30 截面的 RankIC（本地 2023–2024），依次加入收盘竞价惯性、三个反向调整项、两个放量项（〔因03〕§7.4；〔因02〕§4.7–4.9）
SLOT1430 = [("加入前", 0.105, 0.105), ("+收盘竞价惯性", 0.110, 0.110),
            ("+14:30 反向调整", 0.117, 0.119), ("+14:30 放量项", 0.128, 0.130)]
# 研究阶段：首次提交公榜的日期与阶段名（〔00〕§2.2 逐日时间线；〔因06〕§0 公榜轨迹）。图中不写综合得分
MILESTONES = [
    (date(2026, 10, 10), "委托笔数单因子"),
    (date(2026, 10, 11), "多尺度反转"),
    (date(2026, 10, 12), "同伴残差与中性化"),
    (date(2026, 10, 18), "浅层树组合"),
    (date(2026, 10, 19), "16 个树平均"),
    (date(2026, 10, 22), "缓冲带与委托规模比"),
    (date(2026, 10, 23), "14:30 收盘竞价惯性"),
    (date(2026, 10, 24), "缓冲前中性化"),
    (date(2026, 10, 28), "尾盘与关注度四项"),
]

# 基础组合 27 列输入的中文名与分组（事实清单 C1）
FEAT_CN = {
    "L_lab10": "同时段收盘相对均价习惯（10 日）", "M_q_lin": "价格冲击加权均价位置", "W_vdir": "量价同向度",
    "W_zero": "零收益分钟占比", "Z_mfr60": "近 60 分钟买方成交量占比", "Z_mfr90": "近 90 分钟买方成交量占比",
    "s139": "同伴残差反转 A", "s155": "同伴残差反转 B", "s161": "同伴残差反转 C", "H152": "同伴残差反转 D",
    "k1381": "日内已实现峰度（取负）", "k1385": "近 5 日已实现偏度（取负）", "k1406": "前一日收盘竞价价格冲击",
    "H1363": "线性合成初值", "L_5": "同时段收盘相对均价习惯（5 日）", "VSH20": "下一时段成交额占比常态",
    "PACE20": "当日成交节奏相对常态", "DOW10": "同星期几同时段习惯", "CLS20": "尾盘收盘相对均价习惯",
    "CLOSEPOS20": "收盘位置强度",
}
FEAT_SHORT = {"L_5": "同时段习惯（5 日）", "L_lab10": "同时段习惯（10 日）", "k1385": "近 5 日偏度（取负）",
              "k1381": "日内峰度（取负）"}
for _s in ("1000", "1030", "1100", "1130", "1330", "1400", "1430"):
    FEAT_CN[f"slot{_s}"] = f"{_s[:2]}:{_s[2:]} 时段哑变量"
GROUPS = [
    ("同伴残差反转", ["s139", "s155", "s161", "H152"]),
    ("分钟资金流与量价", ["L_lab10", "M_q_lin", "W_vdir", "W_zero", "Z_mfr60", "Z_mfr90"]),
    ("同时段季节性", ["L_5", "VSH20", "PACE20", "DOW10"]),
    ("分布形态与竞价", ["k1381", "k1385", "k1406"]),
    ("时段哑变量", [f"slot{s}" for s in ("1000", "1030", "1100", "1130", "1330", "1400", "1430")]),
    ("收盘习惯类", ["CLS20", "CLOSEPOS20"]),
    ("线性合成初值", ["H1363"]),
]

# 增量信号单独使用时的逐年 RankIC
DECAY_ROWS = [
    ("s20_ql_s1only", "一档委托规模比（单独版）"),
    ("s35_of_l10_rel", "主动买卖失衡"),
    ("s20_ql_end", "段末委托结构"),
    ("s16_l10_clv", "最后 10 分钟收盘位置"),
    ("s25_ts_sv_l10", "最后 10 分钟签名成交量"),
    ("s16_l10_ret", "最后 10 分钟收益"),
    ("s6_amt_5d20d", "成交额消退"),
    ("s10_am_drift", "上午回吐（10:30–11:30）"),
    ("s5_c_abnvol", "本段放量（全时段版）"),
    ("s18_abnamt_day", "当日至今放量（全时段版）"),
    ("s37_auc_m20", "收盘竞价惯性 20 日（仅 14:30）"),
    ("s37_auc_m5", "收盘竞价惯性 5 日（仅 14:30）"),
]
DECAY_GROUP_BREAKS = [3, 6, 8, 10]  # 委托结构／尾盘／关注度与上午／放量／竞价

# 里程碑版本的逐年 RankIC（任务名 -> 年份）
MILESTONE_VERSIONS = [
    ("早期反转类\n主信号", ["y20_490", "fm_490_indneu25"]),
    ("同伴残差\n反转", ["o20_s161", "o21_s161", "fm_s161"]),
    ("线性合成\n初值", ["o20_1363", "o21_1363", "fm_1363"]),
    ("线性合成\n基准", ["f10_base"]),
    ("基础组合\n（浅层树）", ["d_1617_2023", "d_1617_2024"]),
    ("最终因子", [FINAL]),
]
TREE_LIN = dict(lin=("d_1610_2023", "d_1610_2024", "1610"), tree=("d_1617_2023", "d_1617_2024", "1617"))


def collect():
    D = {}
    # 树分裂重要性
    m = json.load(open(TREE, encoding="utf-8"))
    fo = m["feat_order"]
    w = collections.Counter()
    root = collections.Counter()
    ntree = 0
    for t in m["model"]["trees"]:
        ntree += 1
        for node in t:
            if node[0] == "split":
                f = fo[int(node[2])]
                depth = node[-1]
                w[f] += 2.0 ** (-depth)
                if depth == 0:
                    root[f] += 1
    s = sum(w.values())
    D["tree_n"] = ntree
    D["feat_pct"] = {f: w[f] / s * 100 for f in fo}
    D["group_pct"] = [(g, sum(w[f] for f in fs) / s * 100) for g, fs in GROUPS]
    D["root"] = root
    assert abs(sum(v for _, v in D["group_pct"]) - 100) < 1e-6
    # 树与线性合成
    tl = {}
    for k, (a, b, p) in TREE_LIN.items():
        y23 = by_year(a)["2023"]
        y24 = by_year(b)["2024"]
        pb = PUBROWS[p]
        tl[k] = dict(ic=[y23["ic_mean"], y24["ic_mean"], pb["ic"]], ir=[y23["ic_ir"], y24["ic_ir"], pb["ir"]],
                     to=[y23["turnover"], y24["turnover"], pb["to"]])
    D["tree_lin"] = tl
    D["chain_a"] = [(lab, five(n)) for lab, n in CHAIN_A]
    D["lin_years"] = (by_year(LIN_BASE), by_year(LIN_FULL))
    D["trade"] = dict(add=[five(n) for n in TRADE_ADD], neu=[five(n) for n in TRADE_NEU],
                      c6=[five(n) for n in TRADE_C6])
    D["pub_turn"] = [(PUBROWS[k]["to"], PUBROWS[k]["ir"], PUBROWS[k]["st"]) for k in PUB_TURN]
    base = five(PLACEBO_BASE)["ic_mean"]
    D["placebo"] = sorted(float(x["ic_mean"]) - base for x in BY.values() if x["name"].startswith("ss_"))
    D["steps"] = dict(
        slot1430=five("t14_a_q30L60buf100")["ic_mean"] - five("c4_a_q30L60buf100")["ic_mean"],
        neutral=five("rslag_w14_q30L60buf70")["ic_mean"] - five("w14_ad_q30L60buf120")["ic_mean"],
        c6=five(FINAL)["ic_mean"] - five("rslag_w14_q30L60buf100")["ic_mean"],
    )
    D["decay"] = [(lab, n, job(n)["status"], {k: v["ic_mean"] for k, v in by_year(n).items()}) for n, lab in DECAY_ROWS]
    D["final_years"] = by_year(FINAL)
    D["final_all"] = five(FINAL)
    D["public_1669"] = dict(ic_mean=PUBROWS["1669"]["ic"], ic_ir=PUBROWS["1669"]["ir"], ls_sharpe=PUBROWS["1669"]["ls"],
                            stress=PUBROWS["1669"]["st"], turnover=PUBROWS["1669"]["to"])
    mv = []
    for lab, ns in MILESTONE_VERSIONS:
        yy = {}
        for n in ns:
            yy.update({k: v["ic_mean"] for k, v in by_year(n).items()})
        if ns == [FINAL]:
            yy = {k: v for k, v in yy.items() if k in ("2023", "2024")}
        mv.append((lab, yy))
    D["milestones_by_year"] = mv
    day = collections.Counter(x["created"][:10] for x in BY.values() if x["created"])
    D["daily"] = sorted(day.items())
    D["n_jobs"] = len(BY)
    D["n_rows"] = len(ROWS)
    return D


# ---------------------------------------------------------------- 图

def fig_tree_importance(D, mode):
    """基础组合 1920 棵树的分裂权重（每次分裂按 2^−深度 加权）。"""
    gp = sorted(D["group_pct"], key=lambda t: -t[1])
    rev = set(dict(GROUPS)["同伴残差反转"])
    top = sorted(D["feat_pct"].items(), key=lambda t: -t[1])[:12]
    root_a = D["root"]["s139"]

    def hbars(ax, names, vals, cols, xmax, bold_first=True):
        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals, height=0.62, color=cols, zorder=2)
        ax.set_yticks(y, names)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(vals) - 0.4)
        ax.set_xticks([])
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=4)
        return y

    gnames = [g for g, _ in gp]
    gvals = [v for _, v in gp]
    gcols = [C["accent"]] + [C["navy"]] * (len(gp) - 1)
    if mode == "s":
        fig = new_fig(7.7, 5.0)
        ax = fig.add_subplot()
        y = hbars(ax, gnames, gvals, gcols, max(gvals) * 1.24)
        for i, (yi, v) in enumerate(zip(y, gvals)):
            label(ax, v, yi, f"{v:.1f}%", color=C["accent"] if i == 0 else C["ink"],
                  weight="semibold" if i == 0 else "normal", dx=3, dy=0, ha="left", va="center")
        save(fig, "fig_tree_importance_s")
        return

    fig = new_fig(15.6, 7.4)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    ax = fig.add_subplot(gs[0])
    y = hbars(ax, gnames, gvals, gcols, max(gvals) * 1.30)
    for i, (yi, v) in enumerate(zip(y, gvals)):
        label(ax, v, yi, f"{v:.1f}%", color=C["accent"] if i == 0 else C["ink"],
              weight="semibold" if i == 0 else "normal", dx=3, dy=0, ha="left", va="center")
    t1 = panel_label(ax, "按信号类别汇总", y=1.01)
    ax2 = fig.add_subplot(gs[1])
    fvals = [v for _, v in top]
    fcols = [C["accent"] if f in rev else C["navy"] for f, _ in top]
    y2 = hbars(ax2, [FEAT_SHORT.get(f, FEAT_CN[f]) for f, _ in top], fvals, fcols, max(fvals) * 2.1)
    for yi, (f, v) in zip(y2, top):
        if f == "s139":
            label(ax2, v, yi, f"{v:.1f}%", color=C["accent"], weight="semibold", dx=3, dy=0, ha="left", va="center")
            label(ax2, v, yi, f"根节点 {root_a} 次", color=C["accent"], dx=3, dy=-6, ha="left", va="top")
        else:
            label(ax2, v, yi, f"{v:.1f}%", color=C["accent"] if f in rev else C["ink"], dx=3, dy=0,
                  ha="left", va="center")
    t2 = panel_label(ax2, "前 12 个基础信号", y=1.01)
    align_panel_labels(fig, [(ax, t1), (ax2, t2)])
    save(fig, "fig_tree_importance_r")


def _pair_bars(ax, groups, a, b, d, ca, cb, delta_fmt, la, lb, ymax, bw=0.34, gap=0.07, val_inside=False,
               inner_a=None):
    """两组对比柱：数值标在柱顶（或柱内竖排），差值写在两柱上方。"""
    x = np.arange(len(groups))
    xa = x - (bw + gap) / 2
    xb = x + (bw + gap) / 2
    ax.bar(xa, a, bw, color=ca, zorder=2, label=la)
    ax.bar(xb, b, bw, color=cb, zorder=2, label=lb)
    for i in range(len(x)):
        if val_inside:
            label(ax, xa[i], a[i], num(a[i], d), color=inner_a or "white", dy=-4, va="top", rotation=90)
            label(ax, xb[i], b[i], num(b[i], d), color="white", dy=-4, va="top", rotation=90)
            dd = 4
        else:
            label(ax, xa[i], a[i], num(a[i], d))
            label(ax, xb[i], b[i], num(b[i], d))
            dd = 18
        label(ax, x[i], max(a[i], b[i]), delta_fmt(b[i] - a[i]), color=cb, weight="normal" if val_inside else "semibold",
              dy=dd)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    hide_y(ax)
    ax.tick_params(axis="x", length=0, pad=4)
    return x


def _legend_top(fig, ncol, spacing=1.6, handles=None, labels=None):
    kw = dict(loc="outside upper center", ncol=ncol, frameon=False, handlelength=1.0, handleheight=0.8,
              columnspacing=spacing, handletextpad=0.5)
    if handles is not None:
        fig.legend(handles, labels, **kw)
    else:
        fig.legend(**kw)


def fig_tree_vs_linear(D, mode):
    """同一批输入：线性合成与浅层树组合（基础组合），2023、2024 年样本外与公榜。"""
    tl = D["tree_lin"]
    groups = ["2023 年\n样本外", "2024 年\n样本外", "公榜"]
    if mode == "s":
        fig = new_fig(7.7, 5.0)
        ax = fig.add_subplot()
        _pair_bars(ax, groups, tl["lin"]["ic"], tl["tree"]["ic"], 4, C["light"], C["accent"],
                   lambda v: num(v, 4, True), "线性合成", "浅层树组合", ymax=0.133, bw=0.34, gap=0.14)
        _legend_top(fig, 2)
        save(fig, "fig_tree_vs_linear_s")
        return
    fig = new_fig(15.6, 6.8)
    gs = fig.add_gridspec(1, 2, wspace=0.10)
    ax = fig.add_subplot(gs[0])
    _pair_bars(ax, groups, tl["lin"]["ic"], tl["tree"]["ic"], 4, C["light"], C["accent"],
               lambda v: num(v, 4, True), "线性合成", "浅层树组合（基础组合）", ymax=0.131, bw=0.30, gap=0.24)
    panel_label(ax, "RankIC", y=0.98)
    ax2 = fig.add_subplot(gs[1])
    _pair_bars(ax2, groups, tl["lin"]["ir"], tl["tree"]["ir"], 3, C["light"], C["accent"],
               lambda v: num(v, 3, True), None, None, ymax=2.69, bw=0.30, gap=0.24)
    panel_label(ax2, "ICIR", y=0.98)
    h, l_ = ax.get_legend_handles_labels()
    _legend_top(fig, 2, handles=h, labels=l_)
    save(fig, "fig_tree_vs_linear_r")


def fig_increment_chain(D, mode):
    """链 A：从基础组合到最终因子，逐步加入增量信号与后处理（本地 2023–2024）。"""
    labs = [lab for lab, _ in D["chain_a"]]
    ic = [v["ic_mean"] for _, v in D["chain_a"]]
    to = [v["turnover"] for _, v in D["chain_a"]]
    n = len(labs)
    x = np.arange(n)
    xr = n - 1 + 0.62  # 右侧差值箭头的位置
    fig = new_fig(15.6, 10.4)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[0])
    ax.hlines(ic[0], 0, xr, color=C["sub"], lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.plot(x, ic, color=C["navy"], lw=1.0, zorder=2)
    ax.scatter(x[:-1], ic[:-1], s=22, color=C["navy"], zorder=3)
    ax.scatter([x[-1]], [ic[-1]], s=50, color=C["accent"], zorder=4)
    for xi, v in zip(x, ic):
        last = xi == n - 1
        below = xi in (1, 3, 5)  # 这几个点的数值写在下方，避开虚线与相邻数值
        label(ax, xi, v, num(v, 4), color=C["accent"] if last else C["ink"],
              weight="semibold" if last else "normal", dy=-6 if below else 5, va="top" if below else "bottom")
    arrow(ax, xr, ic[0], xr, ic[-1], C["accent"], lw=1.2)
    label(ax, xr, (ic[0] + ic[-1]) / 2, num(ic[-1] - ic[0], 4, True), color=C["accent"], weight="semibold",
          dx=4, dy=0, ha="left", va="center")
    # 两组同换手配对：括线画在数值上方
    for i, j in ((7, 8), (9, 10)):
        d = ic[j] - ic[i]
        yb = max(ic[i], ic[j]) + 0.0031
        ax.plot([i, i, j, j], [yb - 0.0003, yb, yb, yb - 0.0003], color=C["mid"], lw=0.8)
        label(ax, (i + j) / 2, yb, f"同换手\n{num(d, 4, True)}", color=C["mid"], dy=2)
    ax.set_xlim(-0.5, n - 1 + 1.95)
    ax.set_ylim(0.0815, 0.1000)
    ax.yaxis.set_major_locator(MultipleLocator(0.004))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    clean(ax)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    panel_label(ax, "RankIC", y=1.0)

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    cols = [C["navy"]] + [C["light"]] * (n - 2) + [C["accent"]]
    ax2.bar(x, to, 0.62, color=cols, zorder=2)
    for xi, v in zip(x, to):
        last = xi == n - 1
        label(ax2, xi, v, num(v, 3), color=C["accent"] if last else C["ink"], weight="semibold" if last else "normal")
    ax2.hlines(to[0], 0.31, xr, color=C["sub"], lw=0.7, ls=(0, (3, 2)), zorder=1)
    arrow(ax2, xr, to[0], xr, to[-1], C["accent"], lw=1.2)
    label(ax2, xr, (to[0] + to[-1]) / 2, pct(to[-1] / to[0] - 1), color=C["accent"], weight="semibold",
          dx=4, dy=0, ha="left", va="center")
    ax2.set_ylim(0, 0.60)
    hide_y(ax2)
    ax2.set_xticks(x, labs, rotation=32, ha="right", rotation_mode="anchor")
    ax2.tick_params(axis="x", length=0, pad=3)
    panel_label(ax2, "换手率", y=1.0)
    save(fig, "fig_increment_chain_r")


def fig_five_years(D, mode):
    """以线性合成基准为基础组合，加入同一套增量信号与后处理前后的逐年 RankIC。"""
    a, b = D["lin_years"]
    years = ["2020", "2021", "2022", "2023", "2024"]
    va = [a[y]["ic_mean"] for y in years]
    vb = [b[y]["ic_mean"] for y in years]
    s = mode == "s"
    fig = new_fig(8.1, 5.0) if s else new_fig(15.6, 6.8)
    ax = fig.add_subplot()
    ymax = 0.157 if s else 0.153
    ax.axvspan(-0.55, 2.5, color=C["tint"], zorder=0, lw=0)
    ax.text(1.0, ymax * 0.99, "未参与增量信号筛选", ha="center", va="top", color=C["sub"])
    ax.text(3.5, ymax * 0.99, "筛选所用年份", ha="center", va="top", color=C["sub"])
    _pair_bars(ax, [f"{y} 年" for y in years], va, vb, 4, C["light"], C["accent"],
               lambda v: num(v, 4, True), "线性合成基准",
               "加增量信号与后处理" if s else "加入同一套增量信号与后处理", ymax=ymax,
               bw=0.40 if s else 0.34, gap=0.04 if s else 0.07, val_inside=s, inner_a=C["navy"])
    ax.set_xlim(-0.55, 4.55)
    _legend_top(fig, 2, spacing=1.0 if s else 1.6)
    save(fig, f"fig_five_years_{mode}")


def fig_turnover_tradeoff(D, mode):
    """换手率与 RankIC 的交换（本地 2023–2024）；报告版右栏为公榜 ICIR、压力稳定性随换手率的变化。"""
    tr = D["trade"]
    s = mode == "s"

    def slope(rows):
        return (rows[0]["ic_mean"] - rows[-1]["ic_mean"]) / (rows[0]["turnover"] - rows[-1]["turnover"]) * 0.1

    def left(ax):
        add, neu, c6 = tr["add"], tr["neu"], tr["c6"]
        for rows, col in ((add, C["light"]), (neu, C["navy"]), (c6, C["accent"])):
            xs = [r["turnover"] for r in rows]
            ys = [r["ic_mean"] for r in rows]
            ax.plot(xs, ys, color=col, lw=1.6, zorder=2)
            ax.scatter(xs, ys, s=16, color=col, zorder=3)
        f = D["final_all"]
        fx, fy = f["turnover"], f["ic_mean"]
        ax.scatter([fx], [fy], s=80, color=C["accent"], edgecolor="white", lw=1.4, zorder=5)
        if s:
            add_txt = f"整体加分的缓冲带\n每 0.1 换手 {num(-slope(add), 4)}"
            neu_txt = f"中性化后最小幅度调整\n每 0.1 换手 {num(-slope(neu), 4)}"
            c6_txt = "加尾盘与关注度四项"
            fin_txt = f"最终因子\n{num(fx, 3)}，{num(fy, 4)}"
            fin_end = (0.262, 0.0951)
        else:
            add_txt = f"缓冲带（整体加分）\n每降 0.1 换手，RankIC {num(-slope(add), 4)}"
            neu_txt = f"中性化后最小幅度调整\n每降 0.1 换手，RankIC {num(-slope(neu), 4)}"
            c6_txt = "再加尾盘与关注度四项"
            fin_txt = f"最终因子\n换手 {num(fx, 3)}\nRankIC {num(fy, 4)}"
            fin_end = (0.236, 0.0950)
        ax.text(0.418, 0.0706, add_txt, ha="right", va="bottom", color=C["mid"], linespacing=1.3)
        ax.text(0.172, 0.0878 if s else 0.0874, neu_txt, ha="left", va="top", color=C["navy"], linespacing=1.3)
        ax.text(0.418, c6[0]["ic_mean"] + 0.0008, c6_txt, ha="right", va="bottom", color=C["accent"])
        ax.text(0.172, 0.0990, fin_txt, ha="left", va="top", color=C["accent"], fontweight="semibold",
                linespacing=1.3)
        ax.plot([fin_end[0], fx - 0.005], [fin_end[1], fy + 0.0003], color=C["accent"], lw=0.8)
        ax.set_xlim(0.165, 0.42)
        ax.set_ylim(0.0700, 0.0995)
        ax.xaxis.set_major_locator(MultipleLocator(0.1 if s else 0.05))
        ax.yaxis.set_major_locator(MultipleLocator(0.01 if s else 0.005))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}" if s else f"{v:.2f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}" if s else f"{v:.3f}"))
        ax.set_xlabel("换手率（本地 2023–2024）")
        ax.set_ylabel("RankIC")
        clean(ax)

    if s:
        fig = new_fig(8.1, 5.4)
        left(fig.add_subplot())
        save(fig, "fig_turnover_tradeoff_s")
        return
    fig = new_fig(15.6, 7.4)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.06)
    left(fig.add_subplot(gs[0]))
    ax2 = fig.add_subplot(gs[1])
    pts = sorted(D["pub_turn"])  # (公榜换手率, ICIR, 压力稳定性)，同一基础组合、整体加分缓冲强度不同
    xs = [p[0] for p in pts]
    k = int(np.argmin([abs(x - 0.39) for x in xs]))  # 换手 0.39 的版本：这一族里公榜得分最高（只作定性标注）
    ax2.axvline(xs[k], color=C["sub"], lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax2.text(xs[k] - 0.008, 2.035, "公榜得分\n在此最高", ha="right", va="top", color=C["sub"], linespacing=1.25)
    # 线名写在线外的空白处；左端点（换手 0.20）的数值写在点的左上（ICIR）与右下（压力稳定性）
    for j, (name, col, (nx, ny, nha), dy) in enumerate((("ICIR", C["navy"], (0.285, 1.765, "right"), 4),
                                                        ("压力稳定性", C["mid"], (0.318, 1.615, "left"), -4))):
        ys = [p[1 + j] for p in pts]
        ax2.plot(xs, ys, color=col, lw=1.6, zorder=2)
        ax2.scatter(xs, ys, s=18, color=col, zorder=3)
        ax2.text(nx, ny, name, ha=nha, va="center", color=col, fontweight="semibold")
        label(ax2, xs[0], ys[0], num(ys[0], 3), color=col, dx=-3 if dy > 0 else 3, dy=dy,
              ha="right" if dy > 0 else "left", va="bottom" if dy > 0 else "top")
    ax2.set_xlim(0.14, 0.50)
    ax2.set_ylim(1.35, 2.05)
    ax2.xaxis.set_major_locator(MultipleLocator(0.1))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax2.set_xlabel("公榜换手率")
    ax2.set_ylabel("公榜 ICIR、压力稳定性")
    clean(ax2)
    save(fig, "fig_turnover_tradeoff_r")


def fig_slot_placebo(D, mode):
    """把时段专用项放到 10:00–14:00 六个时段的 48 组扫描，与 14:30 一步对比。"""
    v = np.array(D["placebo"])
    mu, sd = v.mean(), v.std(ddof=1)
    st = D["steps"]
    fig = new_fig(15.6, 6.2)
    ax = fig.add_subplot()
    bw = 0.00005
    bins = np.floor(v / bw + 1e-9).astype(int)
    cnt = collections.Counter()
    for bi in sorted(bins):
        cnt[bi] += 1
        ax.scatter([(bi + 0.5) * bw], [cnt[bi]], s=26, color=C["light"], edgecolor=C["mid"], lw=0.5, zorder=3)
    hmax = max(cnt.values())
    lo, hi = v.min(), v.max()
    yb = hmax + 1.3
    ax.plot([lo, lo, hi, hi], [yb - 0.4, yb, yb, yb - 0.4], color=C["mid"], lw=0.8)
    ax.text((lo + hi) / 2, yb + 0.35, f"48 组时段副本（10:00–14:00）\n均值 {num(mu, 5)}，标准差 {num(sd, 5)}",
            ha="center", va="bottom", color=C["sub"], linespacing=1.3)
    items = [(st["slot1430"], "14:30 反向调整", C["accent"], hmax + 6.4, True),
             (st["c6"], "尾盘与关注度四项（同换手）", C["navy"], hmax + 2.6, False),
             (st["neutral"], "缓冲前中性化（同换手）", C["navy"], hmax + 6.4, False)]
    for xv, lab, col, h, key in items:
        ax.vlines(xv, 0.3, h, color=col, lw=1.8 if key else 1.0, zorder=2)
        ax.scatter([xv], [h], s=40 if key else 20, color=col, zorder=4)
        if key:
            txt = f"{lab}  {num(xv, 4, True)}\n约为扫描标准差的 {xv / sd:.0f} 倍"
            ax.text(xv - 0.00004, h, txt, ha="right", va="center", color=col, fontweight="semibold",
                    linespacing=1.3)
        else:
            ax.text(xv + 0.00004, h, f"{lab}\n{num(xv, 4, True)}", ha="left", va="center", color=col,
                    linespacing=1.3)
    ax.vlines(0, 0, hmax + 0.9, color=C["sub"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(-0.0011, 0.0037)
    ax.set_ylim(0, hmax + 8.4)
    ax.xaxis.set_major_locator(MultipleLocator(0.0005))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: num(x, 4, x > 1e-12) if abs(x) > 1e-12 else "0"))
    ax.set_xlabel("相对对照的 RankIC 变化（本地 2023–2024）")
    hide_y(ax)
    save(fig, "fig_slot_placebo_r")


def fig_signal_decay(D, mode):
    """增量信号单独使用时的逐年 RankIC。"""
    years = ["2020", "2021", "2022", "2023", "2024"]
    labs = [r[0] for r in D["decay"]]
    M = np.array([[r[3][y] for y in years] for r in D["decay"]])
    fig = new_fig(15.6, 9.6)
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1.25], wspace=0.02)
    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("one", ["#FFFFFF", C["navy"]])
    vmax = 0.095
    ax.imshow(np.clip(M, 0, None), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            ax.text(j, i, num(val, 4), ha="center", va="center",
                    color="white" if val > 0.5 * vmax else C["ink"])
    ax.set_xticks(range(5), [f"{y} 年" for y in years])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labs)), labs)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for b in DECAY_GROUP_BREAKS:
        ax.axhline(b - 0.5, color="white", lw=2.4)
    ax2 = fig.add_subplot(gs[1], sharey=ax)
    ax2.set_xlim(0, 1)
    for i, r in enumerate(D["decay"]):
        v20, v24 = r[3]["2020"], r[3]["2024"]
        if v20 < 0.01 or v24 <= 0:
            s, col, fw = "—", C["sub"], "normal"
        else:
            ch = v24 / v20 - 1
            s = pct(ch)
            col = C["accent"] if ch <= -0.4 else C["ink"]
            fw = "semibold" if ch <= -0.4 else "normal"
        ax2.text(0.5, i, s, ha="center", va="center", color=col, fontweight=fw)
    ax2.set_xticks([0.5], ["2024 较 2020"])
    ax2.xaxis.tick_top()
    ax2.tick_params(length=0, labelleft=False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    for b in DECAY_GROUP_BREAKS:
        ax2.axhline(b - 0.5, color=C["grid"], lw=0.8)
    save(fig, "fig_signal_decay_r")


def fig_public_private(D, mode):
    """1669 五项：本地 2024 年与私榜，均以公榜为 1。"""
    keys = list(KEYS)
    names = ["RankIC", "ICIR", "多空组合夏普比率", "压力稳定性", "换手率（越低越好）"]
    pub = D["public_1669"]
    series = [("本地 2024 年", [D["final_years"]["2024"][k] / pub[k] for k in keys], C["navy"]),
              ("私榜", [PRIVATE_1669[k] / pub[k] for k in keys], C["accent"])]
    fig = new_fig(15.6, 7.4)
    ax = fig.add_subplot()
    bh, step, gsep = 0.40, 0.44, 0.50
    yt, yl = [], []
    y = 0.0
    for gi, nm in enumerate(names):
        ys = []
        for si, (lab, vals, col) in enumerate(series):
            yy = y - si * step
            ax.barh(yy, vals[gi], bh, color=col, zorder=2, label=lab if gi == 0 else None)
            label(ax, vals[gi], yy, num(vals[gi], 3), color="white", weight="semibold" if si == 1 else "normal",
                  dx=-5, dy=0, ha="right", va="center")
            ys.append(yy)
        yt.append((ys[0] + ys[-1]) / 2)
        yl.append(nm)
        y = ys[-1] - step - gsep
    # 公榜基准线画在柱子下面，只在柱间空隙露出
    ax.axvline(1.0, color=C["sub"], lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.text(1.0, 0.30, "公榜 = 1", ha="center", va="bottom", color=C["sub"])
    ax.set_yticks(yt, yl)
    ax.set_ylim(y + step + gsep - 0.30, 0.62)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)
    _legend_top(fig, 2)
    save(fig, "fig_public_private_r")


def fig_milestones_by_year(D, mode):
    """各阶段代表版本的逐年 RankIC：2024 年都是最低。"""
    years = ["2020", "2021", "2022", "2023", "2024"]
    cols = {"2020": "#E3E7EC", "2021": "#C9D0DA", "2022": "#A3AFBF", "2023": C["mid"], "2024": C["accent"]}
    fig = new_fig(15.6, 7.4)
    ax = fig.add_subplot()
    bw = 0.15
    for gi, (lab, yy) in enumerate(D["milestones_by_year"]):
        for j, y in enumerate(years):
            xx = gi + (j - 2) * (bw + 0.01)
            if y in yy:
                ax.bar(xx, yy[y], bw, color=cols[y], zorder=2, label=f"{y} 年" if gi == 1 else None)
                if y == "2024":
                    label(ax, xx, yy[y], num(yy[y], 4), color=C["accent"], weight="semibold", rotation=90)
        missing = [y for y in years if y not in yy]
        if missing:
            xm = np.mean([gi + (years.index(y) - 2) * (bw + 0.01) for y in missing])
            if "2023" in yy and len(yy) == 2:
                ax.text(xm, 0.004, "样本内\n不列", ha="center", va="bottom", color=C["sub"], linespacing=1.3)
            else:
                ax.text(xm, 0.004, "未评测", ha="center", va="bottom", color=C["sub"], rotation=90)
    ax.set_xticks(range(len(D["milestones_by_year"])), [lab for lab, _ in D["milestones_by_year"]])
    ax.set_ylim(0, 0.118)
    ax.set_xlim(-0.55, len(D["milestones_by_year"]) - 0.45)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_ylabel("RankIC")
    clean(ax)
    ax.tick_params(axis="x", length=0)
    h, l_ = ax.get_legend_handles_labels()
    _legend_top(fig, 5, spacing=1.4, handles=h, labels=l_)
    save(fig, "fig_milestones_by_year_r")


def fig_research_process(D, mode):
    """上：研究阶段时间线（首次提交公榜的日期，不写综合得分）；下：每日本地评测任务数。"""
    days = [date.fromisoformat(d) for d, _ in D["daily"]]
    cnts = [c for _, c in D["daily"]]
    d0 = date(2026, 10, 9)
    xd = [(d - d0).days for d in days]
    fig = new_fig(15.6, 8.6)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.78, 1])
    ax = fig.add_subplot(gs[0])
    mx = [(d - d0).days for d, _ in MILESTONES]
    ax.hlines(0, 0.4, 19.6, color=C["light"], lw=1.2, zorder=1)
    # 阶段名位置：(文字高度，ha)。高度为正写在轴线上方，为负写在下方；同侧相邻的阶段错开高度
    place = [
        (1.00, "left"),    # 委托笔数单因子
        (-0.55, "left"),   # 多尺度反转
        (0.45, "left"),    # 同伴残差与中性化
        (-0.55, "center"), # 浅层树组合
        (0.45, "center"),  # 16 个树平均
        (-0.55, "left"),   # 缓冲带与委托规模比
        (1.00, "left"),    # 14:30 收盘竞价惯性
        (0.45, "left"),    # 缓冲前中性化
        (-1.10, "right"),  # 尾盘与关注度四项（最终因子）
    ]
    for (ty, ha), xv, (_, lab) in zip(place, mx, MILESTONES):
        last = xv == mx[-1]
        col = C["accent"] if last else C["navy"]
        ax.plot([xv, xv], [0, ty * 0.92], color=C["mid"], lw=0.7, zorder=1)
        ax.scatter([xv], [0], s=50 if last else 24, color=col, zorder=3)
        ax.text(xv + (-0.1 if ha == "right" else 0.1 if ha == "left" else 0), ty, lab, ha=ha,
                va="bottom" if ty > 0 else "top", color=C["accent"] if last else C["ink"],
                fontweight="semibold" if last else "normal")
    ax.set_ylim(-1.55, 1.62)
    hide_y(ax)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    panel_label(ax, "研究阶段（按首次提交公榜的日期）", y=0.98)

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    k = int(np.argmax(cnts))
    ax2.bar(xd, cnts, 0.7, color=[C["accent"] if i == k else C["navy"] for i in range(len(cnts))], zorder=2)
    for i, (xv, c) in enumerate(zip(xd, cnts)):
        label(ax2, xv, c, f"{c}", color=C["accent"] if i == k else C["ink"],
              weight="semibold" if i == k else "normal", dy=2)
    ax2.text(0.6, 1040, f"本地评测任务合计 {D['n_jobs']} 个", ha="left", va="top", color=C["ink"])
    ax2.set_ylim(0, 1080)
    hide_y(ax2)
    ax2.set_ylabel("每日任务数")
    ticks = list(range(1, 20))
    ax2.set_xticks(ticks, [str((d0 + timedelta(days=t)).day) for t in ticks])
    ax2.set_xlim(0.4, 19.6)
    ax2.tick_params(axis="x", length=0)
    ax2.set_xlabel("2026 年 10 月（日）")
    save(fig, "fig_research_process_r")


def fig_1430_steps(D, mode):
    """14:30 截面 RankIC：依次加入收盘竞价惯性、反向调整项、放量项。"""
    s = mode == "s"
    fig = new_fig(7.7, 5.0) if s else new_fig(15.6, 6.0)
    ax = fig.add_subplot()
    x = np.arange(len(SLOT1430))
    cols = [C["navy"], C["light"], C["light"], C["accent"]]
    bw = 0.62 if s else 0.5
    for xi, (lab, lo, hi), col in zip(x, SLOT1430, cols):
        h = (lo + hi) / 2  # 区间取中值画高，柱上写出区间
        ax.bar(xi, h, bw, color=col, zorder=2)
        txt = num(lo, 3) if hi == lo else f"{num(lo, 3)}～{num(hi, 3)}"
        last = xi == x[-1]
        if s:  # 幻灯片版柱距窄，数值竖排写在柱内
            label(ax, xi, h, txt, color=C["navy"] if col == C["light"] else "white", dy=-5, va="top",
                  rotation=90, weight="semibold" if last else "normal")
        else:
            label(ax, xi, h, txt, color=C["accent"] if last else C["ink"], weight="semibold" if last else "normal")
    lo0 = SLOT1430[0][1]
    lo3, hi3 = SLOT1430[-1][1], SLOT1430[-1][2]
    h0, h3 = lo0, (lo3 + hi3) / 2
    yb = h3 + (0.010 if s else 0.022)
    leg0 = h0 + (0.004 if s else 0.016)
    leg3 = h3 + (0.004 if s else 0.016)
    ax.plot([0, 0, 3, 3], [leg0, yb, yb, leg3], color=C["accent"], lw=1.0)
    ax.text(1.5, yb + 0.002, f"{num(lo3 - lo0, 3, True)}～{num(hi3 - lo0, 3, True)}"
            f"（{pct(lo3 / lo0 - 1)}～{pct(hi3 / lo0 - 1)}）",
            ha="center", va="bottom", color=C["accent"], fontweight="semibold")
    names = (["加入前", "+收盘竞价\n惯性", "+14:30\n反向调整", "+14:30\n放量项"] if s else
             ["加入前", "+收盘竞价惯性", "+14:30 反向调整", "+14:30 放量项"])
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.162 if s else 0.178)
    ax.set_xlim(-0.55, len(x) - 0.45)
    hide_y(ax)
    ax.tick_params(axis="x", length=0)
    save(fig, f"fig_1430_steps_{mode}")


def fig_level_vs_volatility(D, mode):
    """1669 公榜到私榜：RankIC 均值下降，逐截面 RankIC 标准差（= RankIC ÷ ICIR）几乎不变。"""
    s = mode == "s"
    pub, pri, loc = D["public_1669"], PRIVATE_1669, D["final_all"]
    sd = lambda r: r["ic_mean"] / r["ic_ir"]  # noqa: E731
    rows = [("公榜", pub, C["navy"]), ("私榜", pri, C["accent"])]
    if not s:
        rows = [("本地 2023–2024", loc, C["light"])] + rows
    fig = new_fig(7.0, 5.0) if s else new_fig(15.6, 6.0)
    ax = fig.add_subplot()
    groups = ["RankIC 均值", "逐截面 RankIC\n标准差" if s else "逐截面 RankIC 标准差"]
    fns = [lambda r: r["ic_mean"], sd]
    x = np.arange(2)
    bw, gap = (0.30, 0.14) if s else (0.22, 0.07)
    k = len(rows)
    offs = [(i - (k - 1) / 2) * (bw + gap) for i in range(k)]
    for (name, r, col), off in zip(rows, offs):
        vals = [fn(r) for fn in fns]
        ax.bar(x + off, vals, bw, color=col, label=name, zorder=2)
        for xi, v in zip(x, vals):
            pri_ = name == "私榜"
            label(ax, xi + off, v, num(v, 4), color=C["accent"] if pri_ else C["ink"],
                  weight="semibold" if pri_ else "normal")
    for xi, fn in zip(x, fns):
        ch = fn(pri) / fn(pub) - 1
        top = max(fn(r) for _, r, _ in rows)
        label(ax, xi, top, pct(ch) if s else f"私榜较公榜 {pct(ch)}", color=C["accent"], weight="semibold", dy=19)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 0.140)
    ax.set_xlim(-0.6, 1.6)
    hide_y(ax)
    ax.tick_params(axis="x", length=0)
    _legend_top(fig, k, spacing=1.4)
    save(fig, f"fig_level_vs_volatility_{mode}")


# ---------------------------------------------------------------- 打印图中数字

def report_numbers(D):
    print("\n==== 图中数字（供图表清单核对）")
    print(f"任务索引：{D['n_rows']} 行，按 job_id 去重 {D['n_jobs']} 个")
    print(f"树：{D['tree_n']} 棵；按组 " + "、".join(f"{g} {v:.1f}%" for g, v in sorted(D['group_pct'], key=lambda t: -t[1])))
    top = sorted(D["feat_pct"].items(), key=lambda t: -t[1])[:12]
    print("  前 12：" + "、".join(f"{FEAT_SHORT.get(f, FEAT_CN[f])} {v:.1f}%" for f, v in top))
    print("  根节点前 5：" + "、".join(f"{FEAT_CN[f]} {c}" for f, c in D["root"].most_common(5)))
    tl = D["tree_lin"]
    for k in ("ic", "ir", "to"):
        print(f"树 vs 线性 {k}: 线性 {[round(v, 4) for v in tl['lin'][k]]} 树 {[round(v, 4) for v in tl['tree'][k]]} "
              f"差 {[round(b - a, 4) for a, b in zip(tl['lin'][k], tl['tree'][k])]}")
    print("链 A：")
    for lab, v in D["chain_a"]:
        print(f"  {lab:16s} RankIC {v['ic_mean']:.4f} ICIR {v['ic_ir']:.3f} 多空 {v['ls_sharpe']:.2f} "
              f"压力 {v['stress']:.3f} 换手 {v['turnover']:.3f}")
    ca = D["chain_a"]
    print(f"  首尾：RankIC {ca[-1][1]['ic_mean'] - ca[0][1]['ic_mean']:+.4f}，换手 "
          f"{ca[-1][1]['turnover'] - ca[0][1]['turnover']:+.3f}（{ca[-1][1]['turnover'] / ca[0][1]['turnover'] - 1:+.1%}）")
    print(f"  同换手：缓冲前中性化 {ca[8][1]['ic_mean'] - ca[7][1]['ic_mean']:+.4f}；四项 {ca[10][1]['ic_mean'] - ca[9][1]['ic_mean']:+.4f}")
    a, b = D["lin_years"]
    for y in sorted(a):
        print(f"五年 {y}: 基准 {a[y]['ic_mean']:.4f} 全部 {b[y]['ic_mean']:.4f} 差 {b[y]['ic_mean'] - a[y]['ic_mean']:+.4f}"
              f" | ICIR 差 {b[y]['ic_ir'] - a[y]['ic_ir']:+.3f} 换手差 {b[y]['turnover'] - a[y]['turnover']:+.3f}")
    tr = D["trade"]
    for k in ("add", "neu", "c6"):
        rows = tr[k]
        sl = (rows[0]['ic_mean'] - rows[-1]['ic_mean']) / (rows[0]['turnover'] - rows[-1]['turnover']) * 0.1
        print(f"交换 {k}: " + "、".join(f"({r['turnover']:.3f}, {r['ic_mean']:.4f})" for r in rows) + f"；每 0.1 换手 {sl:.4f}")
    print("公榜换手族（换手率，ICIR，压力稳定性）：" + "、".join(f"({t:.4f}, {ir:.4f}, {st:.4f})" for t, ir, st in sorted(D["pub_turn"])))
    v = np.array(D["placebo"])
    print(f"安慰剂 n={len(v)} 均值 {v.mean():.5f} 标准差 {v.std(ddof=1):.5f} 最大 {v.max():+.5f} 最小 {v.min():+.5f}")
    print("  对照步：" + "、".join(f"{k} {x:+.4f}" for k, x in D["steps"].items()) +
          f"；14:30 一步 / 标准差 = {D['steps']['slot1430'] / v.std(ddof=1):.1f}")
    print("单项逐年：")
    for lab, n, st, yy in D["decay"]:
        print(f"  {lab:22s} {st:6s} " + " ".join(f"{yy[y]:.4f}" for y in sorted(yy)) +
              (f"  2024/2020 {yy['2024'] / yy['2020'] - 1:+.1%}" if yy['2020'] > 0.01 else ""))
    pub, y23, y24 = D["public_1669"], D["final_years"]["2023"], D["final_years"]["2024"]
    for k in KEYS:
        print(f"公私 {k}: 本地23 {y23[k]:.4f} 本地24 {y24[k]:.4f} 公榜 {pub[k]:.4f} 私榜 {PRIVATE_1669[k]:.4f} | "
              f"÷公榜 {y23[k] / pub[k]:.3f} {y24[k] / pub[k]:.3f} {PRIVATE_1669[k] / pub[k]:.3f}")
    loc = D["final_all"]
    print(f"逐截面标准差：本地 {loc['ic_mean'] / loc['ic_ir']:.4f} 公榜 {pub['ic_mean'] / pub['ic_ir']:.4f} "
          f"私榜 {PRIVATE_1669['ic_mean'] / PRIVATE_1669['ic_ir']:.4f}；私÷公 均值 {PRIVATE_1669['ic_mean'] / pub['ic_mean'] - 1:+.1%}"
          f" 标准差 {(PRIVATE_1669['ic_mean'] / PRIVATE_1669['ic_ir']) / (pub['ic_mean'] / pub['ic_ir']) - 1:+.1%}")
    for lab, yy in D["milestones_by_year"]:
        print(f"里程碑 {lab.replace(chr(10), '')}: " + " ".join(f"{y}:{yy[y]:.4f}" for y in sorted(yy)) +
              f"  最低年 {min(yy, key=yy.get)}")
    print("每日任务：" + " ".join(f"{d[5:]}:{c}" for d, c in D["daily"]) + f"；合计 {sum(c for _, c in D['daily'])}")
    lo0 = SLOT1430[0][1]
    print(f"14:30：{SLOT1430}；增幅 {SLOT1430[-1][1] - lo0:+.3f}～{SLOT1430[-1][2] - lo0:+.3f}"
          f"（{SLOT1430[-1][1] / lo0 - 1:+.1%}～{SLOT1430[-1][2] / lo0 - 1:+.1%}）")


def main():
    global REPORT_PALETTE, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-palette", choices=["report", "slides"], default="report",
                    help="报告版配色：report＝报告模板（深蓝＋红），slides＝幻灯片模板（深蓝＋橙）")
    ap.add_argument("--out", default=str(OUT), help="输出目录，默认为本脚本所在目录")
    args = ap.parse_args()
    REPORT_PALETTE = args.report_palette
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    D = collect()
    jobs = [
        (fig_tree_importance, "rs"), (fig_tree_vs_linear, "rs"), (fig_increment_chain, "r"),
        (fig_five_years, "rs"), (fig_turnover_tradeoff, "rs"), (fig_slot_placebo, "r"),
        (fig_signal_decay, "r"), (fig_public_private, "r"), (fig_milestones_by_year, "r"),
        (fig_research_process, "r"), (fig_1430_steps, "rs"), (fig_level_vs_volatility, "rs"),
    ]
    for fn, modes in jobs:
        for m in modes:
            use_palette(m)
            fn(D, m)
    report_numbers(D)


if __name__ == "__main__":
    main()
