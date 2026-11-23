#!/usr/bin/env python3
"""端到端赛道决赛材料的数据图（报告版 *_r，幻灯片版 *_s）。

只读仓库数据，图输出到本脚本所在目录（PDF 矢量 + 300 dpi PNG）。
运行：python3 make_figures.py                       # 全部重画并打印图中用到的全部数字
      python3 make_figures.py --report-palette report  # 报告版改用报告模板色（深蓝 + 红色强调）
      python3 make_figures.py --only slot_ic        # 只画某一张

依赖：python3、matplotlib 3.9、numpy。字体为微软雅黑（Office 安装包自带的 msyh.ttc／msyhbd.ttc，
用 font_manager.addfont 直接注册，不复制进仓库）。

数据来源（全部只读）：
  e2e/过程记录/analysis_macstudio/cmp_*.csv            留出验证设定（2020–2022 训练）下各集成的 2023、2024 年五项
  e2e/过程记录/analysis_macstudio/metrics_cmp_GRU4_*   最终结构的逐截面时段 RankIC
  e2e/过程记录/analysis_macstudio/metrics_half_ens_*   最终结构的四个半年五项
  e2e/过程记录/runs_macstudio/R15c_P13_ep8/metrics.json 单模型 8 轮训练的验证曲线
  e2e/提交/E21_*/transformer_model.json                冻结权重里梯度提升树子模型的 800 棵树（只读，统计分裂）
以下数字仓库里没有可直接读取的表格，按已核对文档写入常量（出处见各常量注释）：
  最终模型公榜／私榜五项；各版本的公榜五项；单变量线上验证的公榜五项；本地与公榜 21 组配对差。
图中不写排行榜综合得分，只画五项指标。
"""
import argparse
import collections
import csv
import json
import logging
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtrans  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path("/Users/user/Desktop/quant-claude")
AM = ROOT / "e2e/过程记录/analysis_macstudio"
EPOCHS = ROOT / "e2e/过程记录/runs_macstudio/R15c_P13_ep8/metrics.json"
FROZEN = ROOT / "e2e/提交/E21_E18g加两个GRU成员_2020-24/transformer_model.json"
YAHEI = Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")
OUT = Path(__file__).resolve().parent

MINUS = "−"
CM = 1 / 2.54
KEYS = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")
NAMES5 = ("RankIC", "ICIR", "多空组合夏普比率", "压力稳定性", "换手率")


# ---------------------------------------------------------------- 字体与配色

def setup_fonts():
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    for f in ("msyh.ttc", "msyhbd.ttc"):
        fm.fontManager.addfont(str(YAHEI / f))
    plt.rcParams.update({
        "font.family": "Microsoft YaHei",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.2,
    })


PALETTES = {
    # 幻灯片模板 slides_preamble.tex：snavy / sorange / sline / stint / sgrey / sink
    "slides": dict(navy="#0F2E5A", accent="#E8772E", accent2="#F3B78C", light="#C9D3E0", mid="#7C8FAD",
                   tint="#E8EFF8", ink="#1E2329", sub="#5E6672", grid="#E4E9F0"),
    # 报告模板 report_preamble.tex：rnavy / raccent / rline / rgrey
    "report": dict(navy="#1B365D", accent="#C0392B", accent2="#E2A39B", light="#BFC5CC", mid="#7A828C",
                   tint="#EEF2F7", ink="#1E2329", sub="#6B737D", grid="#E6E9ED"),
}
REPORT_PALETTE = "slides"  # 任务要求两版都用橙色强调；如需报告模板色，加 --report-palette report
C = dict(PALETTES["slides"])


def use_palette(mode):
    C.clear()
    C.update(PALETTES["slides" if mode == "s" else REPORT_PALETTE])
    for k in ("text.color", "axes.labelcolor", "xtick.labelcolor", "ytick.labelcolor"):
        plt.rcParams[k] = C["ink"]
    for k in ("xtick.color", "ytick.color", "axes.edgecolor"):
        plt.rcParams[k] = C["sub"]


# ---------------------------------------------------------------- 小工具

def num(v, d, sign=False):
    s = f"{abs(v):.{d}f}"
    if v < 0 and float(s) != 0:
        return MINUS + s
    return ("+" + s) if (sign and float(s) != 0) else s


def num_half_up(v, d):
    """按十进制逢五进一取 d 位（1.8995 → 1.900），与报告成绩表的取位一致；只用于非负原值。"""
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def pct(v, d=0, sign=True):
    """v 为比例（−0.223 → “−22%”）。"""
    s = f"{abs(v) * 100:.{d}f}%"
    if v < 0 and float(s[:-1]) != 0:
        return MINUS + s
    return ("+" + s) if (sign and float(s[:-1]) != 0) else s


def new_fig(w_cm, h_cm, **kw):
    return plt.figure(figsize=(w_cm * CM, h_cm * CM), layout="constrained", **kw)


def clean(ax, left=True, grid=True, bottom=True, xgrid=False):
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if grid:
        (ax.xaxis if xgrid else ax.yaxis).grid(True, color=C["grid"], lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    if not left:
        ax.tick_params(axis="y", length=0)


def panel_label(ax, text, x=0.0, align_ticks=False, ha="left", color=None):
    """面板说明（坐标轴含义），不是图标题。用一个空标题在版面上预留一行高度，说明文字本身不参与版面计算。
    align_ticks=True 时左端对齐到该坐标轴 y 刻度标签的左端（横向条形图用），在 save() 里定位。"""
    ax.set_title(" ", loc="left", pad=3)
    if align_ticks:
        tr = mtrans.blended_transform_factory(ax.figure.transFigure, ax.transAxes)
        xx = 0.006
    else:
        tr, xx = ax.transAxes, x
    t = ax.annotate(text, (xx, 1.0), xycoords=tr, xytext=(0, 3), textcoords="offset points", ha=ha, va="bottom",
                    color=color or C["sub"], annotation_clip=False)
    t.set_in_layout(False)
    if align_ticks:
        ax.figure._fx_align = getattr(ax.figure, "_fx_align", []) + [(ax, t)]
    return t


def label(ax, x, y, text, color=None, weight="normal", dy=3, dx=0, ha="center", va="bottom", rotation=0, bg=False):
    kw = dict(bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.92)) if bg else {}
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                       color=color or C["ink"], fontweight=weight, rotation=rotation, annotation_clip=False,
                       zorder=7, **kw)


def save(fig, name):
    for ax in fig.axes:  # 数据坐标里的文字不参与版面计算，避免压缩绘图区
        for t in ax.texts:
            if t.get_transform() != ax.transAxes:
                t.set_in_layout(False)
    pending = getattr(fig, "_fx_align", [])
    if pending:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for ax, t in pending:
            x0 = min(lb.get_window_extent(r).x0 for lb in ax.get_yticklabels() if lb.get_text())
            t.xy = (fig.transFigure.inverted().transform((x0, 0))[0], 1.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    w, h = fig.get_size_inches() / CM
    plt.close(fig)
    print(f"  -> {name}.pdf / .png  ({w:.1f} × {h:.1f} cm)")


# ---------------------------------------------------------------- 数据

def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


_CMP = {}


def pick(fname, run, year):
    if fname not in _CMP:
        _CMP[fname] = read_csv(AM / fname)
    for r in _CMP[fname]:
        if r["run"] == run and r["year"] == str(year):
            return {k: float(r[k]) for k in KEYS}
    raise KeyError((fname, run, year))


# 留出验证设定（2020–2022 训练，2023 验证、2024 测试）下的集成，与最终模型同配方
ENS = {
    "tf4": ("cmp_R43.csv", "t4"),             # 4 个 Transformer，不加多空组合换手正则项
    "tf4_tree": ("cmp_R43.csv", "t4g"),       # 上一行 + 梯度提升树子模型
    "tf4_reg": ("cmp_R46.csv", "t15x4"),      # 4 个 Transformer，换手正则项 0.1（换了训练机）
    "tf4_reg_tree": ("cmp_GRU.csv", "T4G"),   # 上一行 + 梯度提升树子模型（= 0 个 GRU）
    "gru1": ("cmp_GRU.csv", "T4GR1"),         # + 1 个 GRU
    "final": ("cmp_GRU4.csv", "E21a"),        # + 2 个 GRU = 最终结构
    "gru4": ("cmp_GRU4.csv", "E22a"),         # + 4 个 GRU（未提交）
}


def ens(key, year):
    return pick(*ENS[key], year)


# 最终模型的公榜、私榜五项（平台原值；事实清单 A02、A03，〔00〕§1.2）
PUBLIC = dict(ic_mean=0.1143, ic_ir=2.0186, ls_sharpe=56.3285, stress=1.8995, turnover=0.5081)
PRIVATE = dict(ic_mean=0.0888, ic_ir=1.5635, ls_sharpe=44.9779, stress=1.4601, turnover=0.5002)

# 各版本首次提交公榜时的五项原值（平台原值，不随其他队提交漂移）：日期、改动、五项（KEYS 顺序）、类别。
# 出处：文档/决赛材料/端到端赛道/01_开发历程全记录.md 各版本“公榜”行（E05、E07、E09、E10、E13、E15、E17b、E21），
# E16、E18g 取自 03_信号质量_统计检验与稳定性.md §3.2。从官方标签核实后的第一个版本开始。
PUBLIC_PATH = [
    ("10-11", "官方标签，修复 09:30 截面", (0.0898, 1.4816, 42.42, 1.3957, 0.5846), "n"),
    ("10-12", "目标中性化，09:30 输出 0", (0.0953, 1.5463, 49.75, 1.5552, 0.5635), "n"),
    ("10-14", "多周期辅助目标", (0.0957, 1.7504, 48.66, 1.6662, 0.5541), "n"),
    ("10-16", "全天 5 分钟片段（退步）", (0.0880, 1.6579, 45.31, 1.6558, 0.4590), "down"),
    ("10-17", "4 个随机种子集成", (0.1040, 1.8772, 53.22, 1.7066, 0.5562), "n"),
    ("10-21", "秩高斯目标", (0.1114, 2.0464, 56.05, 1.8939, 0.5700), "n"),
    ("10-24", "6 轮训练与多任务头", (0.1147, 2.0987, 56.88, 1.9397, 0.5610), "n"),
    ("10-26", "加入梯度提升树子模型", (0.1162, 2.1004, 56.24, 1.9468, 0.5640), "n"),
    ("10-27", "加入多空组合换手正则项", (0.1142, 2.0266, 56.14, 1.8977, 0.5143), "n"),
    ("10-28", "加入 2 个 GRU（最终模型）", (0.1143, 2.0186, 56.33, 1.8995, 0.5081), "final"),
]

# 多空组合换手正则项强度 λ 的单变量线上验证（只训练到 2022 年、种子 42）：RankIC、ICIR、换手率、多空组合夏普比率、压力稳定性
# 出处：文档/决赛材料/端到端赛道/03_信号质量_统计检验与稳定性.md §7.4（事实清单 D2）。五项为平台原值，四档可直接比较
REG_PUBLIC = [(0.0, 0.1085, 2.0202, 0.5507, 53.11, 1.859), (0.1, 0.1070, 1.9897, 0.5148, 52.73, 1.863),
              (0.3, 0.0989, 1.7857, 0.4273, 47.90, 1.724), (1.0, 0.0896, 1.6070, 0.3472, 43.02, 1.627)]
# 同一强度序列的本地单模型（留出验证设定，种子 42）：训练机甲 λ = 0、0.1、0.3、1.0；训练机乙 λ = 0、0.1、0.15、0.2
REG_LOCAL_A = [(0.0, "cmp_LG.csv", "R22a"), (0.1, "cmp_LG.csv", "L1"), (0.3, "cmp_LG.csv", "L3"),
               (1.0, "cmp_LG.csv", "L10")]
REG_LOCAL_B = [(0.0, "cmp_R41.csv", "X22a"), (0.1, "cmp_R41.csv", "L01"), (0.15, "cmp_R41.csv", "L015"),
               (0.2, "cmp_R41.csv", "L02")]

# 本地两年平均差 vs 公榜差的 21 组配对（〔端03〕§5.5 配对表；ΔRankIC 本地、公榜，Δ换手率 本地、公榜）
PAIRS = [
    ("盘口窗口通道", +0.0015, +0.0004, -0.0070, -0.0030),
    ("训练 2 轮", -0.0020, -0.0030, -0.0365, -0.0361),
    ("多周期辅助目标", +0.0006, 0.0000, -0.0430, -0.0464),
    ("全局换手惩罚 0.05", -0.0095, -0.0061, -0.1100, -0.0995),
    ("融合头", +0.0031, +0.0013, +0.0265, -0.0031),
    ("全天 5 分钟片段", +0.0006, -0.0033, 0.0000, -0.0117),
    ("融合头＋片段＋辅助 1.0", +0.0011, -0.0012, -0.0285, -0.0331),
    ("权重滑动平均", 0.0000, 0.0000, -0.0200, -0.0160),
    ("当前时段输入", +0.0038, +0.0025, +0.0205, +0.0371),
    ("时段调制＋日线锚", -0.0012, +0.0004, -0.0350, -0.0400),
    ("秩高斯目标＋4 轮", +0.0045, +0.0068, +0.0195, +0.0279),
    ("6 轮＋多任务头", +0.0028, +0.0020, -0.0215, -0.0302),
    ("换手正则项 0.1", -0.0028, -0.0015, -0.0370, -0.0359),
    ("换手正则项 0.3", -0.0090, -0.0096, -0.1335, -0.1234),
    ("换手正则项 1.0", -0.0199, -0.0189, -0.2145, -0.2035),
    ("加树（单模型）", +0.0028, +0.0032, +0.0130, +0.0112),
    ("正则项 0.15 对 0.1", -0.0015, -0.0032, -0.0308, -0.0254),
    ("中间时刻样本", -0.0002, -0.0007, +0.0047, +0.0030),
    ("加树（全量）", +0.0016, +0.0015, +0.0046, +0.0030),
    ("换手正则项 0.1（全量）", -0.0040, -0.0020, -0.0461, -0.0497),
    ("加 2 个 GRU", 0.0000, +0.0001, -0.0080, -0.0062),
]
SIG = 0.002  # 本地 |ΔRankIC| 的分界（约 1.5 个单种子标准差，〔端03〕§5.5）

HALVES = [("2023 上半年", "2023-01_2023-01_2023-06"), ("2023 下半年", "2023-07_2023-07_2023-12"),
          ("2024 上半年", "2024-01_2024-01_2024-06"), ("2024 下半年", "2024-07_2024-07_2024-12")]
SLOTS = ["s1000", "s1030", "s1100", "s1130", "s1330", "s1400", "s1430"]
SLOT_NAMES = ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]


def slot_ic(year):
    r = read_csv(AM / f"metrics_cmp_GRU4_E21a_{year}_{year}-01_{year}-12.csv")[0]
    return [float(r[k]) for k in SLOTS], {k: float(r[k]) for k in KEYS}


def halfyear(ens_name="T4G1R2"):
    out = []
    for lab, h in HALVES:
        r = read_csv(AM / f"metrics_half_ens_{ens_name}_{h}.csv")[0]
        out.append((lab, {k: float(r[k]) for k in KEYS}))
    return out


# 梯度提升树子模型的输入拼接顺序（训练脚本 _gb_flatten）：1 分钟 11 种池化 × 25 字段 → 日内 9 种 × 48 通道
# → 日线 6 种 × 48 通道 → 同伴向量 60。1 分钟字段顺序见训练脚本 FIELDS["full"]；48 个窗口通道按训练脚本
# 构造窗口张量的拼接顺序：价量 4、K 线形态 3、价格位置（窗口高低）2、尾盘习惯 1、窗口盘口 4+4+2、收盘竞价 2，
# 然后是每个 token 相同的个股状态 5、市场离散度 2、价格位置（当日至今与 3 日区间）4、领先滞后 2、星期 5、当前时段 8。
F1 = ["open", "high", "low", "close", "volume", "amount", "deal_number",
      "bid_price1", "bid_price2", "bid_price3", "ask_price1", "ask_price2", "ask_price3",
      "bid_volume1", "bid_volume2", "bid_volume3", "ask_volume1", "ask_volume2", "ask_volume3",
      "bid_num_orders1", "bid_num_orders2", "bid_num_orders3", "ask_num_orders1", "ask_num_orders2", "ask_num_orders3"]
WIN = [(g, i) for g, n in [("base", 4), ("shape", 3), ("pos", 2), ("cls", 1), ("book", 4), ("book2", 4),
                           ("book3", 2), ("auc", 2), ("state", 5), ("mkt", 2), ("pos", 4), ("lead", 2),
                           ("dow", 5), ("tod", 8)]
       for i in range(n)]
assert len(WIN) == 48
GROUP_NAME = {
    "f_vol": "1 分钟量类字段", "base": "窗口价量 4 项", "f_px": "1 分钟价格类字段", "pos": "日内价格位置",
    "peer": "同伴画像向量", "shape": "K 线形态", "cls": "尾盘习惯", "auc": "收盘竞价", "state": "个股状态",
    "lead": "领先滞后", "mkt": "市场截面离散度", "tod": "当前时段", "dow": "星期",
}
# 〔端02〕§7.6 记录的深度加权份额（%），脚本重算后逐项核对
GROUP_DOC = {"f_vol": 20.0, "base": 16.9, "f_px": 13.8, "pos": 12.1, "peer": 11.7, "shape": 7.7, "cls": 4.8,
             "auc": 4.5, "state": 3.7, "lead": 2.8, "mkt": 1.2, "tod": 0.6, "dow": 0.3}
ROOT_NAME = {
    276: "当前窗口 VWAP 相对现价", 663: "日内 18 个窗口上影线占比均值", 277: "当前窗口相对成交额",
    614: "日内倒数第 8 个窗口的收益", 711: "前一交易日上影线占比", 13: "最后一根 1 分钟线的买一量",
    16: "最后一根 1 分钟线的卖一量", 951: "20 日上影线占比均值",
}


def feat_group(j):
    if j < 275:
        return "f_px" if ("price" in F1[j % 25] or F1[j % 25] in ("open", "high", "low", "close")) else "f_vol"
    if j < 275 + 432:
        return WIN[(j - 275) % 48][0]
    if j < 275 + 432 + 288:
        return WIN[(j - 707) % 48][0]
    return "peer"


def tree_stats():
    d = json.load(open(FROZEN, encoding="utf-8"))
    trees = d["gbdt"]["models"][0]
    w = collections.Counter()
    root = collections.Counter()
    nsplit = 0
    for t in trees:
        for i, f in enumerate(t[0]):
            if f < 0:
                continue
            nsplit += 1
            w[feat_group(f)] += 2.0 ** (-int(math.floor(math.log2(i + 1))))
            if i == 0:
                root[f] += 1
    s = sum(w.values())
    share = {g: v / s * 100 for g, v in w.items()}
    return len(trees), nsplit, share, root


# ---------------------------------------------------------------- 图 1：各版本的公榜五项

def fig_public_path(mode):
    s = mode == "s"
    fig = new_fig(*((14.0, 6.4) if s else (15.6, 7.4)))
    axs = fig.subplots(1, len(KEYS), sharey=True, gridspec_kw=dict(wspace=0.10))
    n = len(PUBLIC_PATH)
    ys = np.arange(n)[::-1]
    heads = ("RankIC", "ICIR", "多空组合\n夏普比率", "压力稳定性", "换手率\n（越低越好）")
    digs = (4, 3, 2, 3, 3)
    ticks = ((0.09, 0.11), (1.6, 2.0), (45, 55), (1.4, 1.8), (0.45, 0.55))
    tick_fmt = ("{:.2f}", "{:.1f}", "{:.0f}", "{:.1f}", "{:.2f}")
    pads = ((0.14, 0.14),) * 4 + ((0.14, 0.45),)  # 换手率一栏右侧留出最终模型数值的位置
    for j, ax in enumerate(axs):
        v = np.array([r[2][j] for r in PUBLIC_PATH])
        lo, hi = v.min(), v.max()
        ax.set_xlim(lo - (hi - lo) * pads[j][0], hi + (hi - lo) * pads[j][1])
        ax.axhspan(-0.5, 0.5, color=C["tint"], lw=0, zorder=0)  # 最终模型所在的一行
        ax.plot(v, ys, color=C["light"], lw=1.0, zorder=2)
        for y, val, (_, _, _, kind) in zip(ys, v, PUBLIC_PATH):
            col = {"final": C["accent"], "down": C["mid"], "n": C["navy"]}[kind]
            ax.scatter(val, y, s=34 if kind == "final" else 20, color=col, edgecolor="white", lw=0.6, zorder=4)
        # 只标第一个与最后一个版本的数值，数值放在点的内侧
        for idx in (0, n - 1):
            val = v[idx]
            inner_left = (val - lo) / (hi - lo) > 0.5
            fin = PUBLIC_PATH[idx][3] == "final"
            label(ax, val, ys[idx], num_half_up(val, digs[j]), color=C["accent"] if fin else C["ink"],
                  weight="bold" if fin else "normal", dx=-5 if inner_left else 5, dy=0,
                  ha="right" if inner_left else "left", va="center", bg=True)
        ax.set_xticks(ticks[j], [tick_fmt[j].format(t) for t in ticks[j]])
        ax.xaxis.grid(True, color=C["grid"], lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(" \n ", loc="left", pad=3)  # 为两行的栏目说明预留高度
        t = ax.annotate(heads[j], (0.5, 1.0), xycoords=ax.transAxes, xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", color=C["sub"], annotation_clip=False, linespacing=1.1)
        t.set_in_layout(False)
    ax0 = axs[0]
    ax0.set_yticks(ys, [f"{d}   {lab}" for d, lab, _, _ in PUBLIC_PATH])
    for t, (_, _, _, kind) in zip(ax0.get_yticklabels(), PUBLIC_PATH):
        if kind == "final":
            t.set_color(C["accent"])
            t.set_fontweight("bold")
        elif kind == "down":
            t.set_color(C["sub"])
    # 刻度标签左对齐，使日期与改动说明各成一列
    fig.canvas.draw()
    wmax = max(t.get_window_extent(fig.canvas.get_renderer()).width for t in ax0.get_yticklabels())
    ax0.tick_params(axis="y", pad=wmax * 72 / fig.dpi + 4)
    for t in ax0.get_yticklabels():
        t.set_ha("left")
    ax0.set_ylim(-0.6, n - 0.4)
    print("[公榜五项轨迹] RankIC／ICIR／多空组合夏普比率／压力稳定性／换手率")
    for d, lab, vals, _ in PUBLIC_PATH:
        print(f"   {d} {lab}: " + "／".join(f"{x:g}" for x in vals))
    save(fig, f"fig_public_path_{mode}")


# ---------------------------------------------------------------- 图 2：留出验证模型的增量链

def chain_deltas():
    steps = [("加多空组合换手正则项\n（含训练机差异）", "tf4", "tf4_reg"),
             ("加梯度提升树子模型", "tf4_reg", "tf4_reg_tree"),
             ("加 2 个 GRU 子模型\n（最终结构）", "tf4_reg_tree", "final")]
    side = ("对照：不加正则项时\n加梯度提升树子模型", "tf4", "tf4_tree")
    out = []
    for lab, a, b in steps + [side]:
        dd = {}
        for y in (2023, 2024):
            A, B = ens(a, y), ens(b, y)
            dd[y] = {k: B[k] - A[k] for k in KEYS}
        out.append((lab, dd))
    return out[:3], out[3]


def fig_increment_chain(mode):
    s = mode == "s"
    steps, side = chain_deltas()
    groups = steps if s else steps + [side]
    fig = new_fig(*((14.0, 4.8) if s else (15.6, 6.6)))
    axs = fig.subplots(1, 2, sharey=True, gridspec_kw=dict(wspace=0.05))
    ys = np.arange(len(groups))[::-1].astype(float)
    if not s:
        ys[-1] -= 0.25  # 对照组与主链隔开
    bh = 0.36
    spec = ((axs[0], "ic_mean", 4, "RankIC 变化", (-0.0102, 0.0072 if s else 0.0062)),
            (axs[1], "turnover", 3, "换手率变化（越低越好）", (-0.097, 0.050 if s else 0.064)))
    for ax, key, dig, head, lim in spec:
        for y, (lab, dd) in zip(ys, groups):
            final = "GRU" in lab
            cols = (C["accent"], C["accent2"]) if final else (C["navy"], C["mid"])
            for j, yr in enumerate((2023, 2024)):
                v = dd[yr][key]
                yy = y + (0.5 - j) * bh
                ax.barh(yy, v, height=bh * 0.9, color=cols[j], zorder=3)
                neg = v < 0 or (abs(v) < 1e-12)
                ax.annotate(num(v, dig, sign=True), (v, yy), xytext=(-3 if neg else 3, 0), textcoords="offset points",
                            ha="right" if neg else "left", va="center", annotation_clip=False, zorder=7,
                            color=C["accent"] if final else C["ink"], fontweight="bold" if final else "normal")
        ax.axvline(0, color=C["sub"], lw=0.8, zorder=4)
        if not s:
            ax.axhline((ys[-2] + ys[-1]) / 2, color=C["light"], lw=0.8, ls=(0, (3, 2)))
        ax.set_xlim(*lim)
        ax.set_xticks([])
        clean(ax, left=False, grid=False, bottom=False)
        panel_label(ax, head)
    axs[0].set_yticks(ys, [g[0] for g in groups], linespacing=1.15)
    axs[0].set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    for t, (lab, _) in zip(axs[0].get_yticklabels(), groups):
        if "GRU" in lab:
            t.set_color(C["accent"])
            t.set_fontweight("bold")
        if lab.startswith("对照"):
            t.set_color(C["sub"])
    handles = [Patch(color=C["navy"], label="2023 年（验证年）"), Patch(color=C["mid"], label="2024 年（测试年）")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    for lab, dd in steps + [side]:
        print(f"[增量链] {lab.replace(chr(10), '')}: ΔRankIC {dd[2023]['ic_mean']:+.4f}／{dd[2024]['ic_mean']:+.4f}  "
              f"Δ换手率 {dd[2023]['turnover']:+.4f}／{dd[2024]['turnover']:+.4f}  "
              f"ΔICIR {dd[2023]['ic_ir']:+.3f}／{dd[2024]['ic_ir']:+.3f}")
    for k in ENS:
        a, b = ens(k, 2023), ens(k, 2024)
        print(f"   {k:13s} 2023 " + "／".join(f"{a[x]:.4f}" for x in KEYS) + "   2024 " + "／".join(f"{b[x]:.4f}" for x in KEYS))
    save(fig, f"fig_increment_chain_{mode}")


# ---------------------------------------------------------------- 图 3：多空组合换手正则项的强度

def draw_reg_public(ax, s):
    """柱高为公榜换手率，柱下小表为同一提交的 RankIC、ICIR、压力稳定性（都是平台原值）。"""
    xs = np.arange(len(REG_PUBLIC))
    cols = [C["navy"], C["accent"], C["light"], C["light"]]
    rows = [("正则强度 λ", ["0", "0.1", "0.3", "1.0"]),
            ("RankIC", [num(r[1], 4) for r in REG_PUBLIC]),
            ("ICIR", [num(r[2], 3) for r in REG_PUBLIC]),
            ("压力稳定性", [num(r[5], 3) for r in REG_PUBLIC])]
    dy, y0 = 0.108, -0.076
    ybot = y0 - (len(rows) - 1) * dy - 0.063
    ax.add_patch(plt.Rectangle((0.56, ybot + 0.013), 0.88, -ybot - 0.026, color=C["tint"], lw=0, zorder=1))
    for x, (lam, ic, ir, to, ls, st), col in zip(xs, REG_PUBLIC, cols):
        ax.bar(x, to, width=0.62, color=col, zorder=3)
        label(ax, x, to, num(to, 3), color=C["accent"] if lam == 0.1 else C["ink"],
              weight="bold" if lam == 0.1 else "normal", dy=2)
    ax.plot([-0.45, len(xs) - 0.55], [0, 0], color=C["sub"], lw=0.8, zorder=4)
    for i, (head, vals) in enumerate(rows):
        y = y0 - i * dy
        ax.text(-0.52, y, head, ha="right", va="center", color=C["sub"])
        for x, v in zip(xs, vals):
            sel = x == 1
            ax.text(x, y, v, ha="center", va="center", color=C["accent"] if sel else C["ink"],
                    fontweight="bold" if sel else "normal", zorder=5)
    # 关键差值：λ 0 → 0.1 换手率 −0.036
    t0, t1 = REG_PUBLIC[0][3], REG_PUBLIC[1][3]
    ytop = max(t0, t1) + 0.165
    ax.plot([0, 0, 1, 1], [t0 + 0.13, ytop, ytop, t1 + 0.13], color=C["accent"], lw=0.9, zorder=5)
    ax.text(0.5, ytop + 0.01, num(t1 - t0, 3, sign=True), ha="center", va="bottom", color=C["accent"],
            fontweight="bold")
    ax.set_xlim(-2.0, len(xs) - 0.45)
    ax.set_ylim(ybot, ytop + 0.10)
    ax.set_xticks([])
    ax.set_yticks([])
    clean(ax, left=False, grid=False, bottom=False)
    panel_label(ax, "公榜单变量线上验证：柱高为换手率")


def fig_turnover_reg(mode):
    s = mode == "s"
    if s:
        fig = new_fig(8.2, 5.4)
        ax = fig.add_subplot()
        draw_reg_public(ax, s)
    else:
        fig = new_fig(15.6, 6.6)
        axl, axr = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.0, 1.05], wspace=0.06))
        # 左：本地换手率–ICIR 交换曲线（单模型，留出验证设定）
        for rows, mach, col, lw in ((REG_LOCAL_B, "乙", C["mid"], 1.0), (REG_LOCAL_A, "甲", C["navy"], 1.4)):
            for y, ls in ((2023, "-"), (2024, (0, (3, 2)))):
                pts = [(lam, pick(f, r, y)) for lam, f, r in rows]
                axl.plot([p["turnover"] for _, p in pts], [p["ic_ir"] for _, p in pts], color=col, lw=lw, ls=ls,
                         zorder=3)
                for lam, p in pts:
                    sel = lam == 0.1
                    axl.scatter(p["turnover"], p["ic_ir"], s=30 if sel else 14, zorder=5,
                                color=C["accent"] if sel else col, edgecolor="white", lw=0.6)
                    if y == 2023 and mach == "甲":
                        label(axl, p["turnover"], p["ic_ir"], "1.0" if lam == 1.0 else f"{lam:g}",
                              color=C["accent"] if sel else col,
                              weight="bold" if sel else "normal", dy=5, dx=-2 if lam == 0.1 else 0)
                    if y == 2023 and mach == "乙" and lam in (0.15, 0.2):
                        label(axl, p["turnover"], p["ic_ir"], f"{lam:g}", color=col, dy=-5, va="top",
                              ha="left" if lam == 0.15 else "right", dx=2 if lam == 0.15 else -2)
        axl.set_xlabel("本地换手率")
        axl.set_ylabel("本地 ICIR")
        axl.set_xlim(0.33, 0.60)
        axl.set_ylim(1.2, 2.25)
        axl.set_xticks([0.35, 0.40, 0.45, 0.50, 0.55, 0.60], ["0.35", "0.40", "0.45", "0.50", "0.55", "0.60"])
        axl.set_yticks([1.2, 1.4, 1.6, 1.8, 2.0, 2.2], ["1.2", "1.4", "1.6", "1.8", "2.0", "2.2"])
        clean(axl)
        axl.text(0.336, 2.03, "2023 年", color=C["sub"], ha="left", va="center")
        axl.text(0.336, 1.47, "2024 年", color=C["sub"], ha="left", va="center")
        handles = [Line2D([], [], color=C["navy"], lw=1.4, label="训练机甲"),
                   Line2D([], [], color=C["mid"], lw=1.0, label="训练机乙")]
        axl.legend(handles=handles, loc="lower right")
        panel_label(axl, "单模型本地验证：点旁数字为正则强度 λ")
        draw_reg_public(axr, s)
    b = REG_PUBLIC[0]
    for lam, ic, ir, to, ls, st in REG_PUBLIC:
        print(f"[换手正则] 公榜 λ={lam:g}: RankIC {ic:.4f} ICIR {ir:.4f} 换手率 {to:.4f} 多空 {ls:.2f} 压力 {st:.3f}"
              f"  Δ(对 λ=0) 换手 {to - b[3]:+.4f} ICIR {ir - b[2]:+.4f} RankIC {ic - b[1]:+.4f}"
              f" 多空 {ls - b[4]:+.2f} 压力 {st - b[5]:+.3f}")
    for rows, mach in ((REG_LOCAL_A, "甲"), (REG_LOCAL_B, "乙")):
        for y in (2023, 2024):
            print(f"   本地 训练机{mach} {y}: " + "  ".join(
                f"λ={lam:g} 换手 {pick(f, r, y)['turnover']:.3f} ICIR {pick(f, r, y)['ic_ir']:.3f}" for lam, f, r in rows))
    save(fig, f"fig_turnover_reg_{mode}")


# ---------------------------------------------------------------- 图 4：GRU 子模型个数

def fig_gru_count(mode):
    s = mode == "s"
    keys = [(0, "tf4_reg_tree"), (1, "gru1"), (2, "final"), (4, "gru4")]
    fig = new_fig(*((8.2, 5.2) if s else (15.6, 6.2)))
    ax = fig.add_subplot()
    xs = [k for k, _ in keys]
    ax.axvspan(1.62, 2.38, color=C["tint"], zorder=0, lw=0)
    ax.text(2.0, 0.025, "最终结构", transform=mtrans.blended_transform_factory(ax.transData, ax.transAxes),
            ha="center", va="bottom", color=C["accent"], fontweight="bold")
    rel = {}
    for metric, col in (("ic_ir", C["navy"]), ("turnover", C["accent"])):
        for y, ls in ((2023, "-"), (2024, (0, (3, 2)))):
            v = [ens(k, y)[metric] for _, k in keys]
            r = [(x / v[0] - 1) * 100 for x in v]
            rel[(metric, y)] = (v, r)
            ax.plot(xs, r, color=col, ls=ls, lw=1.6 if metric == "turnover" else 1.3, zorder=3)
            ax.scatter(xs, r, s=[12, 12, 30, 12], color=col, zorder=4, edgecolor="white", lw=0.6)
    ax.axhline(0, color=C["sub"], lw=0.7, zorder=2)

    def pair(metric, i):
        a, b = rel[(metric, 2023)][1][i], rel[(metric, 2024)][1][i]
        return a, b, f"{num(a, 1, sign=True)}%／{num(b, 1, sign=True)}%"

    a, b, t = pair("turnover", 2)
    label(ax, 2, min(a, b), t, color=C["accent"], weight="bold", dy=-7, va="top", bg=True)
    a, b, t = pair("ic_ir", 2)
    label(ax, 2, max(a, b), t, color=C["navy"], weight="bold", dy=7, va="bottom", bg=True)
    for metric, col, name in (("turnover", C["accent"], "换手率"), ("ic_ir", C["navy"], "ICIR")):
        a, b, t = pair(metric, 3)
        ax.text(4.14, (a + b) / 2, name if s else f"{name}\n{t}", color=col, ha="left", va="center",
                linespacing=1.2, fontweight="bold" if s else "normal")
    ax.set_xticks(xs, [f"{k} 个" for k in xs])
    ax.set_xlim(-0.3, 5.0 if s else 5.3)
    ax.set_ylim(-3.2, 0.95)
    ax.set_yticks([-3, -2, -1, 0], [f"{MINUS}3%", f"{MINUS}2%", f"{MINUS}1%", "0"])
    ax.set_xlabel("集成中 GRU 子模型的个数")
    ax.spines["bottom"].set_bounds(-0.3, 4.3)
    clean(ax)
    handles = [Line2D([], [], color=C["sub"], lw=1.3, label="2023 年"),
               Line2D([], [], color=C["sub"], lw=1.3, ls=(0, (3, 2)), label="2024 年")]
    if s:
        panel_label(ax, "相对 0 个 GRU（实线 2023，虚线 2024）")
    else:
        ax.legend(handles=handles, loc="lower left", ncol=2)
        panel_label(ax, "相对不含 GRU 的集成的变化")
    for (metric, y), (v, r) in rel.items():
        print(f"[GRU 个数] {metric} {y}: " + "  ".join(f"{k}个 {a:.4f} ({b:+.2f}%)" for k, a, b in zip(xs, v, r)))
    save(fig, f"fig_gru_count_{mode}")


# ---------------------------------------------------------------- 图 5：逐时段 RankIC

def fig_slot_ic(mode):
    s = mode == "s"
    fig = new_fig(*((8.2, 5.2) if s else (15.6, 6.2)))
    ax = fig.add_subplot()
    data = {y: slot_ic(y)[0] for y in (2023, 2024)}
    xs = np.arange(len(SLOTS))
    bw = 0.39
    for j, y in enumerate((2023, 2024)):
        for i, v in enumerate(data[y]):
            last = i == len(SLOTS) - 1
            col = (C["accent"] if j == 0 else C["accent2"]) if last else (C["navy"] if j == 0 else C["mid"])
            x = xs[i] + (j - 0.5) * bw
            ax.bar(x, v, width=bw * 0.92, color=col, zorder=3)
            if not s:
                ax.text(x, v - 0.004, num(v, 3), rotation=90, ha="center", va="top",
                        color=C["ink"] if (last and j == 1) else "white", zorder=6)
            elif last and j == 0:
                label(ax, x, v, num(v, 3), color=C["accent"], weight="bold", dy=2)
            elif last:
                label(ax, x + bw * 0.46, v, num(v, 3), color=C["accent"], weight="bold", dx=2, dy=0, ha="left",
                      va="center")
    ratio = {}
    for j, y in enumerate((2023, 2024)):
        m = float(np.mean(data[y][:-1]))
        ratio[y] = (m, data[y][-1] / m)
        ax.plot([xs[0] - 0.45, xs[-2] + 0.45], [m, m], color=C["navy"] if j == 0 else C["mid"], lw=1.0,
                ls=(0, (3, 2)), zorder=5)
    if s:
        txt = f"为其余时段均值的\n{ratio[2023][1]:.2f} 倍／{ratio[2024][1]:.2f} 倍"
        xyt = (3.1, 0.165)
    else:
        txt = f"约为其余六个时段均值的\n{ratio[2023][1]:.2f} 倍（2023）、{ratio[2024][1]:.2f} 倍（2024）"
        xyt = (xs[-1] - 1.05, 0.160)
    ax.annotate(txt, xy=(xs[-1] - bw - 0.08, data[2023][-1] * 0.93), xytext=xyt,
                ha="center" if s else "right", va="center", color=C["accent"], fontweight="bold", linespacing=1.25,
                arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=0.9, mutation_scale=8, shrinkA=3, shrinkB=1))
    ax.set_xticks(xs, SLOT_NAMES)
    t = ax.get_xticklabels()[-1]
    t.set_color(C["accent"])
    t.set_fontweight("bold")
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 0.198)
    ax.set_xlim(-0.6, len(SLOTS) + (0.45 if s else -0.35))
    ax.set_yticks([])
    clean(ax, left=False, grid=False)
    mtxt = (f"其余时段均值 {ratio[2023][0]:.3f}／{ratio[2024][0]:.3f}" if s
            else f"其余六个时段均值（{ratio[2023][0]:.3f}／{ratio[2024][0]:.3f}）")
    handles = [Patch(color=C["navy"], label="2023 年"), Patch(color=C["mid"], label="2024 年"),
               Line2D([], [], color=C["sub"], lw=1.0, ls=(0, (3, 2)), label=mtxt)]
    fig.legend(handles=handles, loc="outside lower center", ncol=2 if s else 3)
    panel_label(ax, "逐截面 RankIC（留出验证模型）")
    for y in (2023, 2024):
        print(f"[逐时段] {y}: " + "  ".join(f"{n} {v:.4f}" for n, v in zip(SLOT_NAMES, data[y])) +
              f"  其余均值 {ratio[y][0]:.4f}  倍数 {ratio[y][1]:.3f}")
    save(fig, f"fig_slot_ic_{mode}")


# ---------------------------------------------------------------- 图 6：四个半年

def fig_halfyear(mode):
    s = mode == "s"
    rows = halfyear("T4G1R2")
    fig = new_fig(*((14.0, 4.6) if s else (15.6, 5.6)))
    axs = fig.subplots(1, 3, gridspec_kw=dict(wspace=0.08))
    spec = (("ic_mean", "RankIC", 4), ("ic_ir", "ICIR", 3), ("turnover", "换手率", 4))
    xs = np.array([0.0, 1.0, 2.35, 3.35])
    for ax, (k, name, dig) in zip(axs, spec):
        v = [r[1][k] for r in rows]
        top = max(v) * 1.06
        for i, (x, val) in enumerate(zip(xs, v)):
            ax.bar(x, val, width=0.80, color=C["accent"] if i < 2 else C["accent2"], zorder=3)
            ax.text(x, val - top * 0.035, num(val, dig), rotation=90, ha="center", va="top",
                    color="white" if i < 2 else C["ink"], zorder=6)
        ratio = max(v) / min(v)
        ax.set_ylim(0, top)
        ax.set_xlim(-0.6, 3.95)
        ax.set_xticks(xs, ["上", "下", "上", "下"])
        ax.set_xticks([0.5, 2.85], ["2023 年", "2024 年"], minor=True)
        ax.tick_params(axis="x", which="major", length=0)
        ax.tick_params(axis="x", which="minor", length=0, pad=17)
        ax.set_yticks([])
        clean(ax, left=False, grid=False)
        panel_label(ax, f"{name}（最大÷最小 {ratio:.2f}）")
        print(f"[半年] {name}: " + "  ".join(f"{r[0]} {x:.4f}" for r, x in zip(rows, v)) + f"  最大÷最小 {ratio:.3f}")
    for k, name in (("ls_sharpe", "多空组合夏普比率"), ("stress", "压力稳定性")):
        v = [r[1][k] for r in rows]
        print(f"[半年] {name}: " + "  ".join(f"{x:.3f}" for x in v) + f"  最大÷最小 {max(v) / min(v):.3f}")
    save(fig, f"fig_halfyear_{mode}")


# ---------------------------------------------------------------- 图 7：本地增量与公榜增量

def agreement():
    def sgn(v):
        return 0 if abs(v) < 5e-5 else (1 if v > 0 else -1)

    big = [p for p in PAIRS if abs(p[1]) >= SIG - 1e-9]
    small = [p for p in PAIRS if abs(p[1]) < SIG - 1e-9]

    def count(ps, a, b):
        same = sum(1 for p in ps if sgn(p[a]) != 0 and sgn(p[b]) != 0 and sgn(p[a]) == sgn(p[b]))
        opp = sum(1 for p in ps if sgn(p[a]) != 0 and sgn(p[b]) != 0 and sgn(p[a]) != sgn(p[b]))
        return same, opp, len(ps) - same - opp

    return [(f"RankIC：本地增量 ≥ 0.002 的 {len(big)} 组", len(big), count(big, 1, 2)),
            (f"RankIC：本地增量 < 0.002 的 {len(small)} 组", len(small), count(small, 1, 2)),
            (f"换手率：全部 {len(PAIRS)} 组", len(PAIRS), count(PAIRS, 3, 4))]


def draw_agreement(ax, s):
    rows = agreement()
    for i, (lab, n, (same, opp, zero)) in enumerate(rows):
        y = -i * 1.0
        left = 0.0
        for cnt, col, tc in ((same, C["accent"], "white"), (opp, C["navy"], "white"), (zero, C["light"], C["ink"])):
            if cnt:
                w = cnt / n
                ax.barh(y, w, left=left, height=0.40, color=col, edgecolor="white", lw=1.0, zorder=3)
                ax.text(left + w / 2, y, f"{cnt}", ha="center", va="center", color=tc, fontweight="bold", zorder=5)
                left += w
        ax.text(0, y + 0.29, lab, ha="left", va="bottom", color=C["ink"])
        good = opp == 0
        ax.text(1, y + 0.29, f"{same}／{same + opp} 一致", ha="right", va="bottom",
                color=C["accent"] if good else C["ink"], fontweight="bold" if good else "normal")
    ax.set_xlim(0, 1)
    ax.set_ylim(-(len(rows) - 1) - 0.32, 0.78)
    ax.set_xticks([])
    ax.set_yticks([])
    clean(ax, left=False, grid=False, bottom=False)
    handles = [Patch(color=C["accent"], label="方向一致"), Patch(color=C["navy"], label="方向相反"),
               Patch(color=C["light"], label="一侧为 0")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    return rows


def fig_local_public(mode):
    s = mode == "s"
    if s:
        fig = new_fig(8.2, 5.2)
        ax = fig.add_subplot()
        rows = draw_agreement(ax, s)
        panel_label(ax, "公榜增量与本地增量的方向（21 组改动）")
    else:
        fig = new_fig(15.6, 6.8)
        axl, axr = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[0.78, 1.22], wspace=0.08))
        axl.axvspan(-SIG, SIG, color=C["tint"], zorder=0, lw=0)
        lo, hi = -0.0215, 0.0085
        axl.plot([lo, hi], [lo, hi], color=C["sub"], lw=0.8, ls=(0, (3, 2)), zorder=1)
        axl.axhline(0, color=C["light"], lw=0.6, zorder=1)
        axl.axvline(0, color=C["light"], lw=0.6, zorder=1)
        for name, il, ip, tl, tp in PAIRS:
            big = abs(il) >= SIG - 1e-9
            axl.scatter(il, ip, s=26 if big else 18, zorder=4, color=C["accent"] if big else C["mid"],
                        edgecolor="white", lw=0.6)
        ann = {"秩高斯目标＋4 轮": ("秩高斯目标", -5, 0, "right", "center"),
               "换手正则项 1.0": ("换手正则项 1.0", 6, 0, "left", "center")}
        for name, il, ip, tl, tp in PAIRS:
            if name in ann:
                t, dx, dy, ha, va = ann[name]
                label(axl, il, ip, t, color=C["ink"], dx=dx, dy=dy, ha=ha, va=va)
            if name == "全天 5 分钟片段":
                axl.annotate("全天 5 分钟片段", xy=(il, ip), xytext=(-0.0025, -0.0118), ha="center", va="top",
                             color=C["ink"], arrowprops=dict(arrowstyle="-", color=C["sub"], lw=0.7,
                                                             shrinkA=1, shrinkB=3))
        axl.set_xlim(lo, hi)
        axl.set_ylim(lo, hi)
        tk = [-0.02, -0.01, 0.0]
        axl.set_xticks(tk, [num(v, 2) if v else "0" for v in tk])
        axl.set_yticks(tk, [num(v, 2) if v else "0" for v in tk])
        axl.set_xlabel("本地 RankIC 增量（两年平均）")
        axl.set_ylabel("公榜 RankIC 增量")
        axl.set_aspect("equal", adjustable="box")
        clean(axl, grid=False)
        panel_label(axl, "虚线为 y = x，浅色带为 ±0.002")
        rows = draw_agreement(axr, s)
        panel_label(axr, "公榜增量与本地增量的方向")
    il = np.array([p[1] for p in PAIRS])
    ip = np.array([p[2] for p in PAIRS])
    tl = np.array([p[3] for p in PAIRS])
    tp = np.array([p[4] for p in PAIRS])
    print(f"[本地对公榜] 配对 {len(PAIRS)} 组；相关：RankIC {np.corrcoef(il, ip)[0, 1]:.3f}，换手率 {np.corrcoef(tl, tp)[0, 1]:.3f}")
    for lab, n, (same, opp, zero) in rows:
        print(f"   {lab}：一致 {same}，相反 {opp}，一侧为 0 {zero}")
    save(fig, f"fig_local_public_{mode}")


# ---------------------------------------------------------------- 图 8：公榜到私榜

def fig_public_private(mode):
    s = mode == "s"
    d23 = ens("final", 2023)
    d24 = ens("final", 2024)
    series = [("私榜", PRIVATE, C["accent"]), ("留出验证模型 2024 年", d24, C["navy"])]
    if not s:
        series.append(("留出验证模型 2023 年", d23, C["light"]))
    fig = new_fig(*((8.2, 5.4) if s else (15.6, 7.2)))
    ax = fig.add_subplot()
    nser = len(series)
    bh = 0.8 / nser
    ys = np.arange(len(KEYS))[::-1]
    for y, k in zip(ys, KEYS):
        for j, (lab, d, col) in enumerate(series):
            v = d[k] / PUBLIC[k] - 1
            yy = y + (nser - 1) / 2 * bh - j * bh
            ax.barh(yy, v * 100, height=bh * 0.9, color=col, zorder=3)
            tc = C["accent"] if j == 0 else (C["sub"] if col == C["light"] else C["ink"])
            ax.annotate(pct(v, 0 if s else 1), (v * 100, yy), xytext=(-3 if v < 0 else 3, 0),
                        textcoords="offset points", ha="right" if v < 0 else "left", va="center", color=tc,
                        fontweight="bold" if j == 0 else "normal", zorder=6, annotation_clip=False)
    ax.axvline(0, color=C["sub"], lw=1.0, zorder=4)
    panel_label(ax, "相对最终模型公榜的变化", align_ticks=True)
    t = ax.annotate("公榜", (0, 1.0), xycoords=mtrans.blended_transform_factory(ax.transData, ax.transAxes),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", color=C["sub"])
    t.set_in_layout(False)
    ax.set_yticks(ys, NAMES5)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlim(-40 if s else -38, 19 if s else 22)
    ax.set_ylim(-0.55, len(KEYS) - 0.45)
    ax.set_xticks([])
    clean(ax, left=False, grid=False, bottom=False)
    handles = [Patch(color=col, label=lab) for lab, _, col in series]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(series))
    print("[公私榜] 相对最终模型公榜的变化：")
    for k, nm in zip(KEYS, NAMES5):
        print(f"   {nm}: 公榜 {PUBLIC[k]}  私榜 {PRIVATE[k]} ({PRIVATE[k] / PUBLIC[k] - 1:+.4f}, 比值 {PRIVATE[k] / PUBLIC[k]:.3f})"
              f"  留出 2024 {d24[k]:.4f} ({d24[k] / PUBLIC[k] - 1:+.4f})  留出 2023 {d23[k]:.4f} ({d23[k] / PUBLIC[k] - 1:+.4f})"
              f"  私榜÷留出2024 {PRIVATE[k] / d24[k]:.3f}  留出2024÷2023 {d24[k] / d23[k]:.3f}")
    save(fig, f"fig_public_private_{mode}")


# ---------------------------------------------------------------- 图 9：梯度提升树子模型的分裂统计

def draw_groups(ax, share, s):
    order = sorted(share, key=lambda g: -share[g])
    if s:
        keep, rest = order[:7], order[7:]
        items = [(GROUP_NAME[g], share[g], g) for g in keep] + [(f"其余 {len(rest)} 组", sum(share[g] for g in rest), "rest")]
    else:
        items = [(GROUP_NAME[g], share[g], g) for g in order]
    ys = np.arange(len(items))[::-1]
    for y, (lab, v, g) in zip(ys, items):
        col = C["accent"] if g == "f_vol" else (C["light"] if g == "rest" else C["navy"])
        ax.barh(y, v, height=0.68, color=col, zorder=3)
        label(ax, v, y, f"{v:.1f}%", color=C["accent"] if g == "f_vol" else C["ink"],
              weight="bold" if g == "f_vol" else "normal", dx=3, dy=0, ha="left", va="center")
    ax.set_yticks(ys, [it[0] for it in items])
    for t, it in zip(ax.get_yticklabels(), items):
        if it[2] == "f_vol":
            t.set_color(C["accent"])
            t.set_fontweight("bold")
    ax.set_xlim(0, 28.5 if s else 26.0)
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    clean(ax, left=False, grid=False, bottom=False)
    panel_label(ax, "分裂份额（按节点深度加权）", align_ticks=True)
    return items


def fig_tree_splits(mode):
    s = mode == "s"
    ntree, nsplit, share, root = tree_stats()
    for g, v in GROUP_DOC.items():
        assert abs(share[g] - v) < 0.051, (g, share[g], v)
    if s:
        fig = new_fig(8.2, 5.2)
        ax = fig.add_subplot()
        items = draw_groups(ax, share, s)
    else:
        fig = new_fig(15.6, 7.2)
        axl, axr = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.0], wspace=0.04))
        items = draw_groups(axl, share, s)
        top = root.most_common(8)
        ys = np.arange(len(top))[::-1]
        book = (13, 16)  # 最后一根 1 分钟线的买一量、卖一量
        for y, (f, n) in zip(ys, top):
            col = C["accent"] if f in book else C["navy"]
            axr.barh(y, n, height=0.62, color=col, zorder=3)
            label(axr, n, y, f"{n}", color=C["accent"] if f in book else C["ink"],
                  weight="bold" if f in book else "normal", dx=3, dy=0, ha="left", va="center")
        axr.set_yticks(ys, [ROOT_NAME[f] for f, _ in top])
        for t, (f, _) in zip(axr.get_yticklabels(), top):
            if f in book:
                t.set_color(C["accent"])
                t.set_fontweight("bold")
        axr.set_xlim(0, 43)
        axr.set_ylim(-0.6, len(top) - 0.4)
        axr.set_xticks([])
        axr.tick_params(axis="y", length=0)
        clean(axr, left=False, grid=False, bottom=False)
        panel_label(axr, f"根节点使用次数（共 {ntree} 棵树）", align_ticks=True)
    print(f"[树分裂] {ntree} 棵树，{nsplit} 次分裂；份额 " +
          "  ".join(f"{GROUP_NAME[g]} {share[g]:.2f}" for g in sorted(share, key=lambda g: -share[g])))
    if s:
        print(f"   幻灯片版合并：{items[-1][0]} {items[-1][1]:.2f}")
    print("   根节点：" + "  ".join(f"{ROOT_NAME.get(f, f)} {n}" for f, n in root.most_common(8)))
    save(fig, f"fig_tree_splits_{mode}")


# ---------------------------------------------------------------- 图 10：训练轮数

def fig_epochs(mode):
    s = mode == "s"
    hist = json.load(open(EPOCHS, encoding="utf-8"))["history"]
    ep = [h["epoch"] for h in hist]
    fig = new_fig(*((14.0, 4.6) if s else (15.6, 5.4)))
    axs = fig.subplots(1, 3, gridspec_kw=dict(wspace=0.08))
    spec = (("ic", "RankIC", 4, (0.0915, 0.1108)), ("icir", "ICIR", 3, (1.90, 2.19)),
            ("turnover", "换手率（越低越好）", 3, (0.562, 0.618)))
    for ax, (k, name, dig, lim) in zip(axs, spec):
        v = [h[k] for h in hist]
        ax.axvspan(5.6, 6.4, color=C["tint"], lw=0, zorder=0)
        ax.plot(ep, v, color=C["navy"], lw=1.3, zorder=3)
        ax.scatter(ep, v, s=[30 if e == 6 else 12 for e in ep], zorder=4,
                   color=[C["accent"] if e == 6 else C["navy"] for e in ep], edgecolor="white", lw=0.6)
        i6 = ep.index(6)
        below6 = k == "turnover"
        label(ax, 6, v[i6], num(v[i6], dig), color=C["accent"], weight="bold", dy=-7 if below6 else 7,
              va="top" if below6 else "bottom")
        # 另标出第 1 轮、第 8 轮（ICIR 另标峰值第 3 轮）：(轮, 竖向偏移, 水平对齐, 横向偏移)
        marks = {"ic": [(1, -4, "left", 5), (8, -7, "right", 4)],
                 "icir": [(1, -4, "left", 5), (3, 7, "center", 0), (8, -7, "right", 4)],
                 "turnover": [(1, 7, "left", -4), (8, 7, "right", 4)]}[k]
        for e, dy, ha, dx in marks:
            i = ep.index(e)
            label(ax, e, v[i], num(v[i], dig), color=C["sub"], dy=dy, va="top" if dy < 0 else "bottom", ha=ha, dx=dx)
        ax.set_ylim(*lim)
        ax.set_xlim(0.5, 8.5)
        ax.set_xticks(ep)
        ax.set_yticks([])
        ax.set_xlabel("训练轮数")
        clean(ax, left=False, grid=False)
        panel_label(ax, name)
        print(f"[训练轮数] {name}: " + "  ".join(f"{e}:{x:.4f}" for e, x in zip(ep, v)))
    save(fig, f"fig_epochs_{mode}")


FIGS = {
    "public_path": fig_public_path,
    "increment_chain": fig_increment_chain,
    "turnover_reg": fig_turnover_reg,
    "gru_count": fig_gru_count,
    "slot_ic": fig_slot_ic,
    "halfyear": fig_halfyear,
    "local_public": fig_local_public,
    "public_private": fig_public_private,
    "tree_splits": fig_tree_splits,
    "epochs": fig_epochs,
}


def main():
    global REPORT_PALETTE, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-palette", choices=["slides", "report"], default="slides")
    ap.add_argument("--only", choices=list(FIGS), nargs="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    REPORT_PALETTE = a.report_palette
    if a.out:
        OUT = Path(a.out)
        OUT.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    for name in a.only or FIGS:
        for mode in ("s", "r"):
            use_palette(mode)
            FIGS[name](mode)


if __name__ == "__main__":
    main()
