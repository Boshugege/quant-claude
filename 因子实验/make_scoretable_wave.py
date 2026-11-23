# -*- coding: utf-8 -*-
"""把评分机里 160–196 的已完成结果生成 Markdown 表格，追加到《因子得分总表.md》的第十五节。

用法：python3 因子实验/make_scoretable_wave.py        # 只打印
      python3 因子实验/make_scoretable_wave.py --write # 覆盖写入第十五节
"""
import glob
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
START, END = 160, 196
DESC = {
    160: "同伴残差化：买卖压力", 161: "同伴残差化：振幅", 162: "同伴残差化：竞价放量",
    163: "同伴残差化：成交集中度", 164: "日内同伴 + 六尺度", 165: "日内同伴 5 日窗口",
    166: "日内同伴 60 日窗口", 167: "日内同伴 + 行业限定", 168: "日内同伴 + 大单冲击",
    169: "日内同伴 + 竞价占比", 170: "同伴窗口 240 日", 171: "同伴窗口 480 日",
    172: "日内同伴 240 日", 173: "长窗口同伴 + 六尺度", 174: "长窗口同伴 + 大单冲击",
    175: "长窗口同伴 k=400", 176: "长窗口同伴 + 分钟上影线", 177: "长窗口同伴 + 14:30 首半小时动量",
    178: "长窗口同伴 + 六尺度 + 上影线", 179: "长窗口同伴 + 六特征中性化", 180: "日内同伴 + 分钟上影线",
    181: "四杠杆叠加", 182: "路径效率附加（新方向）", 183: "路径效率正号对照",
    184: "1 分钟粒度已实现偏度", 185: "上涨分钟占比", 186: "分钟价格冲击斜率（Kyle 近似）",
    187: "单分钟最大涨幅", 188: "零变化分钟占比", 189: "成交额-波动共同运动",
    190: "大单加权方向偏离", 191: "大单加权方向偏离正号对照", 192: "极端共跌依赖加权同伴",
    193: "同伴相关性半衰期 10 日", 194: "同伴相关性半衰期 30 日", 195: "大单成本锚阶梯附加尺度",
    196: "大单成本锚替换等权 VWAP 尺度",
}
EXTRA = {"132_indpeer2": "行业内部软同伴（修正后重跑）", "141_ind200p3b": "行业内部软同伴 k=200/3（修正后重跑）",
         "152_indintra2": "行业内部 + 日内同伴（修正后重跑）", "167_intra_ind2": "日内同伴 + 行业（修正后重跑）"}


def collect():
    best = {}
    for d in sorted(glob.glob(str(ROOT / "工具" / "评分机" / "结果" / "*"))):
        f = os.path.join(d, "result.json")
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        name = (j.get("meta") or {}).get("name") or "?"
        m = j.get("metrics") or {}
        if not m:
            continue
        q = {k: (0.0 if v is None else float(v)) for k, v in ((j.get("reference_pool") or {}).get("percentile") or {}).items()}
        proxy = sum(q.get(k, 0) for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")) / 5 if q else float("nan")
        num = None
        mm = re.match(r"^(\d{3})_", name)
        if mm:
            num = int(mm.group(1))
        elif name in EXTRA:
            num = int(name.split("_")[0])
        if num is None:
            continue
        if not (START <= num <= END) and name not in EXTRA:
            continue
        row = (proxy, name, os.path.basename(d), m, q)
        if name not in best or row[0] > best[name][0]:
            best[name] = row
    return best


def main():
    best = collect()
    lines = ["## 十五、第十八批补全（160–196）——discover 波次后段与两个新族（评分机平台口径自评）", "",
             "与第十四节同一口径：区间 2023-01~2024-12、9 个自然月、8 个截面（**已按用户要求补 09:30 截面**）、",
             "股票池为平台成分股。代理分 = 参照池五项百分位均值。", "",
             "| 编号 | 说明 | IC | IC_IR | 多空夏普 | 压力 | 换手 | 分位（IC/IR/LS/压/换） | 代理分 | 任务号 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for num in range(START, END + 1):
        hits = [(n, r) for n, r in best.items() if n.startswith("%d_" % num)]
        if not hits:
            lines.append("| %d | %s | 评测中 | — | — | — | — | — | — | — |" % (num, DESC.get(num, "")))
            continue
        for name, (proxy, _n, jid, m, q) in sorted(hits):
            lines.append("| %d | %s | %.4f | %.3f | %.2f | %.3f | %.3f | %.2f / %.2f / %.2f / %.2f / %.2f | %.3f | %s |" % (
                num, DESC.get(num, name), m.get("ic_mean", 0), m.get("ic_ir", 0), m.get("ls_sharpe", 0),
                m.get("stress", 0), m.get("turnover", 0),
                q.get("ic_mean", 0), q.get("ic_ir", 0), q.get("ls_sharpe", 0), q.get("stress", 0), q.get("turnover", 0),
                proxy, jid))
    for name, label in EXTRA.items():
        if name in best:
            proxy, _n, jid, m, q = best[name]
            lines.append("| %s | %s | %.4f | %.3f | %.2f | %.3f | %.3f | %.2f / %.2f / %.2f / %.2f / %.2f | %.3f | %s |" % (
                name.split("_")[0], label, m.get("ic_mean", 0), m.get("ic_ir", 0), m.get("ls_sharpe", 0),
                m.get("stress", 0), m.get("turnover", 0),
                q.get("ic_mean", 0), q.get("ic_ir", 0), q.get("ls_sharpe", 0), q.get("stress", 0), q.get("turnover", 0),
                proxy, jid))
    lines += ["", "> 说明：同编号的多次运行只保留代理分最高的一条；失败/修正后重跑的记录见《第十八批因子说明》。", ""]
    text = "\n".join(lines)
    if "--write" in sys.argv:
        p = ROOT / "因子得分总表.md"
        s = p.read_text(encoding="utf-8")
        marker = "## 十五、"
        if marker in s:
            s = s[:s.index(marker)]
        p.write_text(s.rstrip("\n") + "\n\n" + text, encoding="utf-8")
        print("已写入", p, "行数", len(lines))
    else:
        print(text)


if __name__ == "__main__":
    main()
