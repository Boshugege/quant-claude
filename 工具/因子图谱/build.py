# -*- coding: utf-8 -*-
"""汇总因子图谱数据：raw_factors.json + 本地跑分 CSV + 总表里的公榜分数 + 总表阶段标题 + 分量词汇表 → factor_graph.json。"""
import csv, json, os, re, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
R = json.load(open(os.path.join(HERE, "raw_factors.json"), encoding="utf-8"))
byn = {r["num"]: r for r in R}

# ---- 本地跑分（评测机 result.json 汇总，2023–2024 全区间，同号取最新）
local = {}
for row in csv.DictReader(open(os.path.join(ROOT, "文档", "因子跑分明细_2026-10-13.csv"), encoding="utf-8")):
    if row["start"] != "2023-01-01" or row["end"] != "2024-12-31" or not row["num"].strip().isdigit():
        continue
    n = int(row["num"])
    if n in local and local[n]["created"] >= row["created"]:
        continue
    local[n] = {"created": row["created"], "ic": float(row["ic_mean"]), "ir": float(row["ic_ir"]), "ls": float(row["ls_sharpe"]),
                "st": float(row["stress"]), "to": float(row["turnover"]),
                "c104": float(row["corr104_rank"]) if row["corr104_rank"] not in ("", "nan") else None,
                "fam_remote": row["family"]}

# ---- 公榜分数：总表里的若干快照
txt = open(os.path.join(ROOT, "因子得分总表.md"), encoding="utf-8").read()
lines = txt.splitlines()
pub = collections.defaultdict(list)

def sec_of(i):
    for j in range(i, -1, -1):
        if lines[j].startswith("## "):
            return lines[j][3:].strip()
    return ""

def cells(l):
    return [c.strip() for c in l.strip().strip("|").split("|")]

def num(s):
    m = re.search(r"0\.\d{4,5}", s.replace("*", ""))
    return float(m.group()) if m else None

i = 0
while i < len(lines):
    l = lines[i]
    if l.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1]):
        head = cells(l)
        sec = sec_of(i)
        j = i + 2
        rows = []
        while j < len(lines) and lines[j].startswith("|"):
            rows.append(cells(lines[j])); j += 1
        hn = head.index("编号") if "编号" in head else None
        target = None
        label = None
        if sec.startswith("一、"):
            pass   # 当前最佳表与提交记录重复，略
        elif sec.startswith("二、") and "分数" in head:
            target, label = head.index("分数"), "早期提交记录（10-10 规则调整前后混合，仅作参考）"
        elif sec.startswith("公榜快照") and "现在" in head:
            target, label = head.index("现在"), "公榜快照 2026-10-12"
        elif sec.startswith("二十五、") and "分数" in head:
            target, label = head.index("分数"), "同批快照 2026-10-12（第二十七批 A 组）"
        elif sec.startswith("三十一、") and "公榜" in head:
            target, label = head.index("公榜"), "同批快照 2026-10-13（1012 晚 17 个）"
        if target is not None and hn is not None:
            for r in rows:
                if len(r) <= max(hn, target):
                    continue
                m = re.search(r"\d+", r[hn].replace("*", ""))
                v = num(r[target])
                if m and v:
                    rec = {"snap": label, "score": v}
                    for k, key in (("IC_IR", "ir"), ("IC 均值", "ic"), ("换手", "to"), ("多空夏普", "ls"), ("多空", "ls"), ("压力", "st")):
                        if k in head and len(r) > head.index(k):
                            try:
                                rec[key] = float(r[head.index(k)].replace("*", ""))
                            except ValueError:
                                pass
                    pub[int(m.group())].append(rec)
        i = j
        continue
    i += 1

# ---- 阶段：总表里带编号范围的标题（取覆盖该编号的最窄范围）
stages = []
for l in lines:
    if not re.match(r"^#{2,3} ", l):
        continue
    t = l.lstrip("#").strip()
    for a, b in re.findall(r"(\d{2,4})\s*[–\-~]\s*(\d{2,4})", t):
        a, b = int(a), int(b)
        if 0 < a <= b <= 1100 and b - a < 400:
            stages.append({"a": a, "b": b, "title": t})
def stage_of(n):
    c = [s for s in stages if s["a"] <= n <= s["b"]]
    return min(c, key=lambda s: s["b"] - s["a"])["title"] if c else None

# ---- 时代（规则口径）
ERAS = [(1, 89, "E1 早期单信号（1–89）"), (90, 210, "E2 反转阶梯与同伴残差（90–210）"),
        (211, 352, "E3 头部叠加：上影线/单笔/实体/中性化（211–352）"), (353, 443, "E4 暴露表禁用期的合法家族（353–443）"),
        (444, 560, "E5 暴露表解禁：盘口/低通/中性化力度（444–560）"), (561, 830, "E6 同伴定义/条件反转/平滑族（561–830）"),
        (831, 982, "E7 时点重心臂与条件反转扩展（831–982）"), (983, 1100, "E8 日内微结构新源（983–1049）")]
def era_of(n):
    for a, b, t in ERAS:
        if a <= n <= b:
            return t
    return "?"

# ---- 分量词汇表：每个特征第一次定义处的注释
CODE = os.path.join(ROOT, "因子代码")
gloss = {}
for r in sorted(R, key=lambda x: x["num"]):
    need = [d for d in r["defs"] if d not in gloss]
    if not need:
        continue
    s = open(os.path.join(CODE, r["file"]), encoding="utf-8", errors="ignore").read()
    if r["file"].endswith(".ipynb"):
        try:
            s = "\n".join("".join(c["source"]) for c in json.loads(s)["cells"])
        except Exception:
            pass
    L = s.splitlines()
    for d in need:
        for k, ln in enumerate(L):
            if re.search(r'df\["%s"\]\s*=' % re.escape(d), ln):
                com = []
                q = k - 1
                while q >= 0 and L[q].strip().startswith("#") and len(com) < 3:
                    com.insert(0, L[q].strip().lstrip("#").strip()); q -= 1
                inline = ln.split("#", 1)[1].strip() if "#" in ln and not ln.strip().startswith("#") else ""
                gloss[d] = {"first": r["num"], "desc": " ".join(com + ([inline] if inline else []))[:300], "code": ln.strip()[:240]}
                break
MANUAL = {
    "r_30": "30 分钟反转：ln(上一窗口末价 / 当前价)", "r_day": "日内反转：ln(昨收 / 当前价)",
    "r_rv4": "约 2 小时 VWAP 偏离反转：ln(最近 4 个窗口 VWAP / 当前价)", "r_rv8": "约 4 小时 VWAP 偏离反转（8 个窗口）",
    "r_rv16": "约 2 日尺度 VWAP 偏离反转（16 个窗口）", "r_rv32": "约 4 日尺度 VWAP 偏离反转（32 个窗口）",
    "c_size": "规模：−ln(最近 40 个窗口成交额)", "c_ivol": "特质波动（取负）：40 窗口对市场回归残差的标准差",
    "c_illiq": "非流动性（Amihud）：|收益| / 成交额 的 40 窗口均值", "c_lowpx": "低价：−ln(价格)", "c_max": "MAX（取负）：40 窗口最大收益",
    "c_turn": "换手率代理：40 窗口成交额 ÷ 流通市值", "c_mcap2": "流通市值非线性项：ln(流通市值)²",
    "c_avgtr": "平均单笔成交额：Σ成交额 / Σ成交笔数 的 40 窗口均值", "c_deals": "成交笔数类特征",
    "c_cond_turnself": "条件反转：按自身换手历史分位分组，组内去均值的反转",
    "c_condliq": "条件反转：按官方 LIQUIDTY 风格分组", "c_condsize": "条件反转：按官方 SIZE 风格分组",
    "c_condresvol": "条件反转：按官方 RESVOL 风格分组", "c_condbeta": "条件反转：按官方 BETA 风格分组", "c_condmom": "条件反转：按官方 MOMENTUM 风格分组",
    "c_upshtwres": "换手加权行业上影残差：上影线占比减去行业内换手加权均值", "c_gap": "隔夜跳空（40 窗口均值，取负）",
    "c_ob": "盘口挂单不平衡：Σ(买量−卖量)/Σ(买量+卖量) 的 40 窗口均值",
    "c_indmwrev": "行业市值加权反转（行业层面的平均反转）", "c_amthhi": "成交额日内集中度 HHI（取负）",
    "c_aucshare": "集合竞价成交占比（相对前一日成交额）", "c_maxshare": "单窗口最大成交额占比（取负）",
    "c_open30": "开盘 30 分钟特征（40 窗口均值）", "c_first30": "首个连续竞价窗口（09:30–10:00）收益（取负）",
    "c_m30": "日内按 (股票, 日) 取最大值的特征（⚠ 若不是只在开盘窗口有值，则含未来信息，待确认）",
    "c_auction": "集合竞价成交额 ÷ 全天成交额（⚠ 全天分母含未来信息）",
    "ladder": "同伴残差阶梯：多尺度反转（30 分钟 / 2–4 小时 VWAP / 日内）对相关同伴取残差后截面排名合成（104 号核心）",
          "c_utime": "上影线时点重心（⚠ 按整天聚合，含未来信息，见未来信息审计）",
          "c_hitime": "日内最高价出现时段（⚠ 按整天聚合，含未来信息）", "c_lotime": "日内最低价出现时段（⚠ 按整天聚合，含未来信息）"}
for k, v in MANUAL.items():
    gloss.setdefault(k, {"first": None, "code": ""})
    gloss[k]["desc"] = v

# ---- 树结构统计
kids = collections.defaultdict(list)
for r in R:
    if r["parent"] is not None:
        kids[r["parent"]].append(r["num"])
def subtree(n):
    st, c = [n], 0
    while st:
        x = st.pop(); c += 1; st += kids.get(x, [])
    return c
depth = {}
for r in sorted(R, key=lambda x: x["num"]):
    depth[r["num"]] = 0 if r["parent"] is None else depth.get(r["parent"], 0) + 1

nodes = []
for r in R:
    n = r["num"]
    nodes.append({
        "n": n, "name": r["name"], "file": r["file"], "p": r["parent"], "sim": r["sim"], "ch": r["change"], "dn": r["diff_n"],
        "diff": r["diff"], "comps": r["comps"], "neu": r["neutral"][-1][1] if r["neutral"] else None, "neuset": r["neutral"][-1][0] if r["neutral"] else None,
        "defs": r["defs"], "fields": [f for f in r["fields"] if not f.startswith("cpt_jyc_2026_stock") and f != "cpt_jyc_2026_instruments"],
        "leak": r["leak"], "sig": r["signal"], "loc": local.get(n), "pub": pub.get(n, []), "stage": stage_of(n), "era": era_of(n),
        "kids": sorted(kids.get(n, [])), "sub": subtree(n), "depth": depth[n]})
out = {"nodes": nodes, "gloss": gloss, "eras": [t for _, _, t in ERAS], "stages": stages,
       "meta": {"n": len(nodes), "with_local": sum(1 for x in nodes if x["loc"]), "with_pub": sum(1 for x in nodes if x["pub"]),
                "leaky": sum(1 for x in nodes if x["leak"]), "built": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}}
json.dump(out, open(os.path.join(HERE, "factor_graph.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(out["meta"], "gloss", len(gloss), "stages", len(stages), os.path.getsize(os.path.join(HERE, "factor_graph.json")) // 1024, "KB")

# ---- 生成页面
tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
js = json.dumps(out, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
open(os.path.join(HERE, "因子谱系图.html"), "w", encoding="utf-8").write(tpl.replace("/*DATA*/", js))
print("page", os.path.getsize(os.path.join(HERE, "因子谱系图.html")) // 1024, "KB")
