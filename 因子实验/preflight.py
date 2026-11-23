# -*- coding: utf-8 -*-
"""提交前静态自检：把本轮踩过的坑固化成检查项，任何一项不通过就不要提交。

用法：python3 因子实验/preflight.py [文件...]（默认检查 因子代码/113 起的全部文件）
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX930 = "time_segment = 93000 AND hms = '092500'"
ZERO930 = '== "0930"'       # 显式把 09:30 截面置 0（与 104/103/107 同口径，平台会跳过全 0 截面）
CONTROL = {"120", "204"}   # 对照：120 完全不补 09:30，204 故意让 09:30 有信号（复核用）
SCALES = ("r_rv16", "r_bv4", "r_bv8")


def check(p):
    errs = []
    warns = []
    s = p.read_text(encoding="utf-8")
    num = p.name.split("_")[0]
    try:
        ast.parse(s)
        # ast.parse 不做“return 在函数外”之类的符号表检查，只有 compile 才会；
        # 评分机用的是 compile，所以这里必须用 compile 才能提前拦住（211 曾因此整批 failed）。
        compile(s, p.name, "exec")
    except SyntaxError as e:
        return ["语法错误：%s" % e], []
    if FIX930 not in s and ZERO930 not in s and num not in CONTROL:
        errs.append("09:30 既没修复也没显式置 0")
    # 本地行情只有 3 档盘口：4–5 档在本地永远是 NULL（COALESCE 成 0）。
    # 评测机会因此给告警「SQL 用到了 4–5 档盘口；结果与平台不同」：本地测的其实是 1–3 档版。
    lv45 = set(re.findall(r"(?:bid|ask)_(?:price|volume)([45])", s))
    lv13 = set(re.findall(r"(?:bid|ask)_(?:price|volume)([123])", s))
    if lv45:
        if not lv13:
            errs.append("只用了 4–5 档盘口，本地全为 NULL，信号在评测机上会退化成常数")
        else:
            warns.append("用到 4–5 档盘口：本地为 NULL，评测机结果等价于 1–3 档版，与平台（全 5 档）不同")
    # 禁用表：vwap 表是标签窗口的 VWAP，仍禁止。
    if "cpt_jyc_2026_vwap" in s:
        errs.append("读取了禁用表 cpt_jyc_2026_vwap（只能用 bar1m、股票池表、以及错位后的暴露表）")
    # 暴露表（2026-10-12 晚解禁）：第 t 天的市值/风格按当天 15:00 收盘计算，是未来数据，
    # 必须按交易日错一天（取严格早于截面日期的最近一条）。评分机只给 data_notice，错位由代码保证。
    if "cpt_jyc_2026_exposure" in s or 'datasources.get("exposure"' in s or "datasources.get('exposure'" in s:
        _legacy = num.isdigit() and int(num) < 338   # 270–337 用当天值，已被 store/excluded.json 排除，冻结不改
        _dst = warns if _legacy else errs
        if "# 暴露表错位" not in s and "前一交易日" not in s:
            _dst.append("读暴露表但没有注明错位方式（代码里要有 '# 暴露表错位' 或『前一交易日』说明）")
        if not re.search(r"shift\(1\)", s):
            _dst.append("读暴露表但没有看到按交易日错位（shift(1)）")
        if re.search(r'"ret"', s) and "前一交易日" not in s and "# 暴露表错位" not in s:
            errs.append("读了暴露表的 ret 列，必须是前一天及更早的值")
    # keep 列表必须覆盖阶梯用到的尺度列
    mk = re.search(r"    keep = \[(.*?)\]", s, re.S)
    for c in SCALES:
        if 'df["%s"]' % c in s:
            if not mk or '"%s"' % c not in mk.group(1):
                errs.append("keep 缺列 %s（合并后会 KeyError）" % c)
    # keep 里的列必须在 df 里真的存在（df[...] = ... 赋值过，或来自 SQL 的透传/清零列），否则 merge 直接 KeyError。
    # 447 曾因此整批 failed：keep 写了 r_30_ind 等，但那四列是在 result 上才创建的。
    if mk:
        sql_cols = set()
        for _m in re.finditer(r"for c in \[(.*?)\]:", s, re.S):
            sql_cols |= set(re.findall(r'"([a-z0-9_]+)"', _m.group(1)))
        df_cols = set(re.findall(r'df\["([A-Za-z_][A-Za-z0-9_]*)"\]\s*=', s)) | sql_cols
        prefixes = set(re.findall(r'df\["([A-Za-z_][A-Za-z0-9_]*)"\s*\+', s))
        for c in set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', mk.group(1))):
            if c not in df_cols and not any(c.startswith(x) for x in prefixes):
                errs.append("keep 里有 df 中不存在的列 %s（merge 会 KeyError）" % c)
    # result[...] 读到的列必须来自 keep，或在同一段里被赋值过
    if mk:
        kept = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', mk.group(1)))
        assigned = set(re.findall(r'result\["([A-Za-z_][A-Za-z0-9_]*)"\]\s*=', s))
        for c in set(re.findall(r'result\["([A-Za-z_][A-Za-z0-9_]*)"\]', s)):
            if c not in kept and c not in assigned and c not in ("date", "instrument"):
                errs.append("result 读了未 keep 的列 %s" % c)
    # SQL 三级透传：Window 聚合里 SUM/MAX 出来的列名必须在最终 SELECT 与过滤循环里出现
    win = re.search(r"    cte_window AS \((.*?)\n    \)", s, re.S)
    if win:
        al = set(re.findall(r"AS ([a-z][a-z0-9_]*)", win.group(1)))
        sel = re.search(r"\n    SELECT\n(.*?)\n    FROM cte_window f", s, re.S)
        selbody = sel.group(1) if sel else ""
        flt = re.search(r"for c in \[(.*?)\]:", s, re.S)
        fltbody = flt.group(1) if flt else ""
        for c in sorted(al):
            if c in ("date", "instrument", "trading_day", "time_segment", "hms", "pre_close",
                     "last_px", "sum_amt", "sum_vol"):
                continue
            if ("f.%s" % c) not in selbody:
                errs.append("SQL 列 %s 未在最终 SELECT 透传" % c)
            elif ('"%s"' % c) not in fltbody and not c.endswith("_c"):
                errs.append("SQL 列 %s 未在过滤循环里清零" % c)
    # rank_blend 列表里引用的列必须来自 keep 或本段赋值（399-402 曾因此整批 KeyError）
    if mk:
        assigned0 = set(re.findall(r'result\["([A-Za-z_][A-Za-z0-9_]*)"\]\s*=', s))
        for rb in re.findall(r"rank_blend\(\[(.*?)\]", s, re.S):
            for c in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', rb):
                if c not in kept and c not in assigned0:
                    errs.append("rank_blend 引用了未 keep 的列 %s" % c)
    # 片段引用的 df 列必须来自 SQL 别名或前面定义
    body = s[s.index("def main("):]
    defined = set(re.findall(r'df\["([a-z0-9_]+)"\]\s*=', body)) | set(re.findall(r"AS ([a-z0-9_]+)", s))
    for c in set(re.findall(r'df\["([a-z0-9_]+)"\]', body)):
        if c.startswith("c_") or c in ("date", "instrument", "day", "hms"):
            continue
        if c not in defined:
            errs.append("片段引用了不存在的列 %s" % c)
    return errs, warns


def main():
    args = sys.argv[1:]
    if args:
        paths = [pathlib.Path(a) for a in args]
    else:
        paths = [p for p in sorted((ROOT / "因子代码").glob("*.py")) if p.name.split("_")[0].isdigit() and int(p.name.split("_")[0]) >= 113]
    bad = 0
    warned = 0
    for p in paths:
        errs, warns = check(p)
        if errs:
            bad += 1
            print("✗ %s：%s" % (p.name, "；".join(errs)))
        for w in warns:
            warned += 1
            print("! %s：%s" % (p.name, w))
    print("自检完成：%d 个文件，%d 个有问题，%d 条提示" % (len(paths), bad, warned))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
