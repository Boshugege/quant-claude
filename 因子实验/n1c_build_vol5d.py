# -*- coding: utf-8 -*-
"""第十八轮 N1（更正版）①：用「收益 + 成交量」重写近 5 日类根，并生成去成交笔数版合成。

为什么：公榜数据里 deal_number 很可能为空/全 0 —— 1367（近五日大单方向）报错、1361 里
Y_sd5/X_n5d 贡献为零，共同点只有「用成交额÷成交笔数」。跨日 1200 分钟窗口本身没问题
（1387/1369/1390 都用 w5 且正常出分）。

替换规则（结构一一对应，只把「单笔成交额 lats=ln(成交额÷笔数)」换成「成交量 lv=ln(成交量)」）：
  X_n5d (近5日大单方向, Σr·(lats−之前均值)/Σ|r|) → Y_vd5 (近5日量价同向, 把 lats 换成 lv)
  Y_sd5 (近5日单笔成交额离散度, STDDEV(lats))      → Y_vsd5 (近5日分钟成交量对数离散度, STDDEV(lv))
  N_dir (当日大单方向)                             → 直接删除（当日量价同向 W_vdir 已在组合里）
并在同一段精简 SQL 里顺带算出 Z_mfr5d / Z_net5d / Y_z5（只用 volume，N3 已验证）。

生成（SQL 文本完全一致，评测机只执行一次分钟 SQL）：
  因子代码/1750_近五日量价同向.py            STAT=Y_vd5   独立根
  因子代码/1751_近五日分钟量离散度.py        STAT=Y_vsd5  独立根
  因子代码/1752_合成头部加八根去成交笔数.py  新合成（1363 的替换版）
  因子代码/1753_合成头部加六根去成交笔数与近五日根.py  对照：不含两根新根
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
FD = ROOT / "因子代码"
SRC = FD / "1363_合成头部加九根含六十九十分钟资金流精简按月头部按季.py"

OLD_COLS = ('SUM(r * (lv - lv_prev)) OVER w AS cvd, SUM(CASE WHEN r = 0 THEN 1 ELSE 0 END) OVER w AS czero, '
            'COUNT(r) OVER w AS cnr, SUM(ABS(r)) OVER w5 AS a5d, SUM(r * (lats - lats_prev)) OVER w5 AS nd5, '
            'STDDEV_SAMP(lats) OVER w5 AS sd5,')
NEW_COLS = OLD_COLS + """
                SUM(r * (lv - lv_prev)) OVER w5 AS vd5, STDDEV_SAMP(lv) OVER w5 AS sdv5,
                SUM(v) OVER w5 AS v_5, SUM(r) OVER w5 AS r5d,
                SUM(CASE WHEN r > 0 THEN v ELSE 0 END) OVER w5 AS vup5, SUM(CASE WHEN r < 0 THEN v ELSE 0 END) OVER w5 AS vdn5,
                SUM(CASE WHEN r = 0 THEN 1 ELSE 0 END) OVER w5 AS z5, COUNT(r) OVER w5 AS n5,"""

OLD_LAM = '                "Y_sd5": lambda: df["sd5"],'
NEW_LAM = OLD_LAM + """
                "Y_vsd5": lambda: df["sdv5"],
                "Z_mfr5d": lambda: df["vup5"] / df["v_5"].where(df["v_5"] > 0),
                "Z_net5d": lambda: (df["vup5"] - df["vdn5"]) / df["v_5"].where(df["v_5"] > 0),"""

OLD_W = '    W = {"L_lab10": +0.20, "M_q_lin": +0.10, "N_dir": -0.10, "W_vdir": -0.10, "W_zero": +0.10, "Y_sd5": +0.10, "X_n5d": -0.10, "Z_mfr60": -0.15, "Z_mfr90": -0.10}'
W_FULL = '    W = {"L_lab10": +0.20, "M_q_lin": +0.10, "W_vdir": -0.10, "W_zero": +0.10, "Y_vsd5": +0.10, "Y_vd5": -0.10, "Z_mfr60": -0.15, "Z_mfr90": -0.10}'
W_ALT = '    W = {"L_lab10": +0.20, "M_q_lin": +0.10, "W_vdir": -0.10, "W_zero": +0.10, "Y_z5": +0.10, "Y_vd5": -0.10, "Z_mfr60": -0.15, "Z_mfr90": -0.10}'
W_CTRL = '    W = {"L_lab10": +0.20, "M_q_lin": +0.10, "W_vdir": -0.10, "W_zero": +0.10, "Z_mfr60": -0.15, "Z_mfr90": -0.10}'


def add_cols(txt):
    assert OLD_COLS in txt, "找不到 w5 列定义"
    txt = txt.replace(OLD_COLS, NEW_COLS)
    assert OLD_LAM in txt, "找不到 Y_sd5 lambda"
    return txt.replace(OLD_LAM, NEW_LAM)


def split_root(txt):
    """返回 (root_impl 代码块, 其后内容)。"""
    i = txt.find("def _root_impl(")
    j = txt.find("\n\nif __name__", i)
    return txt[i:j], txt[j:]


def standalone(txt, stat, note):
    root, tail = split_root(txt)
    root = add_cols(root)
    root = root.replace("def _root_impl(datasources, start_date, end_date, STAT, _cache):",
                        "def main(datasources, start_date, end_date):")
    root = root.replace('    SLOTS = ["1000"', '    STAT = "%s"          # %s\n    _cache = {}\n    SLOTS = ["1000"' % (stat, note))
    return root + tail


def main():
    txt = SRC.read_text(encoding="utf-8")          # 原始 1363 文本；下面每条路径各自 add_cols 一次

    f1750 = standalone(txt, "Y_vd5", "近5日量价同向（去 deal_number）")
    (FD / "1750_近五日量价同向.py").write_text(f1750, encoding="utf-8")

    f1751 = standalone(txt, "Y_vsd5", "近5日分钟成交量对数离散度（去 deal_number）")
    (FD / "1751_近五日分钟量离散度.py").write_text(f1751, encoding="utf-8")

    # 同一段 SQL 的其他「近 5 日」候选根（全部只用收益/成交量），评测机只执行一次 SQL
    for stat, name in [("X_q5d", "1754_近五日带符号路径效率.py"),
                       ("Y_z5", "1755_近五日零收益分钟占比.py"),
                       ("Z_mfr5d", "1756_近五日上涨分钟成交量占比.py"),
                       ("Z_net5d", "1757_近五日净流向占比.py")]:
        (FD / name).write_text(standalone(txt, stat, name), encoding="utf-8")

    comp = add_cols(txt).replace(OLD_W, W_FULL)
    assert comp != txt, "找不到 W 定义"
    comp = comp.replace("1363】合成头部加九根含六十九十分钟资金流精简按月头部按季",
                        "1752】合成头部加八根去成交笔数（近 5 日根改用收益+成交量）")
    comp = comp.replace("权重：{\"L_lab10\": +0.20, \"M_q_lin\": +0.10, \"N_dir\": -0.10, \"W_vdir\": -0.10, \"W_zero\": +0.10, \"Y_sd5\": +0.10, \"X_n5d\": -0.10, \"Z_mfr60\": -0.15, \"Z_mfr90\": -0.10}",
                        "权重：{\"L_lab10\": +0.20, \"M_q_lin\": +0.10, \"W_vdir\": -0.10, \"W_zero\": +0.10, \"Y_vsd5\": +0.10, \"Y_vd5\": -0.10, \"Z_mfr60\": -0.15, \"Z_mfr90\": -0.10}（N_dir/Y_sd5/X_n5d 三个用成交笔数的根已去掉）")
    (FD / "1752_合成头部加八根去成交笔数.py").write_text(comp, encoding="utf-8")

    alt = add_cols(txt).replace(OLD_W, W_ALT)
    assert alt != txt
    alt = alt.replace("1363】合成头部加九根含六十九十分钟资金流精简按月头部按季",
                      "1758】合成头部加八根去成交笔数（+0.10 槽位改用近 5 日零收益分钟占比）")
    (FD / "1758_合成头部加八根去成交笔数零收益版.py").write_text(alt, encoding="utf-8")

    ctrl = add_cols(txt).replace(OLD_W, W_CTRL)
    assert ctrl != txt
    ctrl = ctrl.replace("1363】合成头部加九根含六十九十分钟资金流精简按月头部按季",
                        "1753】对照组：只有六根、不含两根新近 5 日根（用于量出新增贡献）")
    (FD / "1753_合成头部加六根去成交笔数对照.py").write_text(ctrl, encoding="utf-8")

    for f in ["1750_近五日量价同向.py", "1751_近五日分钟量离散度.py",
              "1752_合成头部加八根去成交笔数.py", "1753_合成头部加六根去成交笔数对照.py",
              "1754_近五日带符号路径效率.py", "1755_近五日零收益分钟占比.py",
              "1756_近五日上涨分钟成交量占比.py", "1757_近五日净流向占比.py",
              "1758_合成头部加八根去成交笔数零收益版.py"]:
        t = (FD / f).read_text(encoding="utf-8")
        import ast
        ast.parse(t)
        print("OK %-40s %6d 字节" % (f, len(t)))
    sql0 = re.search(r'sql_min = f"""(.*?)"""', (FD / "1750_近五日量价同向.py").read_text(), re.S).group(1)
    sql1 = re.search(r'sql_min = f"""(.*?)"""', (FD / "1751_近五日分钟量离散度.py").read_text(), re.S).group(1)
    sql2 = re.search(r'sql_min = f"""(.*?)"""', (FD / "1752_合成头部加八根去成交笔数.py").read_text(), re.S).group(1)
    print("SQL 三份完全一致：", sql0 == sql1 == sql2)
    alld = sorted(FD.glob("17[5][0-9]_*.py"))
    print("1750-1759 生成：", [p.name for p in alld])


if __name__ == "__main__":
    main()
