# -*- coding: utf-8 -*-
"""第十八轮 N1（更正版）：把去 deal_number 的近 5 日根与合成版交到评分机。

  * 六个近 5 日根（1750/1751/1754/1755/1756/1757）：主区间，看单根有几分、有没有退化；
  * 1752（1363 去成交笔数 + 两根替代）、1753（不含两根替代的对照）：主区间 + o21 + o20。
1752 与 1753 的差就是"两根新根恢复了多少贡献"。
"""
import subprocess
import sys

URL = "http://127.0.0.1:46220/api/jobs"
FD = "/home/user/quant-claude/因子代码/"
ROOTS = ["1750_近五日量价同向", "1751_近五日分钟量离散度", "1754_近五日带符号路径效率",
         "1755_近五日零收益分钟占比", "1756_近五日上涨分钟成交量占比", "1757_近五日净流向占比"]
COMPS = {"1752_合成头部加八根去成交笔数": [("", "2023-01", "2024-12"),
                                        ("o21_", "2021-01", "2022-12"),
                                        ("o20_", "2020-01", "2020-12")],
         "1758_合成头部加八根去成交笔数零收益版": [("", "2023-01", "2024-12"),
                                             ("o21_", "2021-01", "2022-12"),
                                             ("o20_", "2020-01", "2020-12")],
         "1753_合成头部加六根去成交笔数对照": [("", "2023-01", "2024-12"),
                                          ("o21_", "2021-01", "2022-12"),
                                          ("o20_", "2020-01", "2020-12")]}


def post(f, name, start, end):
    r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "start=%s" % start,
                        "-F", "end=%s" % end, "-F", "kind=platform", "-F", "name=%s" % name,
                        "-F", "baseline=ref:104", URL], capture_output=True, text=True)
    out = (r.stdout or r.stderr).strip().replace("\n", " ")
    print("  %-30s %s" % (name, out[:120]), flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "roots"):
        for nm in ROOTS:
            post(FD + nm + ".py", "r_" + nm.split("_")[0], "2023-01", "2024-12")
    if which in ("all", "comps"):
        for nm, wins in COMPS.items():
            for pre, s7, e7 in wins:
                post(FD + nm + ".py", pre + nm.split("_")[0], s7, e7)


if __name__ == "__main__":
    main()
