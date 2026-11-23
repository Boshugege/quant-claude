# -*- coding: utf-8 -*-
"""第十八轮 N1（更正版）：把头部候选 1602（=1398，队友161头部 + 我们九根 + 三小根）改成去 deal_number 版。

**v2 修正（2026-10-18 00:40）**：v1 只把 STAT 字符串从 Y_sd5/X_n5d 改成 Y_z5/Y_vd5，但 1602 里的
`_root_impl_b6` / `_root_impl_b7` 各自只有一条 STAT 的 lambda（b6 只有 `Y_sd5`、b7 只有 `X_n5d`），
SQL 里也没有 `z5/n5/vd5` 列 —— 三个任务全部 `KeyError: 'Y_z5'` 失败（任务 1017-223416-46db/31ec/e27e）。
v2 改成**连 SQL 列和 lambda 一起补**，并且顺手对齐 §140 的"形状安全"选择（纯 SUM / 纯 COUNT，无 STDDEV、无窗口内 LAG）：

改动（其余逐字节不动）：
  * 去掉 B3（`N_dir` 大单方向，权重 −0.1）：平台上因 deal_number 为空本来就是 0；
  * B6：SQL 加 `SUM(CASE WHEN r = 0 THEN 1 ELSE 0 END) OVER w5 AS z5, COUNT(r) OVER w5 AS n5`，
        STAT 从 `Y_sd5` 改成 `Y_z5`（近 5 日零收益分钟占比），权重仍 +0.1；
  * B7：SQL 加 `SUM(r) OVER w5 AS r5d`，STAT 从 `X_n5d` 改成 `X_q5d`（近 5 日带符号路径效率），权重仍 −0.1。

输出：因子代码/1759_队友161头部加去成交笔数八根.py
"""
import ast
import pathlib

FD = pathlib.Path(__file__).resolve().parent.parent / "因子代码"
SRC = FD / "1602_队友161头部加我们九根及三小根头部按季.py"
OUT = FD / "1759_队友161头部加去成交笔数八根.py"

S6_OLD = "SUM(ABS(r)) OVER w AS cabs, STDDEV_SAMP(lats) OVER w5 AS sd5"
S6_NEW = ("SUM(ABS(r)) OVER w AS cabs, STDDEV_SAMP(lats) OVER w5 AS sd5, "
          "SUM(CASE WHEN r = 0 THEN 1 ELSE 0 END) OVER w5 AS z5, COUNT(r) OVER w5 AS n5")
L6_OLD = '                "Y_sd5": lambda: df["sd5"],'
L6_NEW = ('                "Y_sd5": lambda: df["sd5"],\n'
          '                "Y_z5": lambda: df["z5"] / df["n5"].where(df["n5"] > 0),')

S7_OLD = "SUM(ABS(r)) OVER w AS cabs, SUM(ABS(r)) OVER w5 AS a5d, SUM(r * (lats - lats_prev)) OVER w5 AS nd5"
S7_NEW = S7_OLD + ", SUM(r) OVER w5 AS r5d"
L7_OLD = '                "X_n5d": lambda: df["nd5"] / df["a5d"].where(df["a5d"] > 0),'
L7_NEW = (L7_OLD + '\n'
          '                "X_q5d": lambda: df["r5d"] / df["a5d"].where(df["a5d"] > 0),')


def main():
    t = SRC.read_text(encoding="utf-8")
    assert t.count(S6_OLD) == 1 and t.count(S7_OLD) == 1, "SQL 片段不唯一"
    assert t.count(L6_OLD) == 1 and t.count(L7_OLD) == 1, "lambda 片段不唯一"
    t = t.replace("        (main_b3, -0.1, 0.0, {}),\n", "", 1)
    t = t.replace(S6_OLD, S6_NEW, 1).replace(L6_OLD, L6_NEW, 1)
    t = t.replace(S7_OLD, S7_NEW, 1).replace(L7_OLD, L7_NEW, 1)
    t = t.replace('"Y_sd5")', '"Y_z5")', 1).replace('"X_n5d")', '"X_q5d")', 1)
    t = t.replace("【1602：队友161头部加我们九根及三小根头部按季】",
                  "【1759：队友161头部加去成交笔数八根】1602 的去 deal_number 版：去掉 B3（N_dir），"
                  "B6 改 Y_z5、B7 改 X_q5d（都只读收益/成交量，且都是纯 SUM/COUNT 形状）")
    t = t.replace('"""【1602', '"""【1759')
    assert "(main_b3, -0.1, 0.0, {})," not in t, "B3 没删掉"
    assert '"Y_z5")' in t and '"X_q5d")' in t
    ast.parse(t)
    compile(t, str(OUT), "exec")
    OUT.write_text(t, encoding="utf-8")
    print("OK", OUT.name, len(t), "字节；b3 已删、b6=Y_z5、b7=X_q5d（SQL 各补 1–2 列）")


if __name__ == "__main__":
    main()
