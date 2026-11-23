# -*- coding: utf-8 -*-
"""第十八轮 N1（更正版）：第三个去成交笔数合成版 —— −0.10 槽位用与已有分量重叠最小的 X_q5d。"""
import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import n1c_build_vol5d as B   # noqa: E402

FD = pathlib.Path("/home/user/quant-claude/因子代码")
W_LOW = ('    W = {"L_lab10": +0.20, "M_q_lin": +0.10, "W_vdir": -0.10, "W_zero": +0.10, '
         '"Y_vsd5": +0.10, "X_q5d": -0.10, "Z_mfr60": -0.15, "Z_mfr90": -0.10}')


def main():
    txt = (FD / "1363_合成头部加九根含六十九十分钟资金流精简按月头部按季.py").read_text(encoding="utf-8")
    t = B.add_cols(txt).replace(B.OLD_W, W_LOW)
    t = t.replace("1363】合成头部加九根含六十九十分钟资金流精简按月头部按季",
                  "1760】合成头部加八根去成交笔数（−0.10 槽位改用与 W_vdir 重叠最小的 X_q5d）")
    ast.parse(t)
    (FD / "1760_合成头部加八根去成交笔数低重叠版.py").write_text(t, encoding="utf-8")
    print("OK 1760_合成头部加八根去成交笔数低重叠版.py", len(t), "字节")


if __name__ == "__main__":
    main()
