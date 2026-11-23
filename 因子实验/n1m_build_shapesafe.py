# -*- coding: utf-8 -*-
"""第十八轮 N1（更正版）补：**形状安全版** 1761。

背景（与 deal_number 假设高度混淆的第二个解释）：
公榜证据「1361 与 1360 五项完全相同」只说明 **Y_sd5 与 X_n5d 无效**——
注意 1360 里 **本来就已经有 N_dir**（W = L_lab10/M_q_lin/N_dir/W_vdir/W_zero），
1361 只是在它上面再加 Y_sd5(+0.10) 与 X_n5d(−0.10)。所以"1361≡1360"**证明不了 N_dir 是死的**。
能同时解释这一现象的还有第二个假设：**平台在跨日 w5 窗口里用到 STDDEV_SAMP / 窗口内带 LAG 的乘积时不工作**
（1387 的 w5 全是 SUM/COUNT，所以它正常出分；Y_sd5 是 STDDEV_SAMP OVER w5、X_n5d 是 SUM(r·(lats−lats_prev)) OVER w5）。

两个假设对 1752 有完全不同的后果：1752 的两个替代根恰好**继承了同样的形状**——
  Y_vsd5 = STDDEV_SAMP(lv) OVER w5      （与 Y_sd5 同形）
  Y_vd5  = SUM(r·(lv−lv_prev)) OVER w5  （与 X_n5d 同形）
即：若平台坏的是"形状"，1752 会像 1363 一样不兑现。

1761 = 只用**已被 1387 在公榜上验证过的形状**（纯 SUM / 纯 COUNT，无 STDDEV、无窗口内 LAG）：
  Y_z5   = COUNT(r=0) OVER w5 ÷ COUNT(r) OVER w5      （替代 Y_sd5，+0.10）
  X_q5d  = SUM(r) OVER w5 ÷ SUM(|r|) OVER w5          （替代 X_n5d，−0.10）
  N_dir 仍删除。其余六根与 1753 完全一致。
"""
import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import n1c_build_vol5d as B   # noqa: E402

FD = pathlib.Path("/home/user/quant-claude/因子代码")
W_SHAPE = ('    W = {"L_lab10": +0.20, "M_q_lin": +0.10, "W_vdir": -0.10, "W_zero": +0.10, '
           '"Y_z5": +0.10, "X_q5d": -0.10, "Z_mfr60": -0.15, "Z_mfr90": -0.10}')


def main():
    txt = (FD / "1363_合成头部加九根含六十九十分钟资金流精简按月头部按季.py").read_text(encoding="utf-8")
    t = B.add_cols(txt).replace(B.OLD_W, W_SHAPE)
    t = t.replace("1363】合成头部加九根含六十九十分钟资金流精简按月头部按季",
                  "1761】合成头部加八根去成交笔数形状安全版（纯 SUM/COUNT，无 STDDEV、无窗口内 LAG）")
    ast.parse(t)
    (FD / "1761_合成头部加八根去成交笔数形状安全版.py").write_text(t, encoding="utf-8")
    print("OK 1761_合成头部加八根去成交笔数形状安全版.py", len(t), "字节")


if __name__ == "__main__":
    main()
