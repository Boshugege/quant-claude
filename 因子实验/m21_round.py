# -*- coding: utf-8 -*-
"""纯方向深挖 J9（10-20，personal agent.slice）：整数关口（行为锚定）。截面 t 的收盘价（元）相对整数价位的位置：
  RN_FRAC   小数部分 − 0.5（价格在两个整数之间的位置；接近 +0.5 = 刚低于上方整数关口）
  RN_UP     到上方整数关口的距离 ÷ 价格（相对距离；小 = 快要碰到上方关口）
  RN_DN     到下方整数关口的距离 ÷ 价格
  RN_NEAR   到最近整数关口的相对距离（越小越"卡在关口上"）
  RN_TEN    到最近的整十元关口的相对距离
价格单位为分（整数），只用截面收盘价。保存 vc_tmp/h1_RN_*.npy（原始值）。"""
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

t = pd.read_parquet(H1.CACHE); t["day"] = pd.to_datetime(t["day"])
W = H1.wide(t); del t
C = W["close"][H1.CS].astype(float) / 100.0                                  # 元
C = C.where(C > 0)
fl = np.floor(C); fr = C - fl
S = {"RN_FRAC": fr - 0.5, "RN_UP": (fl + 1 - C) / C, "RN_DN": fr / C,
     "RN_NEAR": np.minimum(fr, 1 - fr) / C, "RN_TEN": np.minimum(C % 10, 10 - C % 10) / C}
c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
for nm, fr_ in S.items():
    v = H1.align(c, fr_)
    v = np.where(np.isfinite(v) & (v == 0), 1e-9, v); v[z930] = 0.0
    np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(v, nan=0.0))
    print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(v) != 0).mean()), flush=True)
