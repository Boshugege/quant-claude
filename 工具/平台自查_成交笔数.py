# -*- coding: utf-8 -*-
"""平台自查：分钟表的成交笔数（deal_number）是否为空或为 0。在平台 notebook 里整段运行。
背景：凡是用到"单笔成交额 = 成交额 ÷ 成交笔数"的根（1367 近 5 日大单方向等）在公榜上失效：
单独版报错，合成里贡献为零。只用收益与成交量的根全部正常。怀疑平台数据的 deal_number 为空或为 0。
运行约 1～3 分钟；只读数据，不提交任何东西。
"""
import dai
import pandas as pd

T = "cpt_jyc_2026_stock_bar1m"

# 1. 表里数据的时间范围
rng = dai.query(f"SELECT MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS n_rows FROM {T}",
                filters={"date": ["2019-01-01 00:00:00", "2030-12-31 23:59:59"]}).df()
print("== 数据范围\n", rng.to_string(index=False))

# 2. 按月统计 deal_number 的空值、为 0、为正的比例（只看有成交量的分钟）
sql = f"""
SELECT strftime(date, '%Y-%m') AS ym,
       COUNT(*) AS n_rows,
       AVG(CASE WHEN volume > 0 THEN 1.0 ELSE 0.0 END) AS share_vol_pos,
       AVG(CASE WHEN volume > 0 AND deal_number IS NULL THEN 1.0 ELSE 0.0 END) AS share_vol_but_null,
       AVG(CASE WHEN volume > 0 AND deal_number = 0 THEN 1.0 ELSE 0.0 END) AS share_vol_but_zero,
       AVG(CASE WHEN volume > 0 AND deal_number > 0 THEN 1.0 ELSE 0.0 END) AS share_vol_and_deal,
       AVG(CASE WHEN deal_number > 0 AND amount > 0 THEN amount / deal_number END) AS avg_trade_amount
FROM {T}
GROUP BY 1 ORDER BY 1
"""
stat = dai.query(sql, filters={"date": ["2019-01-01 00:00:00", "2030-12-31 23:59:59"]}).df()
pd.set_option("display.width", 200)
print("\n== 按月统计（只看 volume > 0 的分钟；share_vol_and_deal 应接近 share_vol_pos）\n", stat.to_string(index=False))

# 3. 最近一个交易日的样例
last = pd.to_datetime(rng["last_date"].iloc[0])
d0 = last.normalize().strftime("%Y-%m-%d 00:00:00")
d1 = last.normalize().strftime("%Y-%m-%d 23:59:59")
smp = dai.query(f"SELECT date, instrument_id, close, volume, amount, deal_number FROM {T} ORDER BY instrument_id, date LIMIT 20",
                filters={"date": [d0, d1]}).df()
print("\n== 最近一个交易日的 20 行样例\n", smp.to_string(index=False))

print("\n判读：若最近月份的 share_vol_but_null 或 share_vol_but_zero 接近 1（或 avg_trade_amount 为空），"
      "说明平台数据缺成交笔数，单笔成交额类根在平台上全部失效；若与 2020–2024 各月一样正常，则原因另有所在。")
