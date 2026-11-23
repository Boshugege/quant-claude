def main(datasources, start_date, end_date):
    """
    多层逐档订单流不平衡（Multi-Level OFI，Cont-Kukanov-Stoikov 2014 / Xu-Gould-Howison 2019）。

    经济含义：用相邻两分钟的盘口快照还原每一档的“净新增买压”：
      买 i 档：价上移 -> +当前量；价不变 -> +量变化；价下移 -> -上一分钟量
      卖 i 档：价下移 -> +当前量；价不变 -> +量变化；价上移 -> -上一分钟量
      OFI_i = 买方贡献 - 卖方贡献
    30 分钟内逐档累加 OFI_i，除以该档 30 分钟平均双边深度，做 x/(1+|x|) 压缩后五档等权平均。
    可能有效的理由：静态盘口只看“存量”，OFI 看的是挂单、撤单、成交共同造成的“流量”，
    文献中多层 OFI 对价格的解释力显著强于只看最优档。逐档归一化沿用了 15 号成功的做法。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]
    levels = range(1, 6)

    raw_cols = ",\n            ".join(
        [f"CAST(COALESCE(bid_price{i}, 0) AS DOUBLE) AS bp{i}" for i in levels]
        + [f"CAST(COALESCE(ask_price{i}, 0) AS DOUBLE) AS ap{i}" for i in levels]
        + [f"CAST(COALESCE(bid_volume{i}, 0) AS DOUBLE) AS bv{i}" for i in levels]
        + [f"CAST(COALESCE(ask_volume{i}, 0) AS DOUBLE) AS av{i}" for i in levels]
    )
    lag_cols = ",\n            ".join(
        f"LAG({c}{i}) OVER w AS p{c}{i}" for c in ("bp", "ap", "bv", "av") for i in levels
    )
    ofi_cols = ",\n            ".join(
        f"""CASE WHEN ok THEN
                (CASE WHEN bp{i} > pbp{i} THEN bv{i} WHEN bp{i} = pbp{i} THEN bv{i} - pbv{i} ELSE -pbv{i} END)
              - (CASE WHEN ap{i} < pap{i} THEN av{i} WHEN ap{i} = pap{i} THEN av{i} - pav{i} ELSE -pav{i} END)
            END AS ofi{i},
            CASE WHEN ok THEN bv{i} + av{i} END AS dep{i}"""
        for i in levels
    )
    win_cols = ",\n            ".join(
        f"SUM(ofi{i}) / (AVG(dep{i}) + 1.0) AS o{i}" for i in levels
    )

    sql = f"""
    WITH raw AS (
        SELECT
            instrument_id,
            CAST(CAST(date AS TIMESTAMP) AS DATE) AS trading_day,
            hour(date) * 10000 + minute(date) * 100 + second(date) AS hms,
            {raw_cols}
        FROM {bar1m}
    ),
    lagged AS (
        SELECT
            *,
            {lag_cols}
        FROM raw
        WINDOW w AS (PARTITION BY instrument_id, trading_day ORDER BY hms)
    ),
    seg AS (
        SELECT
            *,
            (bp1 > 0 AND ap1 > bp1 AND pbp1 > 0 AND pap1 > pbp1) AS ok,
            CASE
                WHEN hms > 90000  AND hms <= 93000  THEN 93000
                WHEN hms > 93000  AND hms <= 100000 THEN 100000
                WHEN hms > 100000 AND hms <= 103000 THEN 103000
                WHEN hms > 103000 AND hms <= 110000 THEN 110000
                WHEN hms > 110000 AND hms <= 113000 THEN 113000
                WHEN hms > 130000 AND hms <= 133000 THEN 133000
                WHEN hms > 133000 AND hms <= 140000 THEN 140000
                WHEN hms > 140000 AND hms <= 143000 THEN 143000
                ELSE -1
            END AS time_segment
        FROM lagged
    ),
    snap AS (
        SELECT
            trading_day, time_segment, instrument_id,
            {ofi_cols}
        FROM seg
        WHERE time_segment > 0
    ),
    win AS (
        SELECT
            trading_day, time_segment, instrument_id,
            {win_cols}
        FROM snap
        GROUP BY trading_day, time_segment, instrument_id
    )
    SELECT w.*, i.instrument
    FROM win w
    LEFT JOIN all_instruments i USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()

    seg = df["time_segment"].astype(int)
    df["date"] = (
        pd.to_datetime(df["trading_day"])
        + pd.to_timedelta(seg // 10000, unit="h")
        + pd.to_timedelta(seg // 100 % 100, unit="m")
    )
    df["instrument"] = df["instrument"].astype(str)
    ofi = df[[f"o{i}" for i in levels]].astype("float64").replace([np.inf, -np.inf], np.nan)
    df["factor"] = (ofi / (1.0 + ofi.abs())).mean(axis=1, skipna=True)

    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    stk_pool["date"] = pd.to_datetime(stk_pool["date"])
    stk_pool["instrument"] = stk_pool["instrument"].astype(str)

    result = pd.merge(df[["date", "instrument", "factor"]], stk_pool, how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"])
    # 当日第一根 bar 没有上一分钟盘口，若某截面只有这一根会整段缺失；用中性值 0 补齐，避免截面缺失率超限。
    result["factor"] = result["factor"].fillna(0.0)
    return result[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


if __name__ == "__main__":
    from bigmodule import M
    import structlog

    logger = structlog.get_logger()
    datasources = {"bar1m": "cpt_jyc_2026_stock_bar1m"}
    start_date = "2020-01-01 00:00:00"
    end_date = "2020-03-01 23:59:59"
    logger.info(f"计算因子，区间：{start_date} ~ {end_date}")
    factor_data = main(datasources, start_date, end_date)
    result = M.jyc_eval._latest(factor_data=factor_data, show=True)
