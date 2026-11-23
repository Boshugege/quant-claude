def main(datasources, start_date, end_date):
    """
    多层微价格位移。

    经济含义：将五档价格和双边挂单量共同映射为多层 microprice，再按一档价差归一化其相对中间价的位移。
    可能有效的理由：队列不平衡文献主要研究最优价；多层 microprice 把更深档的价格距离也纳入，直接形成对未来 VWAP 方向的价格型盘口信号。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    def minute_multilevel_microprice(
        ask_price1: float,
        ask_price2: float,
        ask_price3: float,
        ask_price4: float,
        ask_price5: float,
        bid_price1: float,
        bid_price2: float,
        bid_price3: float,
        bid_price4: float,
        bid_price5: float,
        ask_volume1: float,
        ask_volume2: float,
        ask_volume3: float,
        ask_volume4: float,
        ask_volume5: float,
        bid_volume1: float,
        bid_volume2: float,
        bid_volume3: float,
        bid_volume4: float,
        bid_volume5: float,
    ) -> float:
        """双边五档深度交叉加权的 microprice 相对一档中间价位移。"""
        ask_p = (ask_price1, ask_price2, ask_price3, ask_price4, ask_price5)
        bid_p = (bid_price1, bid_price2, bid_price3, bid_price4, bid_price5)
        ask_v = (ask_volume1, ask_volume2, ask_volume3, ask_volume4, ask_volume5)
        bid_v = (bid_volume1, bid_volume2, bid_volume3, bid_volume4, bid_volume5)
        total = sum(ask_v) + sum(bid_v)
        cross_value = sum(p * v for p, v in zip(ask_p, bid_v))
        cross_value += sum(p * v for p, v in zip(bid_p, ask_v))
        micro = cross_value / (total + 1e-8)
        mid = 0.5 * (ask_price1 + bid_price1)
        spread = max(ask_price1 - bid_price1, 1e-6)
        return float((micro - mid) / spread)

    def label_time_segment(value) -> int:
        """按官方示例的左开右闭规则返回固定的 30 分钟截面。"""
        hhmmss = value.hour * 10000 + value.minute * 100 + value.second
        sections = (
            (90000, 93000), (93000, 100000), (100000, 103000),
            (103000, 110000), (110000, 113000), (130000, 133000),
            (133000, 140000), (140000, 143000),
        )
        for section_start, section_end in sections:
            if section_start < hhmmss <= section_end:
                return section_end
        return -1

    sql = f"""
    WITH cte_snapshot AS (
        SELECT
            date,
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            minute_multilevel_microprice(
                COALESCE(ask_price1, 0),
                COALESCE(ask_price2, 0),
                COALESCE(ask_price3, 0),
                COALESCE(ask_price4, 0),
                COALESCE(ask_price5, 0),
                COALESCE(bid_price1, 0),
                COALESCE(bid_price2, 0),
                COALESCE(bid_price3, 0),
                COALESCE(bid_price4, 0),
                COALESCE(bid_price5, 0),
                COALESCE(ask_volume1, 0),
                COALESCE(ask_volume2, 0),
                COALESCE(ask_volume3, 0),
                COALESCE(ask_volume4, 0),
                COALESCE(ask_volume5, 0),
                COALESCE(bid_volume1, 0),
                COALESCE(bid_volume2, 0),
                COALESCE(bid_volume3, 0),
                COALESCE(bid_volume4, 0),
                COALESCE(bid_volume5, 0)
            ) AS minute_signal,
            label_time_segment(date) AS time_segment
        FROM {bar1m}
        WHERE time_segment != -1
    ),
    cte_window AS (
        SELECT trading_day, time_segment, instrument_id,
               AVG(minute_signal) AS factor
        FROM cte_snapshot
        GROUP BY instrument_id, trading_day, time_segment
    )
    SELECT CAST(CONCAT(
        f.trading_day, ' ',
        strftime(strptime(LPAD(f.time_segment, 6, '0'), '%H%M%S'), '%H:%M:%S')
    ) AS DATETIME) AS date,
    all_instruments.instrument,
    f.factor AS factor
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(
        sql,
        filters={"date": [start_date, end_date]},
        compression=True,
        udf_list=[
                        dai.DaiUDF(name="minute_multilevel_microprice", function=minute_multilevel_microprice),

            dai.DaiUDF(name="label_time_segment", function=label_time_segment),
        ],
    ).df()

    df = df.replace([np.inf, -np.inf], np.nan)
    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    result = pd.merge(df, stk_pool, how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"])
    return result[["date", "instrument", "factor"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


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
