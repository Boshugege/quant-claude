def main(datasources, start_date, end_date):
    """
    委托笔数不平衡。

    经济含义：比较买卖五档的委托笔数，而不是委托量，刻画两侧排队订单的广度。
    可能有效的理由：相同总挂单量下，很多小单与少数大单代表不同的队列结构；委托笔数是此前完全没有使用的独立信息。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    def minute_order_count_imbalance(
        ask_num_orders1: float,
        ask_num_orders2: float,
        ask_num_orders3: float,
        ask_num_orders4: float,
        ask_num_orders5: float,
        bid_num_orders1: float,
        bid_num_orders2: float,
        bid_num_orders3: float,
        bid_num_orders4: float,
        bid_num_orders5: float,
    ) -> float:
        """买卖五档委托笔数的不平衡。"""
        ask = ask_num_orders1 + ask_num_orders2 + ask_num_orders3 + ask_num_orders4 + ask_num_orders5
        bid = bid_num_orders1 + bid_num_orders2 + bid_num_orders3 + bid_num_orders4 + bid_num_orders5
        return float((bid - ask) / (bid + ask + 1e-8))

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
            minute_order_count_imbalance(
                COALESCE(ask_num_orders1, 0),
                COALESCE(ask_num_orders2, 0),
                COALESCE(ask_num_orders3, 0),
                COALESCE(ask_num_orders4, 0),
                COALESCE(ask_num_orders5, 0),
                COALESCE(bid_num_orders1, 0),
                COALESCE(bid_num_orders2, 0),
                COALESCE(bid_num_orders3, 0),
                COALESCE(bid_num_orders4, 0),
                COALESCE(bid_num_orders5, 0)
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
                        dai.DaiUDF(name="minute_order_count_imbalance", function=minute_order_count_imbalance),

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
