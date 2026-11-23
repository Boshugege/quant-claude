def main(datasources, start_date, end_date):
    """
    最外第五档盘口不平衡。

    经济含义：只使用买五和卖五的挂单差异，测量最远可见档位的方向压力。
    可能有效的理由：已有结果随权重向外移动而增强；若信息主要集中在最远档，单独第五档会比混合权重更纯，能验证外档效应是否由极端档位驱动。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    def minute_level5_imbalance(
        bid_volume1: float,
        bid_volume2: float,
        bid_volume3: float,
        bid_volume4: float,
        bid_volume5: float,
        ask_volume1: float,
        ask_volume2: float,
        ask_volume3: float,
        ask_volume4: float,
        ask_volume5: float,
    ) -> float:
        """仅用买五和卖五的挂单量计算不平衡度。"""
        return float((bid_volume5 - ask_volume5) / (bid_volume5 + ask_volume5 + 1e-8))

    def label_time_segment(value) -> int:
        """按官方示例的左开右闭规则返回固定的 30 分钟截面。"""
        hhmmss = value.hour * 10000 + value.minute * 100 + value.second
        sections = (
            (90000, 93000),
            (93000, 100000),
            (100000, 103000),
            (103000, 110000),
            (110000, 113000),
            (130000, 133000),
            (133000, 140000),
            (140000, 143000),
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
            minute_level5_imbalance(
                COALESCE(bid_volume1, 0),
                COALESCE(bid_volume2, 0),
                COALESCE(bid_volume3, 0),
                COALESCE(bid_volume4, 0),
                COALESCE(bid_volume5, 0),
                COALESCE(ask_volume1, 0),
                COALESCE(ask_volume2, 0),
                COALESCE(ask_volume3, 0),
                COALESCE(ask_volume4, 0),
                COALESCE(ask_volume5, 0)
            ) AS minute_signal,
            label_time_segment(date) AS time_segment
        FROM {bar1m}
        WHERE time_segment != -1
    ),
    cte_window AS (
        SELECT
            trading_day,
            time_segment,
            instrument_id,
            AVG(minute_signal) AS factor
        FROM cte_snapshot
        GROUP BY instrument_id, trading_day, time_segment
    )
    SELECT
        CAST(CONCAT(
            f.trading_day,
            ' ',
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
            dai.DaiUDF(name="minute_level5_imbalance", function=minute_level5_imbalance),
            dai.DaiUDF(name="label_time_segment", function=label_time_segment),
        ],
    ).df()

    # 清除无穷值；NaN 交给平台按截面缺失率规则统计。
    df = df.replace([np.inf, -np.inf], np.nan)

    # 股票池单独查询，并按当天截面内连接；支持平台注入逻辑表名。
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
