def main(datasources, start_date, end_date):
    """
    用买一卖一挂单量占五档总量的比例，度量流动性是否集中在最优档。
    集中度高表示盘口更薄、贴近价位的挂单更密。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回包含 ['date', 'instrument', 'factor'] 三列的 DataFrame。
    """
    import math
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    def minute_concentration(
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
        """买一卖一挂单量占五档总量比例。"""
        total = (bid_volume1 + bid_volume2 + bid_volume3 + bid_volume4 + bid_volume5
                 + ask_volume1 + ask_volume2 + ask_volume3 + ask_volume4 + ask_volume5)
        return float((bid_volume1 + ask_volume1) / (total + 1e-8))

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
            minute_concentration(
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
            ) AS sig,
            label_time_segment(date) AS time_segment
        FROM {bar1m}
    ),
    cte_window AS (
        SELECT
            trading_day,
            time_segment,
            instrument_id,
            AVG(sig) AS factor
        FROM cte_snapshot
        WHERE time_segment != -1
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
            dai.DaiUDF(name="minute_concentration", function=minute_concentration),
            dai.DaiUDF(name="label_time_segment", function=label_time_segment),
        ],
    ).df()

    # inf 必须清除；NaN 保留，交给平台按缺失率规则统计。
    df = df.replace([np.inf, -np.inf], np.nan)

    # 与官方三份示例一致：股票池单独查询后在 pandas 内连接。
    stk_pool = dai.query(
        "SELECT date, instrument FROM cpt_jyc_2026_instruments",
        filters={"date": [start_date, end_date]},
    ).df()
    result = pd.merge(df, stk_pool, how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"])

    return result[["date", "instrument", "factor"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


if __name__ == '__main__':
    from bigmodule import M
    import structlog

    logger = structlog.get_logger()
    datasources = {'bar1m': 'cpt_jyc_2026_stock_bar1m'}
    start_date = '2020-01-01 00:00:00'
    end_date = '2020-03-01 23:59:59'

    logger.info(f"计算因子，区间：{start_date} ~ {end_date}")
    factor_data = main(datasources, start_date, end_date)
    result = M.jyc_eval._latest(factor_data=factor_data, show=True)
