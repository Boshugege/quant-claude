def main(datasources, start_date, end_date):
    """
    外档半窗口趋势。

    经济含义：将每个 30 分钟窗口前半段赋负权、后半段赋正权，测量外档压力是否在窗口内增强。
    可能有效的理由：末段单独均值较弱，但“从前半段到后半段的变化”可能仍包含信息；这是路径变化因子，不是静态盘口权重。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    def minute_tail_imbalance(
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
        """外侧两档的标准不平衡。"""
        bid_outer = bid_volume4 + bid_volume5
        ask_outer = ask_volume4 + ask_volume5
        return float((bid_outer - ask_outer) / (bid_outer + ask_outer + 1e-8))

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

    def minute_half_phase(value) -> float:
        """前半段为 -1、后半段为 +1，测量窗口内信号是否向收盘方向增强。"""
        hhmmss = value.hour * 10000 + value.minute * 100 + value.second
        sections = (
            (90000, 93000, 91500),
            (93000, 100000, 94500),
            (100000, 103000, 101500),
            (103000, 110000, 104500),
            (110000, 113000, 111500),
            (130000, 133000, 131500),
            (133000, 140000, 134500),
            (140000, 143000, 141500),
        )
        for start, end, midpoint in sections:
            if start < hhmmss <= end:
                return 1.0 if hhmmss > midpoint else -1.0
        return 0.0

    sql = f"""
    WITH cte_snapshot AS (
        SELECT
            date,
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            minute_tail_imbalance(
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
            minute_half_phase(date) AS phase,
            label_time_segment(date) AS time_segment
        FROM {bar1m}
        WHERE time_segment != -1
    ),
    cte_terms AS (
        SELECT
            *,
            minute_signal * phase AS weighted_signal
        FROM cte_snapshot
    ),
    cte_window AS (
        SELECT
            trading_day,
            time_segment,
            instrument_id,
            AVG(weighted_signal) AS factor
        FROM cte_terms
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
            dai.DaiUDF(name="minute_tail_imbalance", function=minute_tail_imbalance),
            dai.DaiUDF(name="label_time_segment", function=label_time_segment),
            dai.DaiUDF(name="minute_half_phase", function=minute_half_phase)
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
