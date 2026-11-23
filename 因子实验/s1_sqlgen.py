# -*- coding: utf-8 -*-
"""评分机直接跑脚本（10-20 夜，用户："评分机是可以直接上交脚本然后跑和评分的"）：
生成一批平台格式的因子 .py（main(datasources, start_date, end_date)），共用一段 SQL（每股每段一行的分钟级统计），
后处理公式各不相同，每个文件 = 一个新信号假设。交给 personal 评分机（kind=platform）在模拟平台上原样运行并评分，
16 路并行；好的直接就是可交付代码形式。

用法：python3 s1_sqlgen.py <输出目录> [批次]   → 写出 <批次>_<名字>.py
分钟级统计（只用过去数据）：分钟内收益 r = ln(收/开) 的 1～4 阶和、正负半方差、极值、零收益分钟数；
签名成交量 sv = 量·(收−开)/(高−低)（分钟内的买卖方向近似）与 r·sv、sv²；成交位置 (收−买一)/(卖一−买一)；
盘口空档 (卖二−卖一)−(买一−买二)（相对中间价）；段内开高低收与高点、低点出现的时刻。
"""
import os
import sys

SQL = r'''
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            CAST(open AS DOUBLE) AS o, CAST(high AS DOUBLE) AS h, CAST(low AS DOUBLE) AS l, CAST(close AS DOUBLE) AS c,
            CAST(COALESCE(volume, 0) AS DOUBLE) AS v, CAST(COALESCE(amount, 0) AS DOUBLE) AS a,
            CAST(COALESCE(deal_number, 0) AS DOUBLE) AS nd, CAST(pre_close AS DOUBLE) AS pc,
            CAST(bid_price1 AS DOUBLE) AS b1, CAST(ask_price1 AS DOUBLE) AS a1,
            CAST(bid_price2 AS DOUBLE) AS b2, CAST(ask_price2 AS DOUBLE) AS a2,
            CAST(bid_price3 AS DOUBLE) AS b3, CAST(ask_price3 AS DOUBLE) AS a3
        FROM {bar1m}
    ),
    cte_m AS (
        SELECT
            instrument_id, trading_day, hms, o, h, l, c, v, a, nd, pc,
            CASE
                WHEN hms > '090000' AND hms <= '093000' THEN 93000
                WHEN hms > '093000' AND hms <= '100000' THEN 100000
                WHEN hms > '100000' AND hms <= '103000' THEN 103000
                WHEN hms > '103000' AND hms <= '110000' THEN 110000
                WHEN hms > '110000' AND hms <= '113000' THEN 113000
                WHEN hms > '130000' AND hms <= '133000' THEN 133000
                WHEN hms > '133000' AND hms <= '140000' THEN 140000
                WHEN hms > '140000' AND hms <= '143000' THEN 143000
                WHEN hms > '143000' AND hms <= '150000' THEN 150000
                ELSE -1
            END AS seg,
            CASE WHEN o > 0 AND c > 0 AND v > 0 THEN ln(c / o) ELSE NULL END AS r,
            CASE WHEN h > l AND v > 0 AND o > 0 AND c > 0 THEN v * (c - o) / (h - l) ELSE 0 END AS sv,
            CASE WHEN b1 > 0 AND a1 > b1 THEN (a1 + b1) / 2.0 ELSE NULL END AS mid,
            CASE WHEN b1 > 0 AND a1 > b1 AND c > 0 THEN LEAST(GREATEST((c - b1) / (a1 - b1), 0.0), 1.0) ELSE NULL END AS tpos,
            CASE WHEN b1 > 0 AND a1 > b1 AND a2 > a1 AND b2 > 0 AND b2 < b1 THEN ((a2 - a1) - (b1 - b2)) / ((a1 + b1) / 2.0) ELSE NULL END AS gap12,
            CASE WHEN b1 > 0 AND a1 > b1 AND a3 > a1 AND b3 > 0 AND b3 < b1 THEN ((a3 - a1) - (b1 - b3)) / ((a1 + b1) / 2.0) ELSE NULL END AS gap13
        FROM cte_raw
    ),
    cte_seg AS (
        SELECT
            trading_day, seg, instrument_id,
            SUM(r) AS sr, SUM(r * r) AS sr2, SUM(r * r * r) AS sr3, SUM(r * r * r * r) AS sr4, SUM(ABS(r)) AS sabs,
            SUM(CASE WHEN r > 0 THEN r * r ELSE 0 END) AS sr2p, SUM(CASE WHEN r < 0 THEN r * r ELSE 0 END) AS sr2n,
            MAX(r) AS rmax, MIN(r) AS rmin, COUNT(r) AS nmin, SUM(CASE WHEN r = 0 THEN 1 ELSE 0 END) AS nzero,
            SUM(sv) AS ssv, SUM(COALESCE(r, 0) * sv) AS srsv, SUM(sv * sv) AS ssv2,
            SUM(CASE WHEN sv > 0 THEN COALESCE(r, 0) * sv ELSE 0 END) AS srsvp, SUM(CASE WHEN sv > 0 THEN sv * sv ELSE 0 END) AS ssv2p,
            SUM(CASE WHEN sv < 0 THEN COALESCE(r, 0) * sv ELSE 0 END) AS srsvn, SUM(CASE WHEN sv < 0 THEN sv * sv ELSE 0 END) AS ssv2n,
            SUM(CASE WHEN r > 0 THEN v ELSE 0 END) AS vup, SUM(CASE WHEN r < 0 THEN v ELSE 0 END) AS vdn,
            SUM(v) AS vol, SUM(a) AS amt, SUM(nd) AS ndeal,
            AVG(tpos) AS tpos, AVG(gap12) AS gap12, AVG(gap13) AS gap13,
            arg_min(CASE WHEN o > 0 THEN o ELSE NULL END, hms) AS s_open, MAX(CASE WHEN h > 0 THEN h ELSE NULL END) AS s_high,
            MIN(CASE WHEN l > 0 THEN l ELSE NULL END) AS s_low,
            arg_max(CASE WHEN mid > 0 THEN mid WHEN c > 0 THEN c ELSE NULL END, hms) AS last_px,
            arg_max(hms, CASE WHEN h > 0 THEN h ELSE NULL END) AS hms_high, arg_min(hms, CASE WHEN l > 0 THEN l ELSE NULL END) AS hms_low,
            MAX(pc) AS pre_close
        FROM cte_m
        WHERE seg != -1
        GROUP BY instrument_id, trading_day, seg
    )
    SELECT
        CAST(CONCAT(f.trading_day, ' ', strftime(strptime(LPAD(CAST(f.seg AS VARCHAR), 6, '0'), '%H%M%S'), '%H:%M:%S')) AS DATETIME) AS date,
        all_instruments.instrument, f.*
    FROM cte_seg f
    LEFT JOIN all_instruments USING (instrument_id)
'''

SQL_MKT = SQL.replace("""    cte_seg AS (""", """    cte_mk AS (
        SELECT trading_day, hms, AVG(r) AS rm FROM cte_m WHERE r IS NOT NULL AND seg != -1 AND seg != 93000 GROUP BY trading_day, hms
    ),
    cte_mk2 AS (
        SELECT trading_day, hms, rm, LAG(rm) OVER (PARTITION BY trading_day ORDER BY hms) AS rm_lag FROM cte_mk
    ),
    cte_mm AS (
        SELECT m.*, k.rm, COALESCE(k.rm_lag, 0) AS rm_lag FROM cte_m m LEFT JOIN cte_mk2 k ON m.trading_day = k.trading_day AND m.hms = k.hms
    ),
    cte_seg AS (""").replace("""        FROM cte_m
        WHERE seg != -1
        GROUP BY""", """            , SUM(CASE WHEN r IS NOT NULL THEN rm ELSE 0 END) AS srm, SUM(CASE WHEN r IS NOT NULL THEN rm * rm ELSE 0 END) AS srm2,
            SUM(COALESCE(r * rm, 0)) AS srrm, SUM(COALESCE(r * rm * rm, 0)) AS srrm2, SUM(COALESCE(r * rm_lag, 0)) AS srrml,
            SUM(CASE WHEN r IS NOT NULL THEN rm_lag * rm_lag ELSE 0 END) AS srml2
        FROM cte_mm
        WHERE seg != -1
        GROUP BY""")
assert "cte_mm" in SQL_MKT and "srrm2" in SQL_MKT
# 盘口委托结构一族：分钟级的笔数失衡、量失衡、平均委托规模比，段内均值、最后 10 分钟均值、段末快照
_IMB = lambda b, a: "CASE WHEN ({b}) + ({a}) > 0 THEN (({b}) - ({a})) / (({b}) + ({a})) ELSE NULL END".format(b=b, a=a)   # noqa: E731
SQL_Q = SQL.replace("""            CAST(bid_price3 AS DOUBLE) AS b3, CAST(ask_price3 AS DOUBLE) AS a3
        FROM {bar1m}""", """            CAST(bid_price3 AS DOUBLE) AS b3, CAST(ask_price3 AS DOUBLE) AS a3,
            CAST(COALESCE(bid_volume1, 0) AS DOUBLE) AS bv1, CAST(COALESCE(ask_volume1, 0) AS DOUBLE) AS av1,
            CAST(COALESCE(bid_volume2, 0) + COALESCE(bid_volume3, 0) AS DOUBLE) AS bv23, CAST(COALESCE(ask_volume2, 0) + COALESCE(ask_volume3, 0) AS DOUBLE) AS av23,
            CAST(COALESCE(bid_num_orders1, 0) AS DOUBLE) AS bn1, CAST(COALESCE(ask_num_orders1, 0) AS DOUBLE) AS an1,
            CAST(COALESCE(bid_num_orders2, 0) AS DOUBLE) AS bn2, CAST(COALESCE(ask_num_orders2, 0) AS DOUBLE) AS an2,
            CAST(COALESCE(bid_num_orders3, 0) AS DOUBLE) AS bn3, CAST(COALESCE(ask_num_orders3, 0) AS DOUBLE) AS an3
        FROM {bar1m}""").replace("""            instrument_id, trading_day, hms, o, h, l, c, v, a, nd, pc,
""", """            instrument_id, trading_day, hms, o, h, l, c, v, a, nd, pc,
            """ + _IMB("bn1", "an1") + """ AS qn1,
            """ + _IMB("bn2", "an2") + """ AS qn2,
            """ + _IMB("bn3", "an3") + """ AS qn3,
            """ + _IMB("bn2 + bn3", "an2 + an3") + """ AS qn23,
            """ + _IMB("bv1", "av1") + """ AS qv1,
            """ + _IMB("bv23", "av23") + """ AS qv23,
            CASE WHEN bn1 > 0 AND an1 > 0 AND bv1 > 0 AND av1 > 0 THEN ln((bv1 / bn1) / (av1 / an1)) ELSE NULL END AS qs1,
            CASE WHEN bn2 + bn3 > 0 AND an2 + an3 > 0 AND bv23 > 0 AND av23 > 0 THEN ln((bv23 / (bn2 + bn3)) / (av23 / (an2 + an3))) ELSE NULL END AS qs23,
            CASE WHEN bv1 > 0 AND av1 > 0 AND bv23 > 0 AND av23 > 0 THEN ln(bv23 / bv1) - ln(av23 / av1) ELSE NULL END AS qth,
            CASE WHEN hms > '090000' AND hms <= '093000' THEN '093000' WHEN hms > '093000' AND hms <= '100000' THEN '100000'
                 WHEN hms > '100000' AND hms <= '103000' THEN '103000' WHEN hms > '103000' AND hms <= '110000' THEN '110000'
                 WHEN hms > '110000' AND hms <= '113000' THEN '113000' WHEN hms > '130000' AND hms <= '133000' THEN '133000'
                 WHEN hms > '133000' AND hms <= '140000' THEN '140000' WHEN hms > '140000' AND hms <= '143000' THEN '143000'
                 ELSE '150000' END AS seg_end_s,
""").replace("""        FROM cte_m
        WHERE seg != -1
        GROUP BY""", """            , AVG(qn1) AS qn1, AVG(qn2) AS qn2, AVG(qn3) AS qn3, AVG(qn23) AS qn23, AVG(qv1) AS qv1, AVG(qv23) AS qv23,
            AVG(qs1) AS qs1, AVG(qs23) AS qs23, AVG(qth) AS qth,
            AVG(CASE WHEN hms > strftime(strptime(seg_end_s, '%H%M%S') - INTERVAL 10 MINUTE, '%H%M%S') THEN qn23 END) AS qn23L,
            AVG(CASE WHEN hms > strftime(strptime(seg_end_s, '%H%M%S') - INTERVAL 10 MINUTE, '%H%M%S') THEN qs1 END) AS qs1L,
            arg_max(qn23, hms) AS qn23E, arg_max(qs1, hms) AS qs1E
        FROM cte_m
        WHERE seg != -1
        GROUP BY""")
assert "qn23L" in SQL_Q and "bn23" not in SQL_Q.split("cte_m AS")[0][-10:]
# 分钟收益自相关一族（同一股票同一天内上一分钟的收益）
SQL_AC = SQL.replace("""    cte_seg AS (""", """    cte_m2 AS (
        SELECT *, LAG(r) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms) AS r_lag FROM cte_m
    ),
    cte_seg AS (""").replace("""        FROM cte_m
        WHERE seg != -1
        GROUP BY""", """            , SUM(r * r_lag) AS srl, SUM(CASE WHEN r IS NOT NULL THEN r_lag * r_lag END) AS srl2, COUNT(r * r_lag) AS nrl
        FROM cte_m2
        WHERE seg != -1
        GROUP BY""")
# 盘口深度与价差一族：三档总深度、深度失衡（量）、相对价差的段内均值与段首／段末
SQL_BOOK = SQL.replace("""            CAST(bid_price3 AS DOUBLE) AS b3, CAST(ask_price3 AS DOUBLE) AS a3
        FROM {bar1m}""", """            CAST(bid_price3 AS DOUBLE) AS b3, CAST(ask_price3 AS DOUBLE) AS a3,
            CAST(COALESCE(bid_volume1, 0) + COALESCE(bid_volume2, 0) + COALESCE(bid_volume3, 0) AS DOUBLE) AS bvs,
            CAST(COALESCE(ask_volume1, 0) + COALESCE(ask_volume2, 0) + COALESCE(ask_volume3, 0) AS DOUBLE) AS avs
        FROM {bar1m}""").replace("""            instrument_id, trading_day, hms, o, h, l, c, v, a, nd, pc,
""", """            instrument_id, trading_day, hms, o, h, l, c, v, a, nd, pc,
            CASE WHEN b1 > 0 AND a1 > b1 THEN (a1 - b1) / ((a1 + b1) / 2.0) ELSE NULL END AS spr,
            CASE WHEN bvs + avs > 0 THEN bvs + avs ELSE NULL END AS dep,
            CASE WHEN bvs + avs > 0 THEN (bvs - avs) / (bvs + avs) ELSE NULL END AS dimb,
""").replace("""        FROM cte_m
        WHERE seg != -1
        GROUP BY""", """            , AVG(spr) AS spr_avg, arg_max(spr, hms) AS spr_end, AVG(dep) AS dep_avg, arg_min(dep, hms) AS dep_first, arg_max(dep, hms) AS dep_end,
            AVG(dimb) AS dimb_avg, arg_max(dimb, hms) AS dimb_end
        FROM cte_m
        WHERE seg != -1
        GROUP BY""")
assert "r_lag" in SQL_AC and "dimb_end" in SQL_BOOK
# 量价一族：分钟 |收益|·量、量²（量价相关），跳跃分钟数（|r| 超过当天分钟波动 4 倍；当天波动用截至该分钟的全天平方和近似不可得，改用段内）
SQL_PV = SQL.replace("""        FROM cte_m
        WHERE seg != -1
        GROUP BY""", """            , SUM(ABS(r) * v) AS srav, SUM(CASE WHEN r IS NOT NULL THEN v * v END) AS svv, SUM(CASE WHEN r IS NOT NULL THEN v END) AS svr
        FROM cte_m
        WHERE seg != -1
        GROUP BY""")
# 成交高潮与涨跌分钟数一族：段内最大分钟量及其收盘价、涨／跌分钟数、15:00 那一分钟（收盘集合竞价）的成交量
SQL_VC = SQL.replace("""        FROM cte_m
        WHERE seg != -1
        GROUP BY""", """            , MAX(v) AS vmax, arg_max(c, v) AS c_vmax, SUM(CASE WHEN r > 0 THEN 1 ELSE 0 END) AS npos, SUM(CASE WHEN r < 0 THEN 1 ELSE 0 END) AS nneg,
            SUM(CASE WHEN hms = '150000' THEN v ELSE 0 END) AS v_close
        FROM cte_m
        WHERE seg != -1
        GROUP BY""")
# 分钟时刻价格一族：09:35、09:45、10:15、10:45 那一分钟的价格（中间价，无则收盘），以及开盘 30 分钟的成交额与量
SQL_TM = SQL.replace("""        FROM cte_m
        WHERE seg != -1
        GROUP BY""", """            , MAX(CASE WHEN hms = '093500' THEN COALESCE(mid, c) END) AS p0935, MAX(CASE WHEN hms = '094500' THEN COALESCE(mid, c) END) AS p0945,
            MAX(CASE WHEN hms = '101500' THEN COALESCE(mid, c) END) AS p1015, MAX(CASE WHEN hms = '104500' THEN COALESCE(mid, c) END) AS p1045
        FROM cte_m
        WHERE seg != -1
        GROUP BY""")
# 段内最后几分钟一族（在委托结构 SQL 上加）：最后 5／10 分钟的起点价格、成交量、三档量失衡、相对价差、最高／最低
_L = lambda m: "hms > strftime(strptime(seg_end_s, '%H%M%S') - INTERVAL {} MINUTE, '%H%M%S')".format(m)   # noqa: E731
SQL_LATE = SQL_Q.replace("""            arg_max(qn23, hms) AS qn23E, arg_max(qs1, hms) AS qs1E
        FROM cte_m""", """            arg_max(qn23, hms) AS qn23E, arg_max(qs1, hms) AS qs1E,
            arg_min(CASE WHEN """ + _L(5) + """ THEN COALESCE(mid, c) END, CASE WHEN """ + _L(5) + """ THEN hms END) AS p_l5,
            arg_min(CASE WHEN """ + _L(10) + """ THEN COALESCE(mid, c) END, CASE WHEN """ + _L(10) + """ THEN hms END) AS p_l10,
            SUM(CASE WHEN """ + _L(5) + """ THEN v ELSE 0 END) AS v_l5,
            AVG(CASE WHEN """ + _L(5) + """ AND bv1 + bv23 + av1 + av23 > 0 THEN (bv1 + bv23 - av1 - av23) / (bv1 + bv23 + av1 + av23) END) AS dimb_l5,
            AVG(CASE WHEN bv1 + bv23 + av1 + av23 > 0 THEN (bv1 + bv23 - av1 - av23) / (bv1 + bv23 + av1 + av23) END) AS dimb_all,
            AVG(CASE WHEN """ + _L(5) + """ AND b1 > 0 AND a1 > b1 THEN (a1 - b1) / ((a1 + b1) / 2.0) END) AS spr_l5,
            AVG(CASE WHEN b1 > 0 AND a1 > b1 THEN (a1 - b1) / ((a1 + b1) / 2.0) END) AS spr_all,
            MAX(CASE WHEN """ + _L(10) + """ AND c > 0 THEN c END) AS hi_l10, MIN(CASE WHEN """ + _L(10) + """ AND c > 0 THEN c END) AS lo_l10
        FROM cte_m""")
SQL_LATE = SQL_LATE.replace("""            instrument_id, trading_day, hms, o, h, l, c, v, a, nd, pc,
""", """            instrument_id, trading_day, hms, o, h, l, c, v, a, nd, pc, b1, a1, bv1, bv23, av1, av23,
""", 1)
assert "p_l5" in SQL_LATE and "pc, b1, a1, bv1" in SQL_LATE
# 委托结构的尾段窗口一族（10-21 晚：最后 10 分钟委托结构加到 1617 上五项全好）：5／15 分钟窗口、一档笔数失衡
_LQ = lambda m, col: "AVG(CASE WHEN " + _L(m) + " THEN " + col + " END)"   # noqa: E731
SQL_Q2 = SQL_Q.replace("""            arg_max(qn23, hms) AS qn23E, arg_max(qs1, hms) AS qs1E
        FROM cte_m""", """            arg_max(qn23, hms) AS qn23E, arg_max(qs1, hms) AS qs1E,
            """ + _LQ(5, "qn23") + """ AS qn23L5, """ + _LQ(5, "qs1") + """ AS qs1L5,
            """ + _LQ(15, "qn23") + """ AS qn23L15, """ + _LQ(15, "qs1") + """ AS qs1L15,
            """ + _LQ(10, "qn1") + """ AS qn1L, """ + _LQ(10, "qv23") + """ AS qv23L
        FROM cte_m""")
assert "qn23L15" in SQL_Q2
# 挂单规模一族（10-21 晚：一档平均委托规模比是相对 1617 唯一独特的信息）：买／卖一单边规模、段首与段尾、深档规模比
_F = lambda m: "hms <= strftime(strptime(seg_end_s, '%H%M%S') - INTERVAL " + str(30 - m) + " MINUTE, '%H%M%S')"   # noqa: E731  段首 m 分钟
SQL_SZ = SQL_Q2.replace("""            CASE WHEN bv1 > 0 AND av1 > 0 AND bv23 > 0 AND av23 > 0 THEN ln(bv23 / bv1) - ln(av23 / av1) ELSE NULL END AS qth,""",
                        """            CASE WHEN bv1 > 0 AND av1 > 0 AND bv23 > 0 AND av23 > 0 THEN ln(bv23 / bv1) - ln(av23 / av1) ELSE NULL END AS qth,
            CASE WHEN bn1 > 0 AND bv1 > 0 THEN ln(bv1 / bn1) ELSE NULL END AS lbs1,
            CASE WHEN an1 > 0 AND av1 > 0 THEN ln(av1 / an1) ELSE NULL END AS las1,""").replace("""            """ + _LQ(10, "qn1") + """ AS qn1L, """ + _LQ(10, "qv23") + """ AS qv23L
        FROM cte_m""", """            """ + _LQ(10, "qn1") + """ AS qn1L, """ + _LQ(10, "qv23") + """ AS qv23L,
            """ + _LQ(10, "lbs1") + """ AS lbs1L, """ + _LQ(10, "las1") + """ AS las1L, """ + _LQ(10, "qs23") + """ AS qs23L,
            AVG(CASE WHEN """ + _F(10) + """ THEN qs1 END) AS qs1F, AVG(lbs1) AS lbs1A, AVG(las1) AS las1A
        FROM cte_m""")
assert "lbs1L" in SQL_SZ and "qs1F" in SQL_SZ and "AS lbs1," in SQL_SZ
# 盘口动态一族（截面前 10 分钟）：委托到达净差、微观价格偏离、队伍厚度不对称
_L10 = _L(10)
SQL_BD = SQL_SZ.replace("""            instrument_id, trading_day, hms, o, h, l, c, v, a, nd, pc,
""", """            instrument_id, trading_day, hms, o, h, l, c, v, a, nd, pc,
            bn1 + bn2 + bn3 AS bna, an1 + an2 + an3 AS ana,
            CASE WHEN b1 > 0 AND a1 > b1 AND bv1 + av1 > 0 THEN ((b1 * av1 + a1 * bv1) / (bv1 + av1) - (a1 + b1) / 2.0) / ((a1 + b1) / 2.0) ELSE NULL END AS mdev,
""", 1).replace("""            AVG(CASE WHEN """ + _F(10) + """ THEN qs1 END) AS qs1F, AVG(lbs1) AS lbs1A, AVG(las1) AS las1A
        FROM cte_m""", """            AVG(CASE WHEN """ + _F(10) + """ THEN qs1 END) AS qs1F, AVG(lbs1) AS lbs1A, AVG(las1) AS las1A,
            arg_max(bna, hms) AS bna_e, arg_max(ana, hms) AS ana_e,
            arg_min(CASE WHEN """ + _L10 + """ THEN bna END, CASE WHEN """ + _L10 + """ THEN hms END) AS bna_s,
            arg_min(CASE WHEN """ + _L10 + """ THEN ana END, CASE WHEN """ + _L10 + """ THEN hms END) AS ana_s,
            """ + _LQ(10, "mdev") + """ AS mdevL, """ + _LQ(10, "qth") + """ AS qthL
        FROM cte_m""")
assert "mdevL" in SQL_BD and "bna_s" in SQL_BD and "AS mdev," in SQL_BD
# 截面那一刻一族（10-21 夜）：段末价差（跳价数）、最后一分钟收益与成交量、段末成交位置、最后 10 分钟零成交分钟数
SQL_X = SQL_BD.replace("""            bn1 + bn2 + bn3 AS bna, an1 + an2 + an3 AS ana,
""", """            bn1 + bn2 + bn3 AS bna, an1 + an2 + an3 AS ana,
            CASE WHEN b1 > 0 AND a1 > b1 THEN a1 - b1 ELSE NULL END AS sprt,
""", 1).replace("""            """ + _LQ(10, "mdev") + """ AS mdevL, """ + _LQ(10, "qth") + """ AS qthL
        FROM cte_m""", """            """ + _LQ(10, "mdev") + """ AS mdevL, """ + _LQ(10, "qth") + """ AS qthL,
            arg_max(sprt, hms) AS spr_e, arg_max(r, hms) AS r_last, arg_max(v, hms) AS v_last, arg_max(tpos, hms) AS tpos_e,
            SUM(CASE WHEN """ + _L10 + """ AND v = 0 THEN 1 ELSE 0 END) AS z10
        FROM cte_m""")
assert "spr_e" in SQL_X and "AS sprt," in SQL_X
# 截面前 10 分钟成交规模一族：最后 10 分钟成交额、笔数、收益
SQL_TS = SQL_X.replace("""            SUM(CASE WHEN """ + _L10 + """ AND v = 0 THEN 1 ELSE 0 END) AS z10
        FROM cte_m""", """            SUM(CASE WHEN """ + _L10 + """ AND v = 0 THEN 1 ELSE 0 END) AS z10,
            SUM(CASE WHEN """ + _L10 + """ THEN a ELSE 0 END) AS a_l10, SUM(CASE WHEN """ + _L10 + """ THEN nd ELSE 0 END) AS nd_l10,
            SUM(CASE WHEN """ + _L10 + """ THEN COALESCE(r, 0) ELSE 0 END) AS r_l10, SUM(CASE WHEN """ + _L10 + """ THEN COALESCE(sv, 0) ELSE 0 END) AS sv_l10
        FROM cte_m""")
assert "a_l10" in SQL_TS
SQLS = {"base": SQL, "mkt": SQL_MKT, "queue": SQL_Q, "ac": SQL_AC, "book": SQL_BOOK, "pv": SQL_PV, "vc": SQL_VC, "tm": SQL_TM, "late": SQL_LATE, "q2": SQL_Q2, "sz": SQL_SZ, "bd": SQL_BD, "x": SQL_X, "ts": SQL_TS}

TEMPLATE = r'''def main(datasources, start_date, end_date):
    """
    {doc}

    评分机批量筛选候选（{batch}）。共用分钟级统计 SQL（单次 GROUP BY，不用窗口函数），跨段运算在 pandas 中完成。
    最终：逐截面排名居中{smooth_doc}，09:30 截面置 0，缺失补 0。只用过去数据。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]
    LOOKBACK_DAYS = {lookback}
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""{sql}"""
    df = dai.query(sql, filters={{"date": [query_start, end_date]}}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c not in ("date", "instrument", "trading_day", "seg", "instrument_id", "hms_high", "hms_low"):
            df[c] = df[c].astype("float64")
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["day"] = df["date"].dt.normalize()
    inst = df["instrument"]
    g = df.groupby("instrument", sort=False)
    gd = df.groupby(["instrument", "day"], sort=False)

    def pos(x):
        return x.where(x > 0)

    def lr(a, b):
        return np.log(pos(a) / pos(b))

    def roll(col, k):
        x = df[col].fillna(0.0) if isinstance(col, str) else col.fillna(0.0)
        cs = x.groupby(inst).cumsum()
        return cs - cs.groupby(inst).shift(k).fillna(0.0)

    def tmin(hms):
        """时刻 → 开盘以来的交易分钟（午休不计）。"""
        t = pd.to_numeric(hms, errors="coerce")
        hh, mm = t // 10000, (t // 100) % 100
        m = hh * 60 + mm
        return np.where(m <= 690, m - 570, m - 660)

    px = pos(df["last_px"])
    prev = g["last_px"].shift(1)
    r_seg = lr(px, prev)
    day_open = gd["s_open"].transform("first")
    day_high = gd["s_high"].cummax()
    day_low = gd["s_low"].cummin()
    newh = df["s_high"] >= day_high
    newl = df["s_low"] <= day_low
    t_high = pd.Series(np.where(newh, tmin(df["hms_high"]), np.nan), index=df.index).groupby([inst, df["day"]]).ffill()
    t_low = pd.Series(np.where(newl, tmin(df["hms_low"]), np.nan), index=df.index).groupby([inst, df["day"]]).ffill()
    t_now = pd.Series(tmin(df["date"].dt.strftime("%H%M%S")), index=df.index)

{formula}

    x = pd.Series(np.asarray(x, dtype="float64"), index=df.index).replace([np.inf, -np.inf], np.nan)
    df["x"] = x
    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    pool = dai.query(f"SELECT date, instrument FROM {{instruments_table}}", filters={{"date": [query_start, end_date]}}, compression=True).df()
    pool["date"] = pd.to_datetime(pool["date"])
    res = pd.merge(pool, df[["date", "instrument", "x"]], how="left", on=["date", "instrument"]).drop_duplicates(["date", "instrument"])
    res["factor"] = res.groupby("date")["x"].rank(pct=True) - 0.5
    res["factor"] = res["factor"].fillna(0.0)
    hhmm = res["date"].dt.hour * 100 + res["date"].dt.minute
    res.loc[hhmm == 930, "factor"] = 0.0
    SMOOTH = {smooth}
    if SMOOTH > 0:
        res = res.sort_values(["instrument", "date"])
        res["factor"] = res.groupby("instrument", sort=False)["factor"].transform(lambda q: q.ewm(alpha=1.0 - SMOOTH, adjust=False).mean())
        res.loc[hhmm.loc[res.index] == 930, "factor"] = 0.0
    res = res[(res["date"] >= pd.to_datetime(start_date)) & (res["date"] <= pd.to_datetime(end_date))]
    res["factor"] = res["factor"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)
    return res[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)
'''

# 名字 → (说明, 公式（给 x 赋值，缩进 4 格）, 平滑 α（过去的权重）, 回看天数)
D8 = "    k = 8\n"
B = {}
B["s1"] = {
    "rev_base": ("对照：段收益反转 −r_seg", "    x = -r_seg", 0.0, 10),
    "rev_base_sm": ("对照：段收益反转（平滑 0.6）", "    x = -r_seg", 0.6, 10),
    "skew_d": ("一天内（8 段）分钟收益偏度取负：彩票型上冲之后回落",
               D8 + "    n = roll('nmin', k)\n    m2, m3 = roll('sr2', k) / n, roll('sr3', k) / n\n    x = -(m3 / m2.where(m2 > 0) ** 1.5)", 0.0, 10),
    "skew_s": ("本段分钟收益偏度取负", "    n = df['nmin']\n    m2, m3 = df['sr2'] / n, df['sr3'] / n\n    x = -(m3 / m2.where(m2 > 0) ** 1.5)", 0.0, 10),
    "kurt_d": ("一天内分钟收益峰度取负", D8 + "    n = roll('nmin', k)\n    m2, m4 = roll('sr2', k) / n, roll('sr4', k) / n\n    x = -(m4 / m2.where(m2 > 0) ** 2)", 0.0, 10),
    "rsj_d": ("一天内带符号跳跃变差 (上行半方差 − 下行半方差)/总方差 取负", D8 + "    x = -(roll('sr2p', k) - roll('sr2n', k)) / pos(roll('sr2', k))", 0.0, 10),
    "rsj_s": ("本段带符号跳跃变差取负", "    x = -(df['sr2p'] - df['sr2n']) / pos(df['sr2'])", 0.0, 10),
    "rsj_s_sm": ("本段带符号跳跃变差取负（平滑 0.6）", "    x = -(df['sr2p'] - df['sr2n']) / pos(df['sr2'])", 0.6, 10),
    "er_noise_rev": ("噪声型波动的反转：−r_seg·(1−效率)，效率 = |Σr|/Σ|r|（路径越曲折越像流动性噪声）",
                     "    er = (df['sr'].abs() / pos(df['sabs'])).clip(0, 1)\n    x = -r_seg * (1 - er)", 0.0, 10),
    "er_eff_rev": ("对照：高效率波动的反转 −r_seg·效率", "    er = (df['sr'].abs() / pos(df['sabs'])).clip(0, 1)\n    x = -r_seg * er", 0.0, 10),
    "er_noise_rev_sm": ("噪声型波动的反转（平滑 0.6）", "    er = (df['sr'].abs() / pos(df['sabs'])).clip(0, 1)\n    x = -r_seg * (1 - er)", 0.6, 10),
    "flowcorr_d": ("一天内价格对签名成交量的相关（订单流推动价格的程度）", D8 + "    x = roll('srsv', k) / np.sqrt(pos(roll('sr2', k)) * pos(roll('ssv2', k)))", 0.0, 10),
    "impact_asym_d": ("一天内买方冲击 − 卖方冲击（相关口径）：买入更能推高价格 → 卖盘薄",
                      D8 + "    bp = roll('srsvp', k) / np.sqrt(pos(roll('sr2', k)) * pos(roll('ssv2p', k)))\n    bn = roll('srsvn', k) / np.sqrt(pos(roll('sr2', k)) * pos(roll('ssv2n', k)))\n    x = bp - bn", 0.0, 10),
    "amihud_abn": ("非流动性异常：一天 Σ|r|/成交额 相对 5 天均值（流动性突然变差）",
                   D8 + "    il = roll('sabs', k) / pos(roll('amt', k))\n    il5 = roll('sabs', 40) / pos(roll('amt', 40))\n    x = np.log(il / il5)", 0.0, 10),
    "zero_d": ("一天内零收益分钟占比（报价停滞，流动性差）", D8 + "    x = roll('nzero', k) / pos(roll('nmin', k))", 0.0, 10),
    "upvol_d": ("一天内上涨分钟成交量占比取负（买方推动的量 → 回吐）", D8 + "    x = -(roll('vup', k) - roll('vdn', k)) / pos(roll('vup', k) + roll('vdn', k))", 0.0, 10),
    "ext_asym_d": ("一天内分钟极值不对称取负：−(最大分钟涨幅 + 最大分钟跌幅)",
                   "    mx = df['rmax'].groupby(inst).transform(lambda q: q.rolling(8, min_periods=1).max())\n    mn = df['rmin'].groupby(inst).transform(lambda q: q.rolling(8, min_periods=1).min())\n    x = -(mx + mn)", 0.0, 10),
    "tpos_s": ("本段成交位置 (收−买一)/(卖一−买一) 均值取负（主动买成交偏多 → 回吐）", "    x = -df['tpos']", 0.0, 10),
    "tpos_s_sm": ("本段成交位置取负（平滑 0.6）", "    x = -df['tpos']", 0.6, 10),
    "gap12_s": ("盘口空档不对称：(卖二−卖一)−(买一−买二) 相对中间价（卖方一侧更空 → 上行阻力小）", "    x = df['gap12']", 0.3, 10),
    "gap13_s": ("盘口空档不对称（到三档）", "    x = df['gap13']", 0.3, 10),
    "shadow_s": ("本段下影线 − 上影线（相对价格）：下方有承接、上方有抛压",
                 "    hi, lo = df['s_high'], df['s_low']\n    op = df['s_open']\n    cl = px\n    x = ((np.minimum(op, cl) - lo) - (hi - np.maximum(op, cl))) / pos(cl)", 0.0, 10),
    "shadow_d": ("当天至今下影线 − 上影线（相对价格）", "    op = day_open\n    x = ((np.minimum(op, px) - day_low) - (day_high - np.maximum(op, px))) / pos(px)", 0.0, 10),
    "toh_d": ("日内高点／低点出现的时间：(距高点分钟 − 距低点分钟)/已开盘分钟；刚创新高 → 回落",
              "    el = (t_now + 1).clip(lower=1)\n    x = ((t_now - t_high) - (t_now - t_low)) / el", 0.0, 10),
    "volexp_rev": ("波动放大时的反转：−r_seg × 本段波动/一天波动",
                   D8 + "    ve = np.sqrt(df['sr2'] / pos(df['nmin'])) / np.sqrt(pos(roll('sr2', k)) / pos(roll('nmin', k)))\n    x = -r_seg * ve.clip(0, 5)", 0.0, 10),
}

# s2：市场模型（分钟截面平均收益 rm）一族 + 慢的高阶矩 + 方差比
K40 = "    k = 40\n"
BETA = "    b = roll('srrm', k) / pos(roll('srm2', k))\n"
B["s2"] = {
    "ivol_d": ("一天内特质波动取负（市场模型残差，分钟级）：低特质波动", D8 + BETA + "    iv = roll('sr2', k) - roll('srrm', k) ** 2 / pos(roll('srm2', k))\n    x = -np.sqrt(pos(iv) / pos(roll('nmin', k)))", 0.0, 10, "mkt"),
    "ivol_5d": ("五天特质波动取负", K40 + BETA + "    iv = roll('sr2', k) - roll('srrm', k) ** 2 / pos(roll('srm2', k))\n    x = -np.sqrt(pos(iv) / pos(roll('nmin', k)))", 0.0, 10, "mkt"),
    "ivol_ratio": ("特质波动占比取负：1 − R²（一天）", D8 + "    r2 = roll('srrm', k) ** 2 / (pos(roll('srm2', k)) * pos(roll('sr2', k)))\n    x = r2", 0.0, 10, "mkt"),
    "beta_d": ("一天内分钟贝塔取负（日内版低贝塔）", D8 + BETA + "    x = -b", 0.0, 10, "mkt"),
    "beta_5d": ("五天分钟贝塔取负", K40 + BETA + "    x = -b", 0.0, 10, "mkt"),
    "coskew_d": ("一天内协偏度取负：E[r·rm²]/(σ_r·E[rm²])", D8 + "    n = pos(roll('nmin', k))\n    x = -(roll('srrm2', k) / n) / (np.sqrt(pos(roll('sr2', k)) / n) * pos(roll('srm2', k)) / n)", 0.0, 10, "mkt"),
    "coskew_5d": ("五天协偏度取负", K40 + "    n = pos(roll('nmin', k))\n    x = -(roll('srrm2', k) / n) / (np.sqrt(pos(roll('sr2', k)) / n) * pos(roll('srm2', k)) / n)", 0.0, 10, "mkt"),
    "mres_rev": ("市场模型残差的段反转：−(Σr − β·Σrm)，β 用五天分钟数据", K40 + BETA + "    x = -(df['sr'] - b * df['srm'])", 0.0, 10, "mkt"),
    "mres_rev_sm": ("市场模型残差的段反转（平滑 0.6）", K40 + BETA + "    x = -(df['sr'] - b * df['srm'])", 0.6, 10, "mkt"),
    "mres_rev_vn": ("市场模型残差的段反转，除以五天特质波动", K40 + BETA + "    iv = roll('sr2', k) - roll('srrm', k) ** 2 / pos(roll('srm2', k))\n    sd = np.sqrt(pos(iv) / pos(roll('nmin', k)) * pos(df['nmin']))\n    x = -(df['sr'] - b * df['srm']) / sd", 0.6, 10, "mkt"),
    "delay_d": ("价格滞后：对上一分钟市场收益的响应占比（反应慢的股票）", K40 + "    bl = roll('srrml', k) / pos(roll('srml2', k))\n    b0 = roll('srrm', k) / pos(roll('srm2', k))\n    x = bl / (b0.abs() + bl.abs() + 1e-6)", 0.0, 10, "mkt"),
    "catchup": ("滞后补涨：滞后贝塔 × 本段市场收益（反应慢的股票随后补上市场的走势）", K40 + "    bl = roll('srrml', k) / pos(roll('srml2', k))\n    x = bl * df['srm']", 0.0, 10, "mkt"),
    "rsj_2d": ("两天带符号跳跃变差取负", "    k = 16\n    x = -(roll('sr2p', k) - roll('sr2n', k)) / pos(roll('sr2', k))", 0.0, 10),
    "rsj_5d": ("五天带符号跳跃变差取负", K40 + "    x = -(roll('sr2p', k) - roll('sr2n', k)) / pos(roll('sr2', k))", 0.0, 10),
    "skew_5d": ("五天分钟收益偏度取负", K40 + "    n = roll('nmin', k)\n    m2, m3 = roll('sr2', k) / n, roll('sr3', k) / n\n    x = -(m3 / m2.where(m2 > 0) ** 1.5)", 0.0, 10),
    "lottery_d": ("彩票偏好合成：一天的带符号跳跃变差、偏度、极值不对称（各自排名）等权取负",
                  D8 + "    n = roll('nmin', k)\n    m2, m3 = roll('sr2', k) / n, roll('sr3', k) / n\n    a1 = -(roll('sr2p', k) - roll('sr2n', k)) / pos(roll('sr2', k))\n    a2 = -(m3 / m2.where(m2 > 0) ** 1.5)\n    mx = df['rmax'].groupby(inst).transform(lambda q: q.rolling(8, min_periods=1).max())\n    mn = df['rmin'].groupby(inst).transform(lambda q: q.rolling(8, min_periods=1).min())\n    a3 = -(mx + mn)\n    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = rk(a1).fillna(0.5) + rk(a2).fillna(0.5) + rk(a3).fillna(0.5)", 0.0, 10),
    "vr_d": ("方差比取负：一天内段收益平方和 / 分钟收益平方和（趋势性强 → 回吐）", D8 + "    x = -(roll(r_seg ** 2, k) / pos(roll('sr2', k)))", 0.0, 10),
    "vr_rev": ("低方差比时的反转：−r_seg / 方差比（分钟层面来回震荡的股票段收益更像噪声）", D8 + "    vr = (roll(r_seg ** 2, k) / pos(roll('sr2', k))).clip(0.2, 5)\n    x = -r_seg / vr", 0.6, 10),
    "toh_sm": ("日内高低点时间（平滑 0.6）", "    el = (t_now + 1).clip(lower=1)\n    x = ((t_now - t_high) - (t_now - t_low)) / el", 0.6, 10),
    "toh_hi": ("距日内高点的分钟占比（只看高点）", "    el = (t_now + 1).clip(lower=1)\n    x = (t_now - t_high) / el", 0.0, 10),
    "toh_lo": ("距日内低点的分钟占比取负（只看低点）", "    el = (t_now + 1).clip(lower=1)\n    x = -(t_now - t_low) / el", 0.0, 10),
}

# s3：最强信息源"委托结构"的变体（平台脚本直接算，不经过 h1 特征）
def _q(expr, doc, a=0.3, k=None):
    f = "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n"
    if k:
        f += "    x = " + expr.replace("COL(", "roll(") .replace(")", ", %d) / pos(roll(df['nmin'].where(df['nmin'] > 0, 0) * 0 + 1, %d))" % (k, k)) if False else ""
    f += "    x = " + expr
    return (doc, f, a, 10, "queue")
B["s3"] = {
    "q_ctrl": _q("rk(df['qn23']).fillna(0.5) + rk(-df['qs1']).fillna(0.5)", "对照：深档笔数失衡 + 一档委托规模比取负（= 现有 Q），平滑 0.3"),
    "q_ctrl_a0": _q("rk(df['qn23']).fillna(0.5) + rk(-df['qs1']).fillna(0.5)", "对照 Q，不平滑", 0.0),
    "q_ctrl_a5": _q("rk(df['qn23']).fillna(0.5) + rk(-df['qs1']).fillna(0.5)", "对照 Q，平滑 0.5", 0.5),
    "q_ctrl_a7": _q("rk(df['qn23']).fillna(0.5) + rk(-df['qs1']).fillna(0.5)", "对照 Q，平滑 0.7", 0.7),
    "q_n23": _q("df['qn23']", "深档（2+3 档）笔数失衡"),
    "q_n2": _q("df['qn2']", "二档笔数失衡"),
    "q_n3": _q("df['qn3']", "三档笔数失衡"),
    "q_n1": _q("df['qn1']", "一档笔数失衡"),
    "q_v23": _q("df['qv23']", "深档挂单量失衡（对照：量 vs 笔数）"),
    "q_n23_minus_v23": _q("rk(df['qn23']).fillna(0.5) - rk(df['qv23']).fillna(0.5)", "深档笔数失衡超出量失衡的部分（买方是很多小单、卖方是少数大单）"),
    "q_s1": _q("-df['qs1']", "一档平均委托规模比取负"),
    "q_s23": _q("-df['qs23']", "深档平均委托规模比取负"),
    "q_th": _q("df['qth']", "队伍厚度不对称：买方深档/一档 − 卖方深档/一档（对数）"),
    "q_n23L": _q("df['qn23L']", "段内最后 10 分钟的深档笔数失衡"),
    "q_n23E": _q("df['qn23E']", "段末快照的深档笔数失衡"),
    "q_s1L": _q("-df['qs1L']", "最后 10 分钟的一档委托规模比取负"),
    "q_mix4": _q("rk(df['qn23']).fillna(0.5) + rk(-df['qs1']).fillna(0.5) + rk(df['qn2']).fillna(0.5) + rk(-df['qs23']).fillna(0.5)", "四项：深档笔数、一档规模比、二档笔数、深档规模比"),
    "q_mix_late": _q("rk(df['qn23L']).fillna(0.5) + rk(-df['qs1L']).fillna(0.5)", "Q 的最后 10 分钟版本"),
    "q_slow2d": _q("rk(roll(df['qn23'], 16) / 16).fillna(0.5) + rk(-roll(df['qs1'], 16) / 16).fillna(0.5)", "Q 的两天均值（慢、低换手）", 0.0),
    "q_trend": _q("rk(df['qn23'] - roll(df['qn23'], 8) / 8).fillna(0.5) + rk(-(df['qs1'] - roll(df['qs1'], 8) / 8)).fillna(0.5)", "Q 相对一天均值的变化（新增的排队压力）"),
}

# s4：日内特质低波动（s2 ivol_d 单独 0.662、换手 0.094；加到 1628 上估分 +0.006）的变体
IV = "    iv = roll('sr2', k) - roll('srrm', k) ** 2 / pos(roll('srm2', k))\n    ivs = np.sqrt(pos(iv) / pos(roll('nmin', k)))\n"
B["s4"] = {
    "ivol_d_sm5": ("一天特质波动取负（平滑 0.5）", D8 + IV + "    x = -ivs", 0.5, 10, "mkt"),
    "ivol_2d": ("两天特质波动取负", "    k = 16\n" + IV + "    x = -ivs", 0.0, 10, "mkt"),
    "ivol_half": ("半天（4 段）特质波动取负", "    k = 4\n" + IV + "    x = -ivs", 0.0, 10, "mkt"),
    "ivol_abn": ("特质波动异常：一天 / 五天（对数）取负（波动突然放大 → 回落）",
                 D8 + IV + "    a1 = ivs\n    k = 40\n" + IV + "    x = -np.log(a1 / ivs)", 0.0, 10, "mkt"),
    "tvol_d": ("对照：一天总波动（分钟收益平方和）取负", D8 + "    x = -np.sqrt(pos(roll('sr2', k)) / pos(roll('nmin', k)))", 0.0, 10, "mkt"),
    "dvol_d": ("一天下行半方差取负（下跌波动大的股票）", D8 + "    x = -np.sqrt(pos(roll('sr2n', k)) / pos(roll('nmin', k)))", 0.0, 10, "mkt"),
    "uvol_d": ("一天上行半方差取负（上冲波动大的股票，彩票型）", D8 + "    x = -np.sqrt(pos(roll('sr2p', k)) / pos(roll('nmin', k)))", 0.0, 10, "mkt"),
    "ivol_x_rev": ("特质波动与反转的交互：低特质波动股票的段反转更可靠 → −r_seg × 排名(−特质波动)",
                   D8 + IV + "    w = pd.Series(np.asarray(-ivs, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = -r_seg * w", 0.6, 10, "mkt"),
}

# s5（10-21，用户："你应该探索很多新方向"）：一批新方向，跑 2020–2024 看逐年稳定
SLOTK = "    slot = df['date'].dt.strftime('%H%M')\n"
ABN = SLOTK + "    lv = np.log(pos(df['vol']))\n    base = lv.groupby([inst, slot]).transform(lambda q: q.shift(1).rolling(20, min_periods=8).mean())\n    abn = lv - base\n"
TR = "    lp = np.log(px)\n    t = pd.Series(np.arange(len(df), dtype='float64'), index=df.index)\n    sx, sy = roll(t, 8), roll(lp, 8)\n    sxx, sxy, syy = roll(t * t, 8), roll(t * lp, 8), roll(lp * lp, 8)\n    n = 8.0\n    vx, vy, cxy = sxx - sx * sx / n, syy - sy * sy / n, sxy - sx * sy / n\n    slope = cxy / vx.where(vx > 0)\n    r2 = (cxy ** 2 / (vx * vy).where((vx * vy) > 0)).clip(0, 1)\n    tstat = slope / np.sqrt(((vy - slope * cxy) / (n - 2)).clip(lower=1e-12) / vx.where(vx > 0))\n"
B["s5"] = {
    # A 市场走势经由贝塔的部分（mres_rev 扣掉的那一半）：贝塔高的股票跟着大盘多走的部分回吐
    "a_beta_mkt": ("贝塔 × 本段市场收益 取负：跟着大盘多走的部分回吐", K40 + BETA + "    x = -b * df['srm']", 0.6, 10, "mkt"),
    "a_beta_mkt_d": ("贝塔 × 一天市场收益 取负", K40 + BETA + "    x = -b * roll('srm', 8)", 0.0, 10, "mkt"),
    # B 分钟收益自相关：正自相关（趋势在分钟层面延续）vs 负自相关（买卖价反弹、流动性提供）
    "b_ac1_d": ("一天内分钟收益一阶自相关取负", D8 + "    x = -roll('srl', k) / pos(roll('sr2', k))", 0.0, 10, "ac"),
    "b_ac1_s": ("本段分钟收益一阶自相关取负", "    x = -df['srl'] / pos(df['sr2'])", 0.3, 10, "ac"),
    "b_roll_d": ("Roll 价差估计（−协方差的平方根，一天）：隐含交易成本高 → 流动性溢价", D8 + "    cv = -roll('srl', k) / pos(roll('nrl', k))\n    x = np.sqrt(cv.clip(lower=0))", 0.0, 10, "ac"),
    "b_ac_rev": ("自相关为负时的反转更可靠：−r_seg × (1 − 自相关排名)", D8 + "    ac = roll('srl', k) / pos(roll('sr2', k))\n    w = 1 - pd.Series(np.asarray(ac, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = -r_seg * w", 0.6, 10, "ac"),
    # C 成交量相对同时段常态（20 日）：信息到达
    "c_abnvol": ("本段成交量相对同时段 20 日常态（对数）取负", ABN + "    x = -abn", 0.3, 45),
    "c_abnvol_rev": ("放量段的反转：−r_seg × 放量程度（截断到 0～3）", ABN + "    x = -r_seg * abn.clip(0, 3)", 0.6, 45),
    "c_abnvol_mom": ("缩量段的延续：r_seg × 缩量程度", ABN + "    x = r_seg * (-abn).clip(0, 3)", 0.6, 45),
    "c_abndeal": ("本段成交笔数相对同时段 20 日常态取负", SLOTK + "    lv = np.log(pos(df['ndeal']))\n    base = lv.groupby([inst, slot]).transform(lambda q: q.shift(1).rolling(20, min_periods=8).mean())\n    x = -(lv - base)", 0.3, 45),
    "c_size_abn": ("单笔成交额相对常态：本段 成交额/笔数 相对同时段 20 日常态（大单涌入）", SLOTK + "    lv = np.log(pos(df['amt']) / pos(df['ndeal']))\n    base = lv.groupby([inst, slot]).transform(lambda q: q.shift(1).rolling(20, min_periods=8).mean())\n    x = -(lv - base)", 0.3, 45),
    # D 趋势质量：过去 8 段（一天）对数价格对时间回归
    "d_trend_t": ("一天的价格趋势 t 值取负（强趋势回吐）", TR + "    x = -tstat", 0.0, 10),
    "d_trend_clean_rev": ("干净趋势的反转：−一天收益 × R²", TR + "    x = -(lp - lp.groupby(inst).shift(8)) * r2", 0.0, 10),
    "d_trend_noisy_rev": ("杂乱路径的反转：−一天收益 × (1 − R²)", TR + "    x = -(lp - lp.groupby(inst).shift(8)) * (1 - r2)", 0.0, 10),
    # E 盘口深度与价差的变化
    "e_spr_widen": ("段末价差相对段内均值（对数）取负：价差突然变宽", "    x = -np.log(pos(df['spr_end']) / pos(df['spr_avg']))", 0.3, 10, "book"),
    "e_spr_rel5d": ("本段平均价差相对五天均值取负：流动性变差", "    x = -np.log(pos(df['spr_avg']) / pos(roll('spr_avg', 40) / 40))", 0.3, 10, "book"),
    "e_dep_chg": ("段内三档总深度变化（段末/段首，对数）：流动性撤出 vs 补充", "    x = np.log(pos(df['dep_end']) / pos(df['dep_first']))", 0.3, 10, "book"),
    "e_dep_rel5d": ("本段平均深度相对五天均值", "    x = np.log(pos(df['dep_avg']) / pos(roll('dep_avg', 40) / 40))", 0.3, 10, "book"),
    "e_dimb_end": ("段末三档量失衡（买 − 卖）/(买 + 卖)", "    x = df['dimb_end']", 0.3, 10, "book"),
    "e_dimb_chg": ("三档量失衡的段内变化（段末 − 段内均值）", "    x = df['dimb_end'] - df['dimb_avg']", 0.3, 10, "book"),
    "e_dimb_d": ("一天三档量失衡均值", D8 + "    x = roll('dimb_avg', k) / k", 0.0, 10, "book"),
    # J 价格加速度、跳过一段的反转
    "j_accel": ("价格加速度取负：−(本段收益 − 上一段收益)", "    x = -(r_seg - r_seg.groupby(inst).shift(1))", 0.3, 10),
    "j_skip1": ("跳过最近一段的反转：−上一段收益", "    x = -r_seg.groupby(inst).shift(1)", 0.3, 10),
    "j_rev2": ("两段（1 小时）反转", "    x = -(r_seg + r_seg.groupby(inst).shift(1))", 0.5, 10),
}

# s6（10-21）：日线级锚点、资金流与区间——全部只用当天截面之前的数据
DAYAGG = ("    dh = df.groupby([inst, df['day']])['s_high'].transform('max')\n"
          "    dl = df.groupby([inst, df['day']])['s_low'].transform('min')\n"
          "    dd = df[['instrument', 'day']].assign(dh=dh, dl=dl).drop_duplicates(['instrument', 'day'])\n"
          "    dd[['ph', 'pl']] = dd.groupby('instrument')[['dh', 'dl']].shift(1)\n"
          "    m = df[['instrument', 'day']].merge(dd[['instrument', 'day', 'ph', 'pl']], on=['instrument', 'day'], how='left')\n"
          "    ph, pl = pd.Series(m['ph'].to_numpy(), index=df.index), pd.Series(m['pl'].to_numpy(), index=df.index)\n")
CLV = "    clv = ((2 * px - df['s_low'] - df['s_high']) / (df['s_high'] - df['s_low']).where(df['s_high'] > df['s_low'])).clip(-1, 1)\n"
B["s6"] = {
    "gap_rev": ("隔夜跳空的日内回补：−ln(当日开盘/昨收)，全天同值", "    x = -np.log(pos(day_open) / pos(df['pre_close']))", 0.0, 10),
    "gap_fill": ("跳空回补进度：(现价 − 开盘)/(昨收 − 开盘)（已回补越多，剩余越少）", "    den = (df['pre_close'] - day_open)\n    x = -((px - day_open) / den.where(den.abs() > 1e-6 * px)).clip(-3, 3)", 0.3, 10),
    "prev_rpos": ("相对昨日价格区间的位置取负：(现价 − 昨低)/(昨高 − 昨低)", DAYAGG + "    x = -((px - pl) / (ph - pl).where(ph > pl)).clip(-2, 3)", 0.3, 10),
    "prev_break": ("突破昨日高点的程度（现价/昨高 − 1，只取正）取负", DAYAGG + "    x = -np.log(pos(px) / pos(ph)).clip(lower=0)", 0.3, 10),
    "vwap_loc_d": ("当天成交重心在区间中的位置取负：(当日 VWAP − 当日低)/(当日高 − 当日低)（量堆在高位 = 高位派发）",
                   "    cv, ca = gd['vol'].cumsum(), gd['amt'].cumsum()\n    vw = ca / cv.where(cv > 0)\n    x = -((vw - day_low) / (day_high - day_low).where(day_high > day_low))", 0.3, 10),
    "vwap_vs_twap_d": ("量价重心：当日 VWAP 相对等权均价（各段收盘均值）取负", "    cv, ca = gd['vol'].cumsum(), gd['amt'].cumsum()\n    vw = ca / cv.where(cv > 0)\n    tw = gd['last_px'].transform(lambda q: q.expanding().mean())\n    x = -np.log(pos(vw) / pos(tw))", 0.3, 10),
    "cmf_d": ("一天的资金流（Chaikin）取负：Σ量·段内收盘位置 / Σ量", D8 + CLV + "    x = -roll(df['vol'] * clv, k) / pos(roll('vol', k))", 0.0, 10),
    "cmf_5d": ("五天的资金流取负", "    k = 40\n" + CLV + "    x = -roll(df['vol'] * clv, k) / pos(roll('vol', k))", 0.0, 10),
    "range_exp_rev": ("区间扩张时的反转：−r_seg × (本段高低区间 / 五天平均段区间)", "    rg = np.log(pos(df['s_high']) / pos(df['s_low']))\n    x = -r_seg * (rg / pos(roll(rg, 40) / 40)).clip(0, 5)", 0.6, 10),
    "range_d": ("一天高低区间（相对价格）取负：日内振幅大的股票", D8 + "    x = -np.log(pos(day_high) / pos(day_low))", 0.0, 10),
    "amt_5d20d": ("关注度消退：五天成交额 / 二十天成交额（对数）取负", "    x = -np.log(pos(roll('amt', 40)) / pos(roll('amt', 160)) * 4)", 0.0, 45),
    "amt_1d5d": ("一天成交额 / 五天成交额（对数）取负", "    x = -np.log(pos(roll('amt', 8)) / pos(roll('amt', 40)) * 5)", 0.0, 10),
}

# s7（10-21，用户："低换手的新方向挺好，继续多探索新方向并记录"）：多日锚点、隔夜与日内的拉锯、趋势平滑度的尺度、昨日的盘中行为
DAILY = ("    _d = df.assign(_hi=df['s_high'], _lo=df['s_low'], _cl=px, _op=df['s_open']).groupby(['instrument', 'day'], sort=True)\n"
         "    dd = pd.DataFrame({'dh': _d['_hi'].max(), 'dl': _d['_lo'].min(), 'dc': _d['_cl'].last(), 'do': _d['_op'].first(),\n"
         "                       'da': _d['amt'].sum(), 'dv': _d['vol'].sum(), 'pc': _d['pre_close'].max()}).reset_index()\n"
         "    gi = dd.groupby('instrument', sort=False)\n"
         "    def dprev(col, k, fn):\n"
         "        return gi[col].transform(lambda q: getattr(q.shift(1).rolling(k, min_periods=max(1, k // 2)), fn)())\n"
         "    def dmap(v):\n"
         "        m = df[['instrument', 'day']].merge(dd[['instrument', 'day']].assign(_v=np.asarray(v)), on=['instrument', 'day'], how='left')\n"
         "        return pd.Series(m['_v'].to_numpy(), index=df.index)\n"
         "    dd['on'] = np.log(pos(dd['do']) / pos(dd['pc']))\n"
         "    dd['id'] = np.log(pos(dd['dc']) / pos(dd['do']))\n")
def RPOS(k):
    return DAILY + "    hk, lk = dmap(dprev('dh', %d, 'max')), dmap(dprev('dl', %d, 'min'))\n    x = -((px - lk) / (hk - lk).where(hk > lk)).clip(-2, 3)" % (k, k)
B["s7"] = {
    "rpos_2d": ("相对过去 2 日价格区间的位置取负", RPOS(2), 0.3, 20),
    "rpos_3d": ("相对过去 3 日价格区间的位置取负", RPOS(3), 0.3, 20),
    "rpos_5d": ("相对过去 5 日价格区间的位置取负", RPOS(5), 0.3, 20),
    "rpos_10d": ("相对过去 10 日价格区间的位置取负", RPOS(10), 0.3, 30),
    "rpos_1d_raw": ("相对昨日区间的位置取负（不平滑，对照 s6_prev_rpos）", RPOS(1), 0.0, 10),
    "vwap5d": ("相对过去 5 日成交均价（VWAP）的偏离取负", DAILY + "    vw = dmap(gi['da'].transform(lambda q: q.shift(1).rolling(5, min_periods=3).sum()) / gi['dv'].transform(lambda q: q.shift(1).rolling(5, min_periods=3).sum()).where(lambda z: z > 0))\n    x = -np.log(pos(px) / pos(vw))", 0.3, 20),
    "prev_clv": ("昨日收盘在昨日区间中的位置取负（昨天收在高位 → 回落）", DAILY + "    v = (2 * dd['dc'] - dd['dh'] - dd['dl']) / (dd['dh'] - dd['dl']).where(dd['dh'] > dd['dl'])\n    x = -dmap(gi['dc'].transform(lambda q: q * 0) + v.groupby(dd['instrument']).shift(1))", 0.0, 10),
    "prev_close_vwap": ("昨日收盘相对昨日 VWAP 取负（尾盘推高 → 回落）", DAILY + "    v = np.log(pos(dd['dc']) / pos(dd['da'] / dd['dv'].where(dd['dv'] > 0)))\n    x = -dmap(v.groupby(dd['instrument']).shift(1))", 0.0, 10),
    "tug_on5": ("隔夜收益 5 日和（隔夜买盘延续）", DAILY + "    x = dmap(gi['on'].transform(lambda q: q.rolling(5, min_periods=3).sum()))", 0.0, 20),
    "tug_id5": ("日内收益 5 日和取负（日内卖压回吐；不含今天）", DAILY + "    x = -dmap(gi['id'].transform(lambda q: q.shift(1).rolling(5, min_periods=3).sum()))", 0.0, 20),
    "tug_diff5": ("拉锯：5 日隔夜和 − 5 日日内和（不含今天的日内）", DAILY + "    x = dmap(gi['on'].transform(lambda q: q.rolling(5, min_periods=3).sum())) - dmap(gi['id'].transform(lambda q: q.shift(1).rolling(5, min_periods=3).sum()))", 0.0, 20),
    "tug_diff20": ("拉锯：20 日隔夜和 − 20 日日内和", DAILY + "    x = dmap(gi['on'].transform(lambda q: q.rolling(20, min_periods=10).sum())) - dmap(gi['id'].transform(lambda q: q.shift(1).rolling(20, min_periods=10).sum()))", 0.0, 45),
    "max5d": ("过去 5 日最大的单日收益取负（彩票效应，短窗）", DAILY + "    r1 = np.log(pos(dd['dc']) / pos(dd['pc']))\n    x = -dmap(r1.groupby(dd['instrument']).transform(lambda q: q.shift(1).rolling(5, min_periods=3).max()))", 0.0, 20),
    "trend_clean_2d": ("两天（16 段）干净趋势的反转：−两天收益 × R²", TR.replace("8)", "16)").replace("n = 8.0", "n = 16.0") + "    x = -(lp - lp.groupby(inst).shift(16)) * r2", 0.0, 10),
    "trend_clean_half": ("半天（4 段）干净趋势的反转", TR.replace("8)", "4)").replace("n = 8.0", "n = 4.0") + "    x = -(lp - lp.groupby(inst).shift(4)) * r2", 0.3, 10),
    "trend_clean_sm": ("一天干净趋势的反转（平滑 0.5）", TR + "    x = -(lp - lp.groupby(inst).shift(8)) * r2", 0.5, 10),
    "trend_r2_only": ("一天价格路径的线性度 R² × 方向（只看路径是否单向，不看幅度）", TR + "    x = -np.sign(lp - lp.groupby(inst).shift(8)) * r2", 0.0, 10),
    "eff_2d": ("两天的路径效率取负：|两天净收益| / Σ|段收益| × 方向", "    k = 16\n    net = roll(r_seg, k)\n    tot = roll(r_seg.abs(), k)\n    x = -net / pos(tot) * (net.abs() / pos(tot))", 0.0, 10),
    "consist_2d": ("两天里上涨段的占比取负（一致的上涨 → 回落）", "    up = (r_seg > 0).astype(float)\n    x = -(roll(up, 16) / 16 - 0.5)", 0.0, 10),
    "prev_last30": ("昨日最后半小时收益取负（收盘段压力的次日回吐）", "    l30 = r_seg.where(df['date'].dt.strftime('%H%M') == '1500')\n    v = l30.groupby([inst, df['day']]).transform('max')\n    dd2 = df[['instrument', 'day']].assign(v=v).drop_duplicates(['instrument', 'day'])\n    dd2['pv'] = dd2.groupby('instrument')['v'].shift(1)\n    m = df[['instrument', 'day']].merge(dd2[['instrument', 'day', 'pv']], on=['instrument', 'day'], how='left')\n    x = -pd.Series(m['pv'].to_numpy(), index=df.index)", 0.0, 10),
}

# s8（10-21）：与反转相关低的新信息——跳跃、量价关系、波动结构、非流动性水平、签名成交量、20 日锚点、彩票族的尺度
B["s8"] = {
    "pvcorr_d": ("一天内分钟 |收益| 与成交量的相关取负（量价同步 = 信息驱动？）", D8 + "    n = pos(roll('nmin', k))\n    ma, mv = roll('sabs', k) / n, roll('svr', k) / n\n    cov = roll('srav', k) / n - ma * mv\n    va, vv = roll('sr2', k) / n - ma ** 2, roll('svv', k) / n - mv ** 2\n    x = -cov / np.sqrt(pos(va) * pos(vv))", 0.0, 10, "pv"),
    "pvcorr_5d": ("五天的分钟量价相关取负", K40 + "    n = pos(roll('nmin', k))\n    ma, mv = roll('sabs', k) / n, roll('svr', k) / n\n    cov = roll('srav', k) / n - ma * mv\n    va, vv = roll('sr2', k) / n - ma ** 2, roll('svv', k) / n - mv ** 2\n    x = -cov / np.sqrt(pos(va) * pos(vv))", 0.0, 10, "pv"),
    "jump_d": ("一天的跳跃强度取负：最大分钟涨跌幅 / 分钟波动（一天）", D8 + "    mx = df['rmax'].groupby(inst).transform(lambda q: q.rolling(8, min_periods=1).max())\n    mn = df['rmin'].groupby(inst).transform(lambda q: q.rolling(8, min_periods=1).min())\n    sd = np.sqrt(pos(roll('sr2', k)) / pos(roll('nmin', k)))\n    x = -(np.maximum(mx, -mn) / sd)", 0.0, 10),
    "upjump_d": ("一天的上跳强度取负：最大分钟涨幅 / 分钟波动", D8 + "    mx = df['rmax'].groupby(inst).transform(lambda q: q.rolling(8, min_periods=1).max())\n    sd = np.sqrt(pos(roll('sr2', k)) / pos(roll('nmin', k)))\n    x = -(mx / sd)", 0.0, 10),
    "amihud_5d": ("五天的非流动性水平：Σ|分钟收益| / 成交额（对数）", K40 + "    x = np.log(pos(roll('sabs', k)) / pos(roll('amt', k)))", 0.0, 10),
    "pk_ratio_5d": ("五天的区间波动 / 段收益波动取负（来回震荡的股票）", K40 + "    rg = np.log(pos(df['s_high']) / pos(df['s_low']))\n    x = -np.log(pos(roll(rg ** 2, k)) / pos(roll(r_seg ** 2, k)))", 0.0, 10),
    "vol_shock_d": ("当天分钟波动相对五天取负（波动突然放大）", D8 + "    a = roll('sr2', k) / pos(roll('nmin', k))\n    b = roll('sr2', 40) / pos(roll('nmin', 40))\n    x = -np.log(pos(a) / pos(b))", 0.0, 10),
    "updown_vol_5d": ("五天里下跌段与上涨段的成交量之比（对数）取负：放量下跌多 = 派发", K40 + "    dv = roll(df['vol'].where(r_seg < 0, 0.0), k)\n    uv = roll(df['vol'].where(r_seg > 0, 0.0), k)\n    x = -np.log(pos(dv) / pos(uv))", 0.0, 10),
    "bvc_1d": ("一天的签名成交量占比取负（分钟内方向分类：量 × (收 − 开)/(高 − 低)）", D8 + "    x = -roll('ssv', k) / pos(roll('vol', k))", 0.0, 10),
    "bvc_5d": ("五天的签名成交量占比取负", K40 + "    x = -roll('ssv', k) / pos(roll('vol', k))", 0.0, 10),
    "high20_prox": ("相对 20 日最高价的距离：ln(现价 / 过去 20 日最高)", DAILY + "    hk = dmap(dprev('dh', 20, 'max'))\n    x = np.log(pos(px) / pos(hk))", 0.0, 45),
    "low20_prox": ("相对 20 日最低价的距离取负：ln(现价 / 过去 20 日最低)", DAILY + "    lk = dmap(dprev('dl', 20, 'min'))\n    x = -np.log(pos(px) / pos(lk))", 0.0, 45),
    "days_since_high20": ("距 20 日高点的天数（越久没创新高 → ？）", DAILY + "    v = gi['dh'].transform(lambda q: q.shift(1).rolling(20, min_periods=10).apply(lambda a: len(a) - 1 - np.argmax(a), raw=True))\n    x = dmap(v)", 0.0, 45),
    "skew_2d": ("两天的分钟收益偏度取负", "    k = 16\n    n = roll('nmin', k)\n    m2, m3 = roll('sr2', k) / n, roll('sr3', k) / n\n    x = -(m3 / m2.where(m2 > 0) ** 1.5)", 0.0, 10),
    "uvol_5d": ("五天上行半方差取负", K40 + "    x = -np.sqrt(pos(roll('sr2p', k)) / pos(roll('nmin', k)))", 0.0, 10),
    "uvol_share_d": ("一天上行半方差占比取负（与 rsj 同族，换成占比）", D8 + "    x = -roll('sr2p', k) / pos(roll('sr2', k))", 0.0, 10),
    "maxmin_5d": ("五天最大分钟涨幅取负（短窗 MAX 效应，分钟级）", "    x = -df['rmax'].groupby(inst).transform(lambda q: q.rolling(40, min_periods=8).max())", 0.0, 10),
    "boll_2d": ("两天布林带位置取负：(对数价 − 16 段均值)/16 段标准差", "    lp = np.log(px)\n    m = lp.groupby(inst).transform(lambda q: q.rolling(16, min_periods=8).mean())\n    sd = lp.groupby(inst).transform(lambda q: q.rolling(16, min_periods=8).std())\n    x = -(lp - m) / sd.where(sd > 0)", 0.3, 10),
}

# s9（10-21）：日频经典异象（20 日换手波动、成交额趋势、日收益波动／偏度／最大值、5／20 日反转、60 日动量）与今天的盘中结构
DR1 = DAILY + "    dd['r1'] = np.log(pos(dd['dc']) / pos(dd['pc']))\n    dd['la'] = np.log(pos(dd['da']))\n"
def DROLL(col, k, fn, sign="-"):
    return DR1 + "    x = %sdmap(gi['%s'].transform(lambda q: q.shift(1).rolling(%d, min_periods=%d).%s()))" % (sign, col, k, k // 2, fn)
B["s9"] = {
    "turnvol_20d": ("20 日成交额波动取负（对数成交额的 20 日标准差：关注度不稳定）", DROLL("la", 20, "std"), 0.0, 45),
    "amt_trend_20d": ("20 日成交额趋势取负（对数成交额对时间的斜率，近似 = 后 10 日均值 − 前 10 日均值）", DR1 + "    a = gi['la'].transform(lambda q: q.shift(1).rolling(10, min_periods=5).mean())\n    b = gi['la'].transform(lambda q: q.shift(11).rolling(10, min_periods=5).mean())\n    x = -dmap(a - b)", 0.0, 45),
    "dvol_20d": ("20 日日收益波动取负（低波动异象）", DROLL("r1", 20, "std"), 0.0, 45),
    "dskew_20d": ("20 日日收益偏度取负", DROLL("r1", 20, "skew"), 0.0, 45),
    "dmax_20d": ("20 日最大单日收益取负（MAX 效应）", DROLL("r1", 20, "max"), 0.0, 45),
    "drev_5d": ("5 日反转（不含今天）", DROLL("r1", 5, "sum"), 0.0, 20),
    "drev_20d": ("20 日反转（不含今天）", DROLL("r1", 20, "sum"), 0.0, 45),
    "dmom_60_5": ("60 日动量（跳过最近 5 日）", DR1 + "    a = gi['r1'].transform(lambda q: q.shift(6).rolling(55, min_periods=30).sum())\n    x = dmap(a)", 0.0, 100),
    "open30_rev": ("开盘 30 分钟走势的回吐：−(10:00 价 / 昨收)，10:30 起各截面同值", "    t = df['date'].dt.strftime('%H%M')\n    p10 = px.where(t == '1000').groupby([inst, df['day']]).transform('max')\n    x = -np.log(pos(p10) / pos(df['pre_close'])).where(t > '1000')", 0.0, 10),
    "am_pm": ("午后相对上午的走势取负：−(现价/11:30 价)，13:30 起", "    t = df['date'].dt.strftime('%H%M')\n    p11 = px.where(t == '1130').groupby([inst, df['day']]).transform('max')\n    x = -np.log(pos(px) / pos(p11)).where(t > '1130')", 0.0, 10),
    "vol_front": ("今天成交前置程度取负：今天截至此刻的成交量占 5 日日均量，减去过去同一时刻的平均占比", SLOTK +
                  "    cv = gd['vol'].cumsum()\n    dv = df.groupby([inst, df['day']])['vol'].transform('sum')\n    adv = dv.groupby(inst).transform(lambda q: q.shift(9).rolling(45, min_periods=18).mean())\n    sh = cv / pos(adv)\n    typ = sh.groupby([inst, slot]).transform(lambda q: q.shift(1).rolling(20, min_periods=8).mean())\n    x = -(sh - typ)", 0.0, 45),
}

# s10（10-21）：快信息（c103：1610 缺的是快信息）——价格加速度的变体、段内收盘位置、段内签名成交量、按流动性调节的反转、午后／午前的回吐
B["s10"] = {
    "accel_sm3": ("价格加速度取负（平滑 0.3）", "    x = -(r_seg - r_seg.groupby(inst).shift(1))", 0.3, 10),
    "accel_sm6": ("价格加速度取负（平滑 0.6）", "    x = -(r_seg - r_seg.groupby(inst).shift(1))", 0.6, 10),
    "accel2": ("两段加速度取负：−(本段收益 − 前两段平均)", "    x = -(r_seg - 0.5 * (r_seg.groupby(inst).shift(1) + r_seg.groupby(inst).shift(2)))", 0.3, 10),
    "accel_vn": ("波动标准化的加速度取负：÷ 五天段收益波动", "    sd = r_seg.groupby(inst).transform(lambda q: q.rolling(40, min_periods=10).std())\n    x = -(r_seg - r_seg.groupby(inst).shift(1)) / sd.where(sd > 0)", 0.3, 10),
    "seg_clv": ("段内收盘位置取负：(2·收 − 高 − 低)/(高 − 低)", CLV + "    x = -clv", 0.3, 10),
    "seg_clv_vw": ("段内收盘相对段 VWAP 取负（÷ 段区间）", "    vw = df['amt'] / df['vol'].where(df['vol'] > 0)\n    x = -(px - vw) / (df['s_high'] - df['s_low']).where(df['s_high'] > df['s_low'])", 0.3, 10),
    "bvc_seg": ("本段签名成交量占比取负（分钟内方向分类）", "    x = -df['ssv'] / pos(df['vol'])", 0.3, 10),
    "rev_x_liq": ("流动性好的股票的反转更可靠：−r_段 × 成交额排名", "    w = np.log(pos(roll('amt', 40))).groupby(df['date']).rank(pct=True)\n    x = -r_seg * w", 0.6, 10),
    "rev_x_illiq": ("流动性差的股票的反转：−r_段 × (1 − 成交额排名)", "    w = np.log(pos(roll('amt', 40))).groupby(df['date']).rank(pct=True)\n    x = -r_seg * (1 - w)", 0.6, 10),
    "pm_from_open": ("午后相对开盘：−ln(现价/开盘)，13:30 起", "    t = df['date'].dt.strftime('%H%M')\n    x = -np.log(pos(px) / pos(day_open)).where(t > '1130')", 0.0, 10),
    "am_drift": ("上午相对 10:00 的走势取负（10:30–11:30）", "    t = df['date'].dt.strftime('%H%M')\n    p10 = px.where(t == '1000').groupby([inst, df['day']]).transform('max')\n    x = -np.log(pos(px) / pos(p10)).where((t > '1000') & (t <= '1130'))", 0.0, 10),
    "am_pm_full": ("午后相对 11:30、上午相对 10:00 的走势取负（全天都有值）", "    t = df['date'].dt.strftime('%H%M')\n    p10 = px.where(t == '1000').groupby([inst, df['day']]).transform('max')\n    p11 = px.where(t == '1130').groupby([inst, df['day']]).transform('max')\n    x = np.where(t > '1130', -np.log(pos(px) / pos(p11)), np.where(t > '1000', -np.log(pos(px) / pos(p10)), -np.log(pos(px) / pos(df['pre_close']))))", 0.0, 10),
    "half_rev": ("半天反转（4 段收益，平滑 0.3）", "    x = -roll(r_seg, 4)", 0.3, 10),
    "last_seg_vs_day": ("本段收益相对当天至今收益的占比取负（尾段冲刺）", "    dr = np.log(pos(px) / pos(df['pre_close']))\n    x = -(r_seg - dr / 8)", 0.3, 10),
}

# s11（10-21）：个股的反转倾向、成交高潮、开盘区间突破、收盘集合竞价占比、信息离散度、交互项
ACP = ("    r1 = r_seg.groupby(inst).shift(1)\n"
       "    k = 40\n"
       "    sxy, sxx, syy = roll((r_seg * r1).fillna(0), k), roll((r1 * r1).fillna(0), k), roll((r_seg * r_seg).fillna(0), k)\n"
       "    rho = sxy / np.sqrt(pos(sxx) * pos(syy))\n"
       "    rho = rho.groupby(inst).shift(1)\n")
VCL = ("    vm = df['vmax'].groupby([inst, df['day']]).cummax()\n"
       "    newm = df['vmax'] >= vm\n"
       "    cv = pd.Series(np.where(newm, df['c_vmax'], np.nan), index=df.index).groupby([inst, df['day']]).ffill()\n")
ORB = ("    t = df['date'].dt.strftime('%H%M')\n"
       "    oh = df['s_high'].where(t == '1000').groupby([inst, df['day']]).transform('max')\n"
       "    ol = df['s_low'].where(t == '1000').groupby([inst, df['day']]).transform('min')\n")
DISC = "    net = roll('sr', k)\n    npn = pos(roll('npos', k) + roll('nneg', k))\n    disc = np.sign(net) * (roll('nneg', k) - roll('npos', k)) / npn\n"
B["s11"] = {
    "acp_rev": ("按个股反转倾向调节的反转：−r_段 × 排名(−段收益自相关，过去 5 天)", ACP + "    w = (-rho).groupby(df['date']).rank(pct=True)\n    x = -r_seg * w", 0.6, 10),
    "acp_level": ("个股反转倾向本身：−段收益自相关（平时回吐多的股票）", ACP + "    x = -rho", 0.3, 10),
    "acp_mom": ("自相关为正的股票的段延续：r_段 × 排名(段收益自相关)", ACP + "    w = rho.groupby(df['date']).rank(pct=True)\n    x = r_seg * w", 0.6, 10, ),
    "vclimax_rev": ("相对当天成交高潮分钟价格的偏离取负：−ln(现价 / 当天最大量分钟的收盘价)", VCL + "    x = -np.log(pos(px) / pos(cv))", 0.3, 10, "vc"),
    "vclimax_w": ("同上，乘高潮强度（最大分钟量 / 当天平均分钟量）", VCL + "    inten = vm / pos(gd['vol'].cumsum() / pos(gd['nmin'].cumsum()))\n    x = -np.log(pos(px) / pos(cv)) * np.log1p(inten.clip(0, 50))", 0.3, 10, "vc"),
    "orb_pos": ("相对开盘区间（10:00 段）的位置取负：(现价 − 开盘区间高)/开盘区间宽度，10:30 起", ORB + "    x = -((px - oh) / (oh - ol).where(oh > ol)).clip(-5, 5).where(t > '1000')", 0.0, 10),
    "orb_mid": ("相对开盘区间中点的位置取负（÷ 宽度），10:30 起", ORB + "    x = -((px - (oh + ol) / 2) / (oh - ol).where(oh > ol)).clip(-5, 5).where(t > '1000')", 0.0, 10),
    "close_auction_share": ("昨日收盘集合竞价成交占全天比例取负（尾盘抢筹 → 次日回落）", "    vc = df.groupby([inst, df['day']])['v_close'].transform('sum')\n    dv = df.groupby([inst, df['day']])['vol'].transform('sum')\n    sh = (vc / pos(dv))\n    d2 = df[['instrument', 'day']].assign(sh=sh).drop_duplicates(['instrument', 'day'])\n    d2['p'] = d2.groupby('instrument')['sh'].shift(1)\n    m = df[['instrument', 'day']].merge(d2[['instrument', 'day', 'p']], on=['instrument', 'day'], how='left')\n    x = -pd.Series(m['p'].to_numpy(), index=df.index)", 0.0, 10, "vc"),
    "disc_2d": ("两天的信息离散度：sign(净收益) × (跌分钟占比 − 涨分钟占比)（大跳组成的走势）", "    k = 16\n" + DISC + "    x = disc", 0.0, 10, "vc"),
    "disc_rev_2d": ("离散型走势的反转：−两天收益 × 排名(离散度)", "    k = 16\n" + DISC + "    w = disc.groupby(df['date']).rank(pct=True)\n    x = -net * w", 0.0, 10, "vc"),
    "cont_mom_2d": ("连续型走势的延续：两天收益 × (1 − 排名(离散度))", "    k = 16\n" + DISC + "    w = 1 - disc.groupby(df['date']).rank(pct=True)\n    x = net * w", 0.0, 10, "vc"),
    "rev_x_lotto": ("彩票型股票的反转：−r_段 × 排名(一天上行半方差占比)", D8 + "    w = (roll('sr2p', k) / pos(roll('sr2', k))).groupby(df['date']).rank(pct=True)\n    x = -r_seg * w", 0.6, 10),
    "rev_x_anchor": ("靠近昨日高点时的反转：−r_段 × 排名(昨日区间位置)", DAYAGG + "    rp = ((px - pl) / (ph - pl).where(ph > pl)).clip(-2, 3)\n    w = rp.groupby(df['date']).rank(pct=True)\n    x = -r_seg * w", 0.6, 10),
}

# s12（10-21）：日内价格锚点一族（s10 上午相对 10:00 的回吐估 0.843）——换锚点、换覆盖时段、量加权
def ANCH(t0, cond):
    return ("    t = df['date'].dt.strftime('%H%M')\n"
            "    pa = px.where(t == '" + t0 + "').groupby([inst, df['day']]).transform('max')\n"
            "    x = -np.log(pos(px) / pos(pa)).where(" + cond + ")")
B["s12"] = {
    "anc1000_all": ("全天以 10:00 价为锚的回吐（10:30–14:30）", ANCH("1000", "t > '1000'"), 0.0, 10),
    "anc1000_pm": ("午后以 10:00 价为锚的回吐（13:30–14:30）", ANCH("1000", "t > '1130'"), 0.0, 10),
    "anc1030_all": ("以 10:30 价为锚的回吐（11:00–14:30）", ANCH("1030", "t > '1030'"), 0.0, 10),
    "anc1330_pm": ("以 13:30 价为锚的回吐（14:00–14:30）", ANCH("1330", "t > '1330'"), 0.0, 10),
    "anc_open_all": ("全天以今天开盘为锚的回吐（10:00–14:30）", "    t = df['date'].dt.strftime('%H%M')\n    x = -np.log(pos(px) / pos(day_open)).where(t >= '1000')", 0.0, 10),
    "anc_prev1430": ("以昨日 14:30 价为锚的回吐（全天）", "    t = df['date'].dt.strftime('%H%M')\n    p = px.where(t == '1430').groupby([inst, df['day']]).transform('max')\n    d2 = df[['instrument', 'day']].assign(p=p).drop_duplicates(['instrument', 'day'])\n    d2['pp'] = d2.groupby('instrument')['p'].shift(1)\n    m = df[['instrument', 'day']].merge(d2[['instrument', 'day', 'pp']], on=['instrument', 'day'], how='left')\n    x = -np.log(pos(px) / pos(pd.Series(m['pp'].to_numpy(), index=df.index)))", 0.0, 10),
    "anc_vwap1000": ("以 10:00 之后的 VWAP 为锚的回吐（10:30–14:30）", "    t = df['date'].dt.strftime('%H%M')\n    a = df['amt'].where(t > '1000', 0.0).groupby([inst, df['day']]).cumsum()\n    v = df['vol'].where(t > '1000', 0.0).groupby([inst, df['day']]).cumsum()\n    x = -np.log(pos(px) / pos(a / v.where(v > 0))).where(t > '1000')", 0.0, 10),
    "anc_split": ("上午以 10:00、午后以 11:30 为锚的回吐（10:30–14:30；10:00 用昨收）",
                  "    t = df['date'].dt.strftime('%H%M')\n    p10 = px.where(t == '1000').groupby([inst, df['day']]).transform('max')\n    p11 = px.where(t == '1130').groupby([inst, df['day']]).transform('max')\n    x = pd.Series(np.where(t > '1130', -np.log(pos(px) / pos(p11)), np.where(t > '1000', -np.log(pos(px) / pos(p10)), np.nan)), index=df.index)", 0.0, 10),
    "anc_split_vn": ("同上，除以五天段收益波动（波动标准化）",
                     "    t = df['date'].dt.strftime('%H%M')\n    p10 = px.where(t == '1000').groupby([inst, df['day']]).transform('max')\n    p11 = px.where(t == '1130').groupby([inst, df['day']]).transform('max')\n    sd = r_seg.groupby(inst).transform(lambda q: q.rolling(40, min_periods=10).std())\n    x = pd.Series(np.where(t > '1130', -np.log(pos(px) / pos(p11)), np.where(t > '1000', -np.log(pos(px) / pos(p10)), np.nan)), index=df.index) / sd.where(sd > 0)", 0.0, 10),
    "anc_split_sm": ("上午以 10:00、午后以 11:30 为锚（平滑 0.3）",
                     "    t = df['date'].dt.strftime('%H%M')\n    p10 = px.where(t == '1000').groupby([inst, df['day']]).transform('max')\n    p11 = px.where(t == '1130').groupby([inst, df['day']]).transform('max')\n    x = pd.Series(np.where(t > '1130', -np.log(pos(px) / pos(p11)), np.where(t > '1000', -np.log(pos(px) / pos(p10)), np.nan)), index=df.index)", 0.3, 10),
}

# s13（10-21）：午休跳空、上午 VWAP 锚、开盘区间的尺度、区间突破幅度、方向转折次数
T_ = "    t = df['date'].dt.strftime('%H%M')\n"
B["s13"] = {
    "lunch_gap": ("午休跳空的回吐：−ln(13:01 开盘 / 11:30 价)，13:30 起同值", T_ + "    o13 = df['s_open'].where(t == '1330').groupby([inst, df['day']]).transform('max')\n    p11 = px.where(t == '1130').groupby([inst, df['day']]).transform('max')\n    x = -np.log(pos(o13) / pos(p11)).where(t > '1130')", 0.0, 10),
    "pm_vs_am_vwap": ("午后相对上午 VWAP 的偏离取负（13:30 起）", T_ + "    a = df['amt'].where(t <= '1130', 0.0).groupby([inst, df['day']]).transform('sum')\n    v = df['vol'].where(t <= '1130', 0.0).groupby([inst, df['day']]).transform('sum')\n    x = -np.log(pos(px) / pos(a / v.where(v > 0))).where(t > '1130')", 0.0, 10),
    "orb60_pos": ("相对开盘 1 小时（10:00、10:30 两段）区间的位置取负，11:00 起", T_ + "    hi = df['s_high'].where(t <= '1030').groupby([inst, df['day']]).transform('max')\n    lo = df['s_low'].where(t <= '1030').groupby([inst, df['day']]).transform('min')\n    x = -((px - (hi + lo) / 2) / (hi - lo).where(hi > lo)).clip(-5, 5).where(t > '1030')", 0.0, 10),
    "orb_vn": ("开盘区间位置，以五天段区间标准化（而不是开盘区间宽度）", ORB + "    rg = (df['s_high'] - df['s_low']).groupby(inst).transform(lambda q: q.rolling(40, min_periods=10).mean())\n    x = -((px - (oh + ol) / 2) / rg.where(rg > 0)).clip(-10, 10).where(t > '1000')", 0.0, 10),
    "orb_break": ("开盘区间突破幅度取负：只取突破上沿／下沿的部分", ORB + "    up = (px - oh).clip(lower=0)\n    dn = (ol - px).clip(lower=0)\n    x = -((up - dn) / (oh - ol).where(oh > ol)).clip(-5, 5).where(t > '1000')", 0.0, 10),
    "orb_sm": ("开盘区间位置（平滑 0.3）", ORB + "    x = -((px - (oh + ol) / 2) / (oh - ol).where(oh > ol)).clip(-5, 5).where(t > '1000')", 0.3, 10),
    "turns_2d": ("两天里段收益方向转折的次数取负（来回震荡的股票）", "    sg = np.sign(r_seg)\n    tr = (sg * sg.groupby(inst).shift(1) < 0).astype(float)\n    x = -roll(tr, 16)", 0.0, 10),
    "turns_x_rev": ("震荡型股票的反转：−r_段 × 排名(两天转折次数)", "    sg = np.sign(r_seg)\n    tr = (sg * sg.groupby(inst).shift(1) < 0).astype(float)\n    w = roll(tr, 16).groupby(df['date']).rank(pct=True)\n    x = -r_seg * w", 0.6, 10),
    "day_rev_vn": ("当天至今收益（相对昨收）÷ 五天日内波动 取负", "    sd = r_seg.groupby(inst).transform(lambda q: q.rolling(40, min_periods=10).std())\n    x = -np.log(pos(px) / pos(df['pre_close'])) / sd.where(sd > 0)", 0.0, 10),
    "mid_anchor_day": ("相对当天高低点中点的位置取负（全天）：(现价 − (高 + 低)/2) / (高 − 低)", "    x = -((px - (day_high + day_low) / 2) / (day_high - day_low).where(day_high > day_low))", 0.3, 10),
}

# s14（10-21）：时间锚点细化（c103：对 1610 有残差 IC 0.013–0.021、换手不高于 1610）与快信号加平滑
def ANCM(col, cond):
    return (T_ + "    pa = df['" + col + "'].groupby([inst, df['day']]).transform('max')\n"
            "    x = -np.log(pos(px) / pos(pa)).where(" + cond + ")")
SDV = "    sd = r_seg.groupby(inst).transform(lambda q: q.rolling(40, min_periods=10).std())\n"
B["s14"] = {
    "anc0935": ("以 09:35 价为锚的回吐（10:00 起）", ANCM("p0935", "t >= '1000'"), 0.0, 10, "tm"),
    "anc0945": ("以 09:45 价为锚的回吐（10:00 起）", ANCM("p0945", "t >= '1000'"), 0.0, 10, "tm"),
    "anc1015": ("以 10:15 价为锚的回吐（10:30 起）", ANCM("p1015", "t >= '1030'"), 0.0, 10, "tm"),
    "anc1045": ("以 10:45 价为锚的回吐（11:00 起）", ANCM("p1045", "t >= '1100'"), 0.0, 10, "tm"),
    "anc_open_vwap": ("以开盘 30 分钟 VWAP 为锚的回吐（10:30 起）", T_ + "    a = df['amt'].where(t == '1000').groupby([inst, df['day']]).transform('max')\n    v = df['vol'].where(t == '1000').groupby([inst, df['day']]).transform('max')\n    x = -np.log(pos(px) / pos(a / v.where(v > 0))).where(t > '1000')", 0.0, 10),
    "anc_prev_vwap": ("以昨日 VWAP 为锚的回吐（全天）", DAILY + "    vw = dd['da'] / dd['dv'].where(dd['dv'] > 0)\n    x = -np.log(pos(px) / pos(dmap(vw.groupby(dd['instrument']).shift(1))))", 0.0, 10),
    "anc1000_vn": ("全天以 10:00 价为锚，÷ 五天段收益波动", SDV + ANCH("1000", "t > '1000'") + " / sd.where(sd > 0)", 0.0, 10),
    "anc1000_sm3": ("全天以 10:00 价为锚（平滑 0.3）", ANCH("1000", "t > '1000'"), 0.3, 10),
    "anc1030_vn": ("以 10:30 价为锚，÷ 五天段收益波动", SDV + ANCH("1030", "t > '1030'") + " / sd.where(sd > 0)", 0.0, 10),
    "lsd_sm6": ("尾段冲刺取负（平滑 0.6）：−(本段收益 − 当天收益/8)", "    dr = np.log(pos(px) / pos(df['pre_close']))\n    x = -(r_seg - dr / 8)", 0.6, 10),
    "lsd_vn": ("尾段冲刺 ÷ 五天段收益波动（平滑 0.3）", SDV + "    dr = np.log(pos(px) / pos(df['pre_close']))\n    x = -(r_seg - dr / 8) / sd.where(sd > 0)", 0.3, 10),
    "seg_clv_sm6": ("段内收盘位置取负（平滑 0.6）", CLV + "    x = -clv", 0.6, 10),
}

# s15（10-21）：相对 1617 还剩的只有超短线信息（尾段冲刺、加速度、段内收盘位置；换手 0.6–0.9）——做慢一点的版本
B["s15"] = {
    "accel_1h": ("一小时加速度取负：−(最近两段收益 − 再往前两段收益)", "    r2 = roll(r_seg, 2)\n    x = -(r2 - r2.groupby(inst).shift(2))", 0.3, 10),
    "accel_1h_sm6": ("一小时加速度取负（平滑 0.6）", "    r2 = roll(r_seg, 2)\n    x = -(r2 - r2.groupby(inst).shift(2))", 0.6, 10),
    "accel_day": ("日级加速度取负：−(今天至今收益 − 昨天同一时刻至收盘…近似为昨天全天收益)", DAILY + "    r1 = np.log(pos(dd['dc']) / pos(dd['pc']))\n    py = dmap(r1.groupby(dd['instrument']).shift(1))\n    x = -(np.log(pos(px) / pos(df['pre_close'])) - py)", 0.0, 10),
    "accel_sm8": ("价格加速度取负（平滑 0.8）", "    x = -(r_seg - r_seg.groupby(inst).shift(1))", 0.8, 10),
    "lsd_sm8": ("尾段冲刺取负（平滑 0.8）", "    dr = np.log(pos(px) / pos(df['pre_close']))\n    x = -(r_seg - dr / 8)", 0.8, 10),
    "lsd_2seg": ("两段尾段冲刺取负：−(最近两段收益 − 当天收益/4)", "    dr = np.log(pos(px) / pos(df['pre_close']))\n    x = -(roll(r_seg, 2) - dr / 4)", 0.5, 10),
    "seg_clv_sm8": ("段内收盘位置取负（平滑 0.8）", CLV + "    x = -clv", 0.8, 10),
    "seg_clv_2seg": ("两段合并的收盘位置取负：(2·收 − 两段高 − 两段低)/(两段高 − 两段低)", "    h2 = pd.concat([df['s_high'], df['s_high'].groupby(inst).shift(1)], axis=1).max(axis=1)\n    l2 = pd.concat([df['s_low'], df['s_low'].groupby(inst).shift(1)], axis=1).min(axis=1)\n    x = -((2 * px - h2 - l2) / (h2 - l2).where(h2 > l2)).clip(-1, 1)", 0.5, 10),
    "accel_x_lsd": ("加速度与尾段冲刺的平均（排名后）", "    a = -(r_seg - r_seg.groupby(inst).shift(1))\n    dr = np.log(pos(px) / pos(df['pre_close']))\n    b = -(r_seg - dr / 8)\n    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = rk(a).fillna(0.5) + rk(b).fillna(0.5)", 0.7, 10),
}

# s16（10-21）：截面前最后 5～10 分钟（1617 只剩超短线信息有增量）
B["s16"] = {
    "l5_ret": ("最后 5 分钟收益取负", "    x = -np.log(pos(px) / pos(df['p_l5']))", 0.0, 10, "late"),
    "l10_ret": ("最后 10 分钟收益取负", "    x = -np.log(pos(px) / pos(df['p_l10']))", 0.0, 10, "late"),
    "l5_ret_sm5": ("最后 5 分钟收益取负（平滑 0.5）", "    x = -np.log(pos(px) / pos(df['p_l5']))", 0.5, 10, "late"),
    "pre_l5_ret": ("段收益扣掉最后 5 分钟取负：−ln(最后 5 分钟起点价 / 上一截面价)", "    x = -np.log(pos(df['p_l5']) / pos(prev))", 0.3, 10, "late"),
    "l5_vshare": ("最后 5 分钟成交占本段比例取负（截面前抢量）", "    x = -df['v_l5'] / pos(df['vol'])", 0.3, 10, "late"),
    "l5_ret_x_v": ("最后 5 分钟放量的回吐：−最后 5 分钟收益 × 占比", "    x = -np.log(pos(px) / pos(df['p_l5'])) * (df['v_l5'] / pos(df['vol']))", 0.3, 10, "late"),
    "l5_dimb": ("最后 5 分钟三档量失衡", "    x = df['dimb_l5']", 0.3, 10, "late"),
    "l5_dimb_chg": ("最后 5 分钟量失衡相对段内平均的变化", "    x = df['dimb_l5'] - df['dimb_all']", 0.3, 10, "late"),
    "l5_spr": ("最后 5 分钟价差相对段内平均取负（截面前流动性变差）", "    x = -np.log(pos(df['spr_l5']) / pos(df['spr_all']))", 0.3, 10, "late"),
    "l10_clv": ("收盘在最后 10 分钟区间中的位置取负", "    x = -((2 * px - df['hi_l10'] - df['lo_l10']) / (df['hi_l10'] - df['lo_l10']).where(df['hi_l10'] > df['lo_l10'])).clip(-1, 1)", 0.3, 10, "late"),
}

# s17（10-21，用户："冲刺确实不错，再看看有没有更好的优化方式"）：冲刺的定义与用法
NS = ("    nseg = pd.Series(df['date'].dt.strftime('%H%M').map({'1000': 1, '1030': 2, '1100': 3, '1130': 4, '1330': 5, '1400': 6, '1430': 7, '1500': 8}).astype(float).to_numpy(), index=df.index)\n"
      "    dr = np.log(pos(px) / pos(df['pre_close']))\n")
B["s17"] = {
    "lsd_pace": ("冲刺（节奏按已过段数）：−(本段收益 − 当天至今收益 / 已过段数)", NS + "    x = -(r_seg - dr / nseg)", 0.0, 10),
    "lsd_pace_sm6": ("冲刺（按已过段数，平滑 0.6）", NS + "    x = -(r_seg - dr / nseg)", 0.6, 10),
    "lsd_pace_sm4": ("冲刺（按已过段数，平滑 0.4）", NS + "    x = -(r_seg - dr / nseg)", 0.4, 10),
    "lsd_vwap": ("冲刺（VWAP 版）：−(本段 VWAP 收益 − 当天 VWAP 收益 / 已过段数)，平滑 0.6", NS + "    svw = df['amt'] / df['vol'].where(df['vol'] > 0)\n    cv, ca = gd['vol'].cumsum(), gd['amt'].cumsum()\n    dvw = ca / cv.where(cv > 0)\n    x = -(np.log(pos(svw) / pos(prev)) - np.log(pos(dvw) / pos(df['pre_close'])) / nseg)", 0.6, 10),
    "lsd_pace_vn": ("冲刺（按已过段数）÷ 五天段收益波动，平滑 0.6", NS + SDV + "    x = -(r_seg - dr / nseg) / sd.where(sd > 0)", 0.6, 10),
    "lsd_2h": ("相对两小时节奏的冲刺：−(本段收益 − 最近 4 段收益 / 4)，平滑 0.6", "    x = -(r_seg - roll(r_seg, 4) / 4)", 0.6, 10),
    "lsd_x_vol": ("放量冲刺：冲刺 × (1 + 本段成交额排名)，平滑 0.6", NS + "    w = np.log(pos(df['amt']) / pos(roll('amt', 40) / 40)).groupby(df['date']).rank(pct=True)\n    x = -(r_seg - dr / nseg) * (1 + w)", 0.6, 10),
    "lsd_tail": ("冲刺强调极端：z + 0.3·z·|z|（逐截面 z 分数），平滑 0.6", NS + "    v = -(r_seg - dr / nseg)\n    g_ = v.groupby(df['date'])\n    zz = (v - g_.transform('mean')) / g_.transform('std')\n    x = zz + 0.3 * zz * zz.abs()", 0.6, 10),
    "lsd_ex_gap": ("冲刺（当天收益扣掉开盘跳空）：−(本段收益 − 开盘以来收益 / 已过段数)，平滑 0.6", NS + "    x = -(r_seg - np.log(pos(px) / pos(day_open)) / nseg)", 0.6, 10),
}

# s18（10-21 晚）：画像显示慢信号在高离散的日子更强（高／低离散半 IC_IR 比 1.3–1.5），与 1617（0.81）互补——扩展这一族
B["s18"] = {
    "cmf_2d": ("两天资金流取负", "    k = 16\n" + CLV + "    x = -roll(df['vol'] * clv, k) / pos(roll('vol', k))", 0.0, 10),
    "cmf_half": ("半天资金流取负（4 段）", "    k = 4\n" + CLV + "    x = -roll(df['vol'] * clv, k) / pos(roll('vol', k))", 0.3, 10),
    "bvc_2h": ("最近两小时签名成交量占比取负", "    k = 4\n    x = -roll('ssv', k) / pos(roll('vol', k))", 0.3, 10),
    "upvol_2h": ("最近两小时上涨分钟量占比取负", "    k = 4\n    x = -(roll('vup', k) - roll('vdn', k)) / pos(roll('vup', k) + roll('vdn', k))", 0.3, 10),
    "segskew_5d": ("五天段收益偏度取负", "    m = r_seg.groupby(inst).transform(lambda q: q.rolling(40, min_periods=16).mean())\n    sd = r_seg.groupby(inst).transform(lambda q: q.rolling(40, min_periods=16).std())\n    m3 = ((r_seg - m) ** 3).groupby(inst).transform(lambda q: q.rolling(40, min_periods=16).mean())\n    x = -m3 / sd.where(sd > 0) ** 3", 0.0, 10),
    "rmax_2h": ("最近两小时最大分钟涨幅取负", "    x = -df['rmax'].groupby(inst).transform(lambda q: q.rolling(4, min_periods=1).max())", 0.3, 10),
    "uvol_half": ("半天上行半方差取负", "    k = 4\n    x = -np.sqrt(pos(roll('sr2p', k)) / pos(roll('nmin', k)))", 0.3, 10),
    "lotto_2d": ("两天彩票合成：跳跃变差、偏度、上行半方差（各自排名）等权取负",
                 "    k = 16\n    n = roll('nmin', k)\n    m2, m3 = roll('sr2', k) / n, roll('sr3', k) / n\n    a1 = -(roll('sr2p', k) - roll('sr2n', k)) / pos(roll('sr2', k))\n    a2 = -(m3 / m2.where(m2 > 0) ** 1.5)\n    a3 = -np.sqrt(pos(roll('sr2p', k)) / pos(n))\n    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = rk(a1).fillna(0.5) + rk(a2).fillna(0.5) + rk(a3).fillna(0.5)", 0.0, 10),
    "abnamt_day": ("今天截至此刻成交额相对过去 20 日同一时刻均值（对数）取负", SLOTK + "    ca = np.log(pos(gd['amt'].cumsum()))\n    base = ca.groupby([inst, slot]).transform(lambda q: q.shift(1).rolling(20, min_periods=8).mean())\n    x = -(ca - base)", 0.0, 45),
    "flow_lotto": ("资金流 + 彩票（一天，各自排名）等权", D8 + CLV + "    a1 = -roll(df['vol'] * clv, k) / pos(roll('vol', k))\n    a2 = -(roll('sr2p', k) - roll('sr2n', k)) / pos(roll('sr2', k))\n    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = rk(a1).fillna(0.5) + rk(a2).fillna(0.5)", 0.0, 10),
}

# s19（10-21 晚）：冲刺 = 最近 30 分钟的反转 + 一点当天的顺势（s17："按已过段数"的修正反而把反转抵消了）。调顺势的比例 c：x = −r_段 + c·当天收益
DRT = "    dr = np.log(pos(px) / pos(df['pre_close']))\n"
B["s19"] = {
    "spr_c0": ("对照：纯本段反转 −r_段", "    x = -r_seg", 0.0, 10),
    "spr_c125": ("冲刺原始版 −r_段 + 0.125·当天收益（= s10_last_seg_vs_day）", DRT + "    x = -r_seg + 0.125 * dr", 0.0, 10),
    "spr_c25": ("−r_段 + 0.25·当天收益", DRT + "    x = -r_seg + 0.25 * dr", 0.0, 10),
    "spr_c50": ("−r_段 + 0.5·当天收益", DRT + "    x = -r_seg + 0.5 * dr", 0.0, 10),
    "spr_c75": ("−r_段 + 0.75·当天收益", DRT + "    x = -r_seg + 0.75 * dr", 0.0, 10),
    "spr_c100": ("−r_段 + 1.0·当天收益（本段之前的当天收益）", DRT + "    x = -r_seg + 1.0 * dr", 0.0, 10),
    "spr_c25_sm5": ("−r_段 + 0.25·当天收益（平滑 0.5）", DRT + "    x = -r_seg + 0.25 * dr", 0.5, 10),
    "spr_c50_sm5": ("−r_段 + 0.5·当天收益（平滑 0.5）", DRT + "    x = -r_seg + 0.5 * dr", 0.5, 10),
    "spr_2h_c25": ("−r_段 + 0.25·最近两小时收益", "    x = -r_seg + 0.25 * roll(r_seg, 4)", 0.0, 10),
    "spr_prevday": ("−r_段 + 0.25·当天收益 − 0.1·昨日收益", DRT + DAILY + "    r1 = np.log(pos(dd['dc']) / pos(dd['pc']))\n    py = dmap(r1.groupby(dd['instrument']).shift(1))\n    x = -r_seg + 0.25 * dr - 0.1 * py", 0.0, 10),
}

# s20（10-21 晚）：最后 10 分钟委托结构的写法
def _q2(expr, doc, a=0.3):
    return (doc, "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = " + expr, a, 10, "q2")
B["s20"] = {
    "ql_ctrl": _q2("rk(df['qn23L']).fillna(0.5) + rk(-df['qs1L']).fillna(0.5)", "对照：最后 10 分钟（= s3_q_mix_late）"),
    "ql_5": _q2("rk(df['qn23L5']).fillna(0.5) + rk(-df['qs1L5']).fillna(0.5)", "最后 5 分钟"),
    "ql_15": _q2("rk(df['qn23L15']).fillna(0.5) + rk(-df['qs1L15']).fillna(0.5)", "最后 15 分钟"),
    "ql_end": _q2("rk(df['qn23E']).fillna(0.5) + rk(-df['qs1E']).fillna(0.5)", "段末快照"),
    "ql_n23only": _q2("df['qn23L']", "只用深档笔数失衡（最后 10 分钟）"),
    "ql_s1only": _q2("-df['qs1L']", "只用一档委托规模比取负（最后 10 分钟）"),
    "ql_a0": _q2("rk(df['qn23L']).fillna(0.5) + rk(-df['qs1L']).fillna(0.5)", "最后 10 分钟，不平滑", 0.0),
    "ql_a5": _q2("rk(df['qn23L']).fillna(0.5) + rk(-df['qs1L']).fillna(0.5)", "最后 10 分钟，平滑 0.5", 0.5),
    "ql_plus_n1": _q2("rk(df['qn23L']).fillna(0.5) + rk(-df['qs1L']).fillna(0.5) + rk(df['qn1L']).fillna(0.5)", "加一档笔数失衡（最后 10 分钟）"),
    "ql_5_15": _q2("rk(df['qn23L5']).fillna(0.5) + rk(-df['qs1L5']).fillna(0.5) + rk(df['qn23L15']).fillna(0.5) + rk(-df['qs1L15']).fillna(0.5)", "5 分钟与 15 分钟窗口合成"),
    "ql_cnt_vs_vol": _q2("rk(df['qn23L']).fillna(0.5) - rk(df['qv23L']).fillna(0.5) + rk(-df['qs1L']).fillna(0.5)", "笔数失衡超出量失衡的部分 + 规模比"),
}

# s21（10-21 晚）：挂单规模一族
def _sz(expr, doc, a=0.3, lb=10):
    return (doc, "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = " + expr, a, lb, "sz")
B["s21"] = {
    "sz_all": _sz("-df['qs1']", "一档委托规模比取负（整段均值）"),
    "sz_l5": _sz("-df['qs1L5']", "一档委托规模比取负（最后 5 分钟）"),
    "sz_l15": _sz("-df['qs1L15']", "一档委托规模比取负（最后 15 分钟）"),
    "sz_end": _sz("-df['qs1E']", "一档委托规模比取负（段末快照）"),
    "sz_first": _sz("-df['qs1F']", "一档委托规模比取负（段首 10 分钟）"),
    "sz_chg": _sz("-(df['qs1L'] - df['qs1F'])", "一档委托规模比的段内变化取负（最后 10 分钟 − 段首 10 分钟）"),
    "sz_bid_abn": _sz("-(df['lbs1L'] - roll(df['lbs1A'].fillna(0), 40) / 40)", "买一平均单笔挂单量（最后 10 分钟，对数）相对五天均值取负：突然出现大买单", 0.3, 10),
    "sz_ask_abn": _sz("(df['las1L'] - roll(df['las1A'].fillna(0), 40) / 40)", "卖一平均单笔挂单量相对五天均值：突然出现大卖单 → 随后上涨？", 0.3, 10),
    "sz_bid_only": _sz("-df['lbs1L']", "买一平均单笔挂单量（最后 10 分钟，对数）取负"),
    "sz_ask_only": _sz("df['las1L']", "卖一平均单笔挂单量（最后 10 分钟，对数）"),
    "sz_deep": _sz("-df['qs23L']", "深档（2、3 档）委托规模比取负（最后 10 分钟）"),
    "sz_l1_deep": _sz("rk(-df['qs1L']).fillna(0.5) + rk(-df['qs23L']).fillna(0.5)", "一档 + 深档委托规模比（最后 10 分钟，各自排名）"),
    "sz_l5_l15": _sz("rk(-df['qs1L5']).fillna(0.5) + rk(-df['qs1L15']).fillna(0.5)", "一档委托规模比（5 分钟 + 15 分钟窗口）"),
    "sz_l10_a5": _sz("-df['qs1L']", "一档委托规模比取负（最后 10 分钟，平滑 0.5）", 0.5),
}

# s22（10-21 晚）：截面前 10 分钟的盘口动态
def _bd(expr, doc, a=0.3):
    return (doc, "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = " + expr, a, 10, "bd")
B["s22"] = {
    "bd_arr": _bd("((df['bna_e'] - df['bna_s']) - (df['ana_e'] - df['ana_s'])) / pos(df['bna_e'] + df['ana_e'])", "截面前 10 分钟委托到达净差：(买方新增笔数 − 卖方新增笔数) / 总笔数"),
    "bd_arr_neg": _bd("-((df['bna_e'] - df['bna_s']) - (df['ana_e'] - df['ana_s'])) / pos(df['bna_e'] + df['ana_e'])", "同上取负"),
    "bd_mdev": _bd("df['mdevL']", "截面前 10 分钟微观价格相对中间价的偏离（买盘厚 → 正）"),
    "bd_mdev_neg": _bd("-df['mdevL']", "同上取负"),
    "bd_thick": _bd("df['qthL']", "截面前 10 分钟队伍厚度不对称：买方深档/一档 − 卖方深档/一档（对数）"),
    "bd_s1_x_ret": _bd("-df['qs1L'] * (1 + rk(r_seg.abs()).fillna(0.5))", "一档规模比取负 × (1 + 段收益幅度排名)：大波动之后的挂单结构"),
    "bd_s1_plus_mdev": _bd("rk(-df['qs1L']).fillna(0.5) + rk(-df['mdevL']).fillna(0.5)", "一档规模比 + 微观价格偏离（各自排名）"),
}

# s23（10-21 晚）：一档委托规模比——当下的异常 vs 股票长期特征
MQ = ("    qs = df['qs1'].astype(float)\n"
      "    cnt = roll(qs.notna().astype(float), {k})\n"
      "    m = roll(qs.fillna(0.0), {k}) / cnt.where(cnt > 0)\n"
      "    m2 = roll(qs.fillna(0.0) ** 2, {k}) / cnt.where(cnt > 0)\n"
      "    sd = np.sqrt((m2 - m ** 2).clip(lower=1e-12))\n"
      "    m, sd = m.groupby(inst).shift(1), sd.groupby(inst).shift(1)\n")
B["s23"] = {
    "szn_abn5": ("最后 10 分钟一档规模比相对过去五天均值的偏离取负（当下的异常）", MQ.format(k=40) + "    x = -(df['qs1L'] - m)", 0.3, 10, "sz"),
    "szn_abn20": ("最后 10 分钟一档规模比相对过去 20 天均值的偏离取负", MQ.format(k=160) + "    x = -(df['qs1L'] - m)", 0.3, 45, "sz"),
    "szn_z5": ("最后 10 分钟一档规模比的五天 z 分数取负", MQ.format(k=40) + "    x = -(df['qs1L'] - m) / sd.where(sd > 0)", 0.3, 10, "sz"),
    "szn_level5": ("一档规模比的五天均值取负（股票长期特征）", MQ.format(k=40) + "    x = -m", 0.0, 10, "sz"),
    "szn_level20": ("一档规模比的 20 天均值取负（股票长期特征）", MQ.format(k=160) + "    x = -m", 0.0, 45, "sz"),
    "szn_abn5_a5": ("同 szn_abn5，平滑 0.5", MQ.format(k=40) + "    x = -(df['qs1L'] - m)", 0.5, 10, "sz"),
}

# s24（10-21 夜）：截面那一刻的成交与盘口细节
def _x(expr, doc, a=0.3):
    return (doc, "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = " + expr, a, 10, "x")
TICK = "    ticks = (df['spr_e'] / 0.01).round()\n"
B["s24"] = {
    "x_spr_ticks": _x("-(df['spr_e'] / 0.01).round()", "段末价差（跳价数）取负：价差窄 = 排队约束强"),
    "x_s1_tight": _x("-df['qs1L'] * (1 + ((df['spr_e'] / 0.01).round() <= 1).astype(float))", "一档规模比取负，价差只有 1 个跳价时加倍（排队约束强时规模信息更有效？）"),
    "x_s1_wide": _x("-df['qs1L'] * (1 + ((df['spr_e'] / 0.01).round() >= 2).astype(float))", "一档规模比取负，价差 ≥ 2 个跳价时加倍"),
    "x_rlast": _x("-df['r_last']", "最后一分钟收益取负"),
    "x_vlast": _x("-df['v_last'] / pos(df['vol'])", "最后一分钟成交量占本段比例取负"),
    "x_tpos_e": _x("-df['tpos_e']", "段末成交位置取负（最后一笔靠近卖一 = 主动买）"),
    "x_z10": _x("df['z10']", "最后 10 分钟零成交的分钟数（截面前流动性枯竭）"),
    "x_imb_persist": ("一天内深档笔数失衡的持续性：8 段失衡均值 ÷ 8 段失衡标准差（持续一边倒）", "    k = 8\n    q = df['qn23'].astype(float)\n    m = roll(q.fillna(0.0), k) / k\n    m2 = roll(q.fillna(0.0) ** 2, k) / k\n    x = m / np.sqrt((m2 - m ** 2).clip(lower=1e-6))", 0.3, 10, "x"),
    "x_s1_x_tpos": _x("rk(-df['qs1L']).fillna(0.5) + rk(-df['tpos_e']).fillna(0.5)", "一档规模比 + 段末成交位置（各自排名）"),
}

# s25（10-21 夜）：截面前 10 分钟的成交规模
def _ts(expr, doc, a=0.3):
    return (doc, "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    ats = np.log(pos(df['a_l10']) / pos(df['nd_l10'])) - np.log(pos(df['amt']) / pos(df['ndeal']))\n    x = " + expr, a, 10, "ts")
B["s25"] = {
    "ts_big_l10": _ts("-ats", "最后 10 分钟平均每笔成交额相对整段（对数）取负：截面前大单涌入"),
    "ts_big_dir": _ts("-ats.clip(lower=0) * np.sign(df['r_l10'])", "大单涌入 × 最后 10 分钟收益方向，取负（大单推动的价格 → 回吐）"),
    "ts_big_sv": _ts("-ats.clip(lower=0) * df['sv_l10'] / pos(df['vol'])", "大单涌入 × 最后 10 分钟签名成交量占比，取负"),
    "ts_r_l10": _ts("-df['r_l10']", "最后 10 分钟收益取负（对照）"),
    "ts_sv_l10": _ts("-df['sv_l10'] / pos(df['vol'])", "最后 10 分钟签名成交量占本段取负"),
    "ts_small_dir": _ts("(-ats).clip(lower=0) * np.sign(df['r_l10'])", "小单主导 × 最后 10 分钟收益方向（散户推动 → 延续？）"),
}

# s26（10-21 夜，无人值守）：截面前 10 分钟签名成交量（s25 最好的一项）与委托规模比、段内前 20 分钟收益的组合
SVR = "(df['sv_l10'] / pos(df['vol']))"
B["s26"] = {
    "tq_sv_plus_s1": _ts("rk(-" + SVR + ").fillna(0.5) + rk(-df['qs1L']).fillna(0.5)", "最后 10 分钟签名成交量占比 + 一档委托规模比（各自排名，都取负）"),
    "tq_s1_x_sv": _ts("-df['qs1L'] * (1 + rk(-" + SVR + ").fillna(0.5))", "一档委托规模比取负 × (1 + 签名成交量取负的排名)：卖压之后的买方大单"),
    "tq_r_early": _ts("-(r_seg - df['r_l10'])", "段内前 20 分钟收益取负（与最后 10 分钟收益对照：反转来自哪一段）"),
    "tq_sv_a5": _ts("-" + SVR, "最后 10 分钟签名成交量占比取负，平滑 0.5", 0.5),
    "tq_sv_minus_r": _ts("rk(-" + SVR + ").fillna(0.5) - rk(-df['r_l10']).fillna(0.5)", "签名成交量与最后 10 分钟收益的背离（排名差）：量价不一致"),
    "tq_sv_big": _ts("-" + SVR + " * (1 + ats.clip(lower=0))", "签名成交量占比取负 × (1 + 大单涌入)"),
}

# s27（10-22 晨）：涨跌停的约束（从未试过）——离涨停／跌停多近、是否封板、是否炸板、限幅内的位置
LIM = ("    code = df['instrument'].astype(str).str[:3]\n"
       "    big = code.isin(['688', '689']) | (code.isin(['300', '301']) & (df['day'] >= pd.Timestamp('2020-08-24')))\n"
       "    lim = pd.Series(np.where(big, 0.2, 0.1), index=df.index)\n"
       "    pcl = pos(df['pre_close'])\n"
       "    up_px, dn_px = (pcl * (1 + lim) + 1e-9).round(2), (pcl * (1 - lim) + 1e-9).round(2)\n"
       "    dret = px / pcl - 1\n"
       "    d_up, d_dn = (up_px - px) / pcl, (px - dn_px) / pcl\n"
       "    at_up, at_dn = d_up <= 0.001, d_dn <= 0.001\n"
       "    hit_up = (df['s_high'] >= up_px - 0.001).astype(float).groupby([inst, df['day']]).cummax() > 0\n"
       "    hit_dn = (df['s_low'] <= dn_px + 0.001).astype(float).groupby([inst, df['day']]).cummax() > 0\n")
B["s27"] = {
    "lim_band": ("当日涨跌幅占限幅的比例（主板 10%，创业板 2020-08-24 起与科创板 20%）：靠近涨停 → 延续？", LIM + "    x = dret / lim", 0.3, 10),
    "lim_near": ("离涨停 2% 以内的程度 − 离跌停 2% 以内的程度（涨停的磁吸）", LIM + "    x = (0.02 - d_up).clip(lower=0) - (0.02 - d_dn).clip(lower=0)", 0.3, 10),
    "lim_at": ("截面时刻封涨停 − 封跌停", LIM + "    x = at_up.astype(float) - at_dn.astype(float)", 0.3, 10),
    "lim_broke": ("炸板：今天碰过涨停、现在已打开 → 回落（取负）；碰过跌停、现在已打开 → 反弹", LIM + "    x = -(hit_up & ~at_up).astype(float) + (hit_dn & ~at_dn).astype(float)", 0.3, 10),
    "lim_rev_far": ("段收益反转，只在离两个限价都超过 2% 时生效（限价附近的价格不是自由的）", LIM + "    x = (-r_seg).where(np.minimum(d_up, d_dn) > 0.02, 0.0)", 0.3, 10),
    "lim_rev_damp": ("段收益反转 × (1 − |当日涨跌幅| / 限幅)：越靠近限价，反转越弱", LIM + "    x = -r_seg * (1 - (dret.abs() / lim).clip(upper=1))", 0.3, 10),
    "lim_room": ("上方剩余空间 − 下方剩余空间（占限幅）：上涨空间被限价压缩的股票", LIM + "    x = (d_up - d_dn) / lim", 0.3, 10),
    "lim_hit_any": ("今天碰过涨停（不管是否打开）− 碰过跌停", LIM + "    x = hit_up.astype(float) - hit_dn.astype(float)", 0.3, 10),
}

# s28（10-22 晨）：整数关口（从未试过）：价格停在关口上方 = 支撑、下方 = 阻力；穿越关口；昨收、今开两条线的穿越
RN = ("    lg = np.floor(np.log10(px.where(px > 0)))\n"
      "    step = 10.0 ** lg / 10.0\n"                               # 细关口：<10 元每 0.1，10–100 元每 1，100 元以上每 10
      "    stepc = 10.0 ** lg / 2.0\n"                               # 粗关口：<10 元每 0.5，10–100 元每 5，100 元以上每 50
      "    fr = px / step - np.floor(px / step + 1e-9)\n"
      "    frc = px / stepc - np.floor(px / stepc + 1e-9)\n"
      "    tol = 0.003 * px\n")
B["s28"] = {
    "rn_int_sr": ("整数元关口（价格 ≥ 3 元）：刚在整数上方 0.4% 内 = 支撑（+），刚在下方 = 阻力（−）",
                  "    fl, ce = np.floor(px + 1e-9), np.ceil(px - 1e-9)\n    ok = px >= 3\n    x = (((px - fl) / px < 0.004) & ok).astype(float) - (((ce - px) / px < 0.004) & ok).astype(float)", 0.3, 10),
    "rn_fine_sr": ("细关口（按价格量级：0.1／1／10 元）的支撑 − 阻力，容差 0.3%", RN + "    x = ((fr * step < tol)).astype(float) - (((1 - fr) * step < tol)).astype(float)", 0.3, 10),
    "rn_coarse_sr": ("粗关口（0.5／5／50 元）的支撑 − 阻力，容差 0.3%", RN + "    x = ((frc * stepc < tol)).astype(float) - (((1 - frc) * stepc < tol)).astype(float)", 0.3, 10),
    "rn_pos": ("两个细关口之间的位置：0.5 − 小数部分（刚过关口 → +0.5，快到关口 → −0.5）", RN + "    x = 0.5 - fr", 0.3, 10),
    "rn_test": ("冲关口：本段上涨且停在阻力下方 → −1；本段下跌且停在支撑上方 → +1（关口拒绝）",
                RN + "    x = ((r_seg < 0) & (fr * step < tol)).astype(float) - ((r_seg > 0) & ((1 - fr) * step < tol)).astype(float)", 0.3, 10),
    "rn_cross": ("本段穿越细关口的方向（突破延续？）", RN + "    x = np.sign(np.floor(px / step + 1e-9) - np.floor(prev / step + 1e-9))", 0.3, 10),
    "pc_cross": ("本段穿越昨收的方向（红绿线）", "    pcl = pos(df['pre_close'])\n    x = (np.sign(px - pcl) - np.sign(prev - pcl)) / 2", 0.3, 10),
    "op_cross": ("本段穿越今天开盘价的方向", "    x = (np.sign(px - day_open) - np.sign(prev - day_open)) / 2", 0.3, 10),
}

# s29（10-22）：s27 的 lim_rev_far（IC 0.075）对照——同样平滑 0.3 的普通段反转，以及限价过滤的阈值
B["s29"] = {
    "rv_a3": ("对照：段收益反转 −r_seg，平滑 0.3（与 lim_rev_far 同平滑）", "    x = -r_seg", 0.3, 10),
    "lim_far1": ("段收益反转，只在离两个限价都超过 1% 时生效", LIM + "    x = (-r_seg).where(np.minimum(d_up, d_dn) > 0.01, 0.0)", 0.3, 10),
    "lim_far4": ("段收益反转，只在离两个限价都超过 4% 时生效", LIM + "    x = (-r_seg).where(np.minimum(d_up, d_dn) > 0.04, 0.0)", 0.3, 10),
    "lim_far_up": ("段收益反转，只剔除离涨停 2% 以内的（跌停一侧不管）", LIM + "    x = (-r_seg).where(d_up > 0.02, 0.0)", 0.3, 10),
    "lim_far_dn": ("段收益反转，只剔除离跌停 2% 以内的", LIM + "    x = (-r_seg).where(d_dn > 0.02, 0.0)", 0.3, 10),
    "lim_near_mom": ("离限价 2% 以内的股票：段收益方向延续；其余为 0", LIM + "    x = r_seg.where(np.minimum(d_up, d_dn) <= 0.02, 0.0)", 0.3, 10),
}

# s30（10-22）：重缓冲（β 1.5～2.0）下腿基本在 10:00 定下、持有到收盘，慢一点的委托结构可能更值钱：当天至今的均值
TD = ("    dk = [inst, df['day']]\n"
      "    def today(col):\n"
      "        v = df[col].astype(float)\n"
      "        n = v.notna().astype(float).groupby(dk).cumsum()\n"
      "        return v.fillna(0.0).groupby(dk).cumsum() / n.where(n > 0)\n")
B["s30"] = {
    "td_s1": ("一档委托规模比取负，当天至今各段均值（不含 09:30 段）", TD + "    q = df['qs1'].where(df['date'].dt.hour * 100 + df['date'].dt.minute != 930)\n    df['_q'] = q\n    x = -today('_q')", 0.3, 10, "sz"),
    "td_s1L": ("一档委托规模比取负，当天至今各段'最后 10 分钟'均值的平均", TD + "    df['_q'] = df['qs1L'].where(df['date'].dt.hour * 100 + df['date'].dt.minute != 930)\n    x = -today('_q')", 0.3, 10, "sz"),
    "td_mix": ("一档规模比：0.5·本段最后 10 分钟 + 0.5·当天至今（各自排名）", TD + "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    df['_q'] = df['qs1L'].where(df['date'].dt.hour * 100 + df['date'].dt.minute != 930)\n    x = rk(-df['qs1L']).fillna(0.5) + rk(-today('_q')).fillna(0.5)", 0.3, 10, "sz"),
    "td_n23": ("深档（2、3 档）委托笔数失衡，当天至今各段均值", TD + "    df['_q'] = df['qn23'].where(df['date'].dt.hour * 100 + df['date'].dt.minute != 930)\n    x = today('_q')", 0.3, 10, "sz"),
    "td_n23_neg": ("同上取负", TD + "    df['_q'] = df['qn23'].where(df['date'].dt.hour * 100 + df['date'].dt.minute != 930)\n    x = -today('_q')", 0.3, 10, "sz"),
    "td_deep": ("深档委托规模比取负，当天至今各段均值", TD + "    df['_q'] = df['qs23'].where(df['date'].dt.hour * 100 + df['date'].dt.minute != 930)\n    x = -today('_q')", 0.3, 10, "sz"),
}

# s31（10-22）：s30 发现"当天至今的一档规模比"在重缓冲下补 IC_IR、压力（换手 0.19）；同一思路推到其它委托结构，以及"今天相对昨天"
TDQ = TD + "    nz = df['date'].dt.hour * 100 + df['date'].dt.minute != 930\n"
YD = ("    dmean = df.assign(_v=df['qs1'].where(nz)).groupby([inst, df['day']])['_v'].mean()\n"
      "    prevm = dmean.groupby(level=0).shift(1)\n"
      "    yq = pd.Series(prevm.reindex(pd.MultiIndex.from_arrays([inst, df['day']])).to_numpy(), index=df.index)\n")
B["s31"] = {
    "td_qn1": ("一档委托笔数失衡，当天至今各段均值", TDQ + "    df['_q'] = df['qn1'].where(nz)\n    x = today('_q')", 0.3, 10, "sz"),
    "td_qv1": ("一档挂单量失衡，当天至今各段均值", TDQ + "    df['_q'] = df['qv1'].where(nz)\n    x = today('_q')", 0.3, 10, "sz"),
    "td_qth": ("队伍厚度不对称（买方深档/一档 − 卖方），当天至今", TDQ + "    df['_q'] = df['qth'].where(nz)\n    x = today('_q')", 0.3, 10, "sz"),
    "td_s1_vs_yd": ("一档规模比当天至今 − 昨天全天均值，取负（今天的异常）", TDQ + YD + "    df['_q'] = df['qs1'].where(nz)\n    x = -(today('_q') - yq)", 0.3, 15, "sz"),
    "td_s1_x_n23": ("当天至今：一档规模比（取负）+ 深档笔数失衡（各自排名）", TDQ + "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    df['_q'] = df['qs1'].where(nz)\n    df['_n'] = df['qn23'].where(nz)\n    x = rk(-today('_q')).fillna(0.5) + rk(today('_n')).fillna(0.5)", 0.3, 10, "sz"),
    "td_s1_sm0": ("一档规模比取负，当天至今，不做跨截面平滑", TDQ + "    df['_q'] = df['qs1'].where(nz)\n    x = -today('_q')", 0.0, 10, "sz"),
    "yd_s1": ("昨天全天的一档规模比取负（对照：信息是否只在当天）", TDQ + YD + "    x = -yq", 0.0, 15, "sz"),
}

# s32（10-22，新方向）：隐藏流动性与挂单回补（截面前 10 分钟）。逐分钟，一档价格不变时：
#   隐藏吸收（买一）= max(主动卖量 − 买一可见挂单的减少, 0)：卖单打到买一，但买一的可见挂单没少那么多 → 有隐藏买单（冰山）
#   回补（买一）= 买一挂单的增加；卖一对称。主动卖量 ≈ 量·(1 − 成交位置)，成交位置 = (收 − 买一)/(卖一 − 买一)。量与挂单都以股计。
_LAGW = "OVER (PARTITION BY instrument_id, trading_day ORDER BY hms)"
SQL_HL = SQLS["late"].replace("""    cte_seg AS (""", """    cte_m2 AS (
        SELECT *, LAG(b1) """ + _LAGW + """ AS b1p, LAG(a1) """ + _LAGW + """ AS a1p,
               LAG(bv1) """ + _LAGW + """ AS bv1p, LAG(av1) """ + _LAGW + """ AS av1p FROM cte_m
    ),
    cte_seg AS (""", 1)
_HB = "GREATEST(v * (1 - COALESCE(tpos, 0.5)) - GREATEST(bv1p - bv1, 0), 0)"
_HA = "GREATEST(v * COALESCE(tpos, 0.5) - GREATEST(av1p - av1, 0), 0)"
_SB, _SA = "b1 > 0 AND b1 = b1p", "a1 > 0 AND a1 = a1p"
_k = SQL_HL.rindex("        FROM cte_m\n")
SQL_HL = SQL_HL[:_k] + """            , SUM(CASE WHEN """ + _L10 + """ AND """ + _SB + """ THEN """ + _HB + """ ELSE 0 END) AS hb_l,
            SUM(CASE WHEN """ + _L10 + """ AND """ + _SA + """ THEN """ + _HA + """ ELSE 0 END) AS ha_l,
            SUM(CASE WHEN """ + _SB + """ THEN """ + _HB + """ ELSE 0 END) AS hb_a,
            SUM(CASE WHEN """ + _SA + """ THEN """ + _HA + """ ELSE 0 END) AS ha_a,
            SUM(CASE WHEN """ + _L10 + """ AND """ + _SB + """ THEN GREATEST(bv1 - bv1p, 0) ELSE 0 END) AS rb_l,
            SUM(CASE WHEN """ + _L10 + """ AND """ + _SA + """ THEN GREATEST(av1 - av1p, 0) ELSE 0 END) AS ra_l,
            SUM(CASE WHEN """ + _L10 + """ THEN v * (1 - COALESCE(tpos, 0.5)) ELSE 0 END) AS vs_l,
            SUM(CASE WHEN """ + _L10 + """ THEN v * COALESCE(tpos, 0.5) ELSE 0 END) AS vb_l
        FROM cte_m2\n""" + SQL_HL[_k + len("        FROM cte_m\n"):]
assert "cte_m2" in SQL_HL and "hb_l" in SQL_HL and SQL_HL.count("FROM cte_m2") == 1
SQLS["hl"] = SQL_HL
def _hl(expr, doc, a=0.3):
    return (doc, "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = " + expr, a, 10, "hl")
B["s32"] = {
    "hl_hid": _hl("(df['hb_l'] - df['ha_l']) / pos(df['vol'])", "截面前 10 分钟隐藏吸收：买一 − 卖一，占本段成交量（隐藏买单托底 → 随后上涨？）"),
    "hl_hid_neg": _hl("-(df['hb_l'] - df['ha_l']) / pos(df['vol'])", "同上取负"),
    "hl_hid_rel": _hl("(df['hb_l'] - df['ha_l']) / pos(df['hb_l'] + df['ha_l'])", "截面前 10 分钟隐藏吸收的买卖失衡（占隐藏吸收合计）"),
    "hl_hidA": _hl("(df['hb_a'] - df['ha_a']) / pos(df['vol'])", "整段隐藏吸收：买一 − 卖一，占本段成交量"),
    "hl_res": _hl("df['rb_l'] / pos(df['vs_l']) - df['ra_l'] / pos(df['vb_l'])", "截面前 10 分钟挂单回补：买一回补 ÷ 主动卖量 − 卖一回补 ÷ 主动买量（买方越打越补 → 支撑）"),
    "hl_ice": _hl("(df['hb_l'] + df['ha_l']) / pos(df['vol'])", "截面前 10 分钟隐藏吸收合计占成交量（有冰山单的股票）"),
    "hl_hid_x_s1": _hl("rk((df['hb_l'] - df['ha_l']) / pos(df['vol'])).fillna(0.5) + rk(-df['qs1L']).fillna(0.5)", "隐藏吸收失衡 + 一档委托规模比取负（各自排名）"),
}

# s33（10-22，新方向）：截面前 10 分钟的成交到达加速（笔数、成交额集中在最后 10 分钟的程度），及其与收益、签名成交量的交互
BR = "    burst = df['nd_l10'] / pos(df['ndeal']) - 1.0 / 3.0\n    aburst = df['a_l10'] / pos(df['amt']) - 1.0 / 3.0\n"
B["s33"] = {
    "br_nd": _ts(BR.strip() + "\n    x = -burst", "截面前 10 分钟成交笔数占本段比例 − 1/3，取负（最后突然热闹 → 随后回落？）"),
    "br_amt": _ts(BR.strip() + "\n    x = -aburst", "截面前 10 分钟成交额占本段比例 − 1/3，取负"),
    "br_r": _ts(BR.strip() + "\n    x = -df['r_l10'] * (1 + rk(burst).fillna(0.5))", "最后 10 分钟收益取负 × (1 + 笔数加速排名)：热闹中的涨跌更容易回吐"),
    "br_r_calm": _ts(BR.strip() + "\n    x = -df['r_l10'] * (2 - rk(burst).fillna(0.5))", "最后 10 分钟收益取负 × (2 − 笔数加速排名)：冷清中的涨跌更容易回吐（对照）"),
    "br_sv": _ts(BR.strip() + "\n    x = -df['sv_l10'] / pos(df['vol']) * (1 + rk(burst).fillna(0.5))", "签名成交量占比取负 × (1 + 笔数加速排名)"),
    "br_seg_r": _ts(BR.strip() + "\n    x = -r_seg * (1 + rk(burst).fillna(0.5))", "段收益取负 × (1 + 笔数加速排名)"),
}

# s34（10-22）：s32 的"隐藏吸收"加到 1617 上同换手四项都升，但单独 IC 很弱且方向与"冰山托底"相反——拆开：主动成交失衡本身 vs 超出它的隐藏部分
B["s34"] = {
    "of_l10": _hl("(df['vb_l'] - df['vs_l']) / pos(df['vol'])", "截面前 10 分钟主动买量 − 主动卖量（量·成交位置），占本段成交量（订单流延续？）"),
    "of_l10_rel": _hl("(df['vb_l'] - df['vs_l']) / pos(df['vb_l'] + df['vs_l'])", "截面前 10 分钟主动买卖失衡（占最后 10 分钟成交）"),
    "of_hidden": _hl("((df['ha_l'] - df['hb_l']) - (df['vb_l'] - df['vs_l'])) / pos(df['vol'])", "隐藏部分：（卖一隐藏吸收 − 买一隐藏吸收）减去主动成交失衡 = 可见挂单的消耗失衡（取反）"),
    "of_vis": _hl("((df['vb_l'] - df['ha_l']) - (df['vs_l'] - df['hb_l'])) / pos(df['vol'])", "可见挂单被吃掉的失衡：主动买吃掉的卖一可见量 − 主动卖吃掉的买一可见量"),
    "of_l10_sm5": _hl("(df['vb_l'] - df['vs_l']) / pos(df['vol'])", "主动成交失衡（同 of_l10），平滑 0.5", 0.5),
}

# s35（10-22）：订单流（of_l10_rel）的无 LAG 写法（平台上宽 LAG 有过 OOM）：主动买卖量只用本分钟的成交位置，不需要上一分钟的挂单
_k2 = SQLS["late"].rindex("        FROM cte_m\n")
SQL_OF = SQLS["late"][:_k2] + """            , SUM(CASE WHEN """ + _L10 + """ THEN v * (1 - COALESCE(tpos, 0.5)) ELSE 0 END) AS vs_l,
            SUM(CASE WHEN """ + _L10 + """ THEN v * COALESCE(tpos, 0.5) ELSE 0 END) AS vb_l
""" + SQLS["late"][_k2:]
assert "LAG(" not in SQL_OF and "vb_l" in SQL_OF
SQLS["of"] = SQL_OF
B["s35"] = {
    "of_l10_rel": ("截面前 10 分钟主动买卖失衡（占最后 10 分钟成交）；无 LAG 写法，值应与 s34_of_l10_rel 相同",
                   "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = (df['vb_l'] - df['vs_l']) / pos(df['vb_l'] + df['vs_l'])", 0.3, 10, "of"),
}

# s36（10-22）：订单流（s35 有用：在反转给定后，截面前的主动成交方向延续）的变体：窗口（5／15 分钟、整段）、按一档深度标准化、加速（最后 10 分钟 − 整段）
_k3 = SQLS["late"].rindex("        FROM cte_m\n")
_VS = lambda cond: "SUM(CASE WHEN " + cond + " THEN v * (1 - COALESCE(tpos, 0.5)) ELSE 0 END)"   # noqa: E731
_VB = lambda cond: "SUM(CASE WHEN " + cond + " THEN v * COALESCE(tpos, 0.5) ELSE 0 END)"         # noqa: E731
SQL_OF2 = SQLS["late"][:_k3] + """            , """ + _VS(_L10) + """ AS vs_l, """ + _VB(_L10) + """ AS vb_l,
            """ + _VS(_L(5)) + """ AS vs_5, """ + _VB(_L(5)) + """ AS vb_5,
            """ + _VS(_L(15)) + """ AS vs_15, """ + _VB(_L(15)) + """ AS vb_15,
            """ + _VS("1 = 1") + """ AS vs_a, """ + _VB("1 = 1") + """ AS vb_a,
            AVG(CASE WHEN """ + _L10 + """ AND bv1 + av1 > 0 THEN bv1 + av1 END) AS dep_l
""" + SQLS["late"][_k3:]
assert "LAG(" not in SQL_OF2 and "vb_15" in SQL_OF2
SQLS["of2"] = SQL_OF2
def _of(expr, doc, a=0.3):
    return (doc, "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    rel = lambda b, s: (df[b] - df[s]) / pos(df[b] + df[s])\n    x = " + expr, a, 10, "of2")
B["s36"] = {
    "of_5": _of("rel('vb_5', 'vs_5')", "截面前 5 分钟主动买卖失衡（占这 5 分钟成交）"),
    "of_15": _of("rel('vb_15', 'vs_15')", "截面前 15 分钟主动买卖失衡"),
    "of_a": _of("rel('vb_a', 'vs_a')", "整段（30 分钟）主动买卖失衡"),
    "of_dep": _of("(df['vb_l'] - df['vs_l']) / pos(df['dep_l'])", "截面前 10 分钟主动买卖失衡 ÷ 一档平均深度（相对盘口的冲击）"),
    "of_acc": _of("rel('vb_l', 'vs_l') - rel('vb_a', 'vs_a')", "订单流加速：最后 10 分钟失衡 − 整段失衡"),
    "of_mix": _of("rk(rel('vb_5', 'vs_5')).fillna(0.5) + rk(rel('vb_l', 'vs_l')).fillna(0.5) + rk(rel('vb_15', 'vs_15')).fillna(0.5)", "5／10／15 分钟失衡（各自排名相加）"),
    "of_l10_sm0": _of("rel('vb_l', 'vs_l')", "截面前 10 分钟失衡，不平滑", 0.0),
}

# s37（10-22）：14:30 截面的标签（之后 30 分钟的 VWAP）含收盘集合竞价。看个股收盘竞价的惯性：过去几天"竞价价 ÷ 14:56 收盘"的方向能否预测今天的竞价。
# 只在 14:30 截面取值，其余截面为 0。15:00 那一段只用来算历史（不是评测截面），当天 14:30 只用前几天的竞价。
SQL_AUC = SQL.replace("""            MAX(pc) AS pre_close
        FROM cte_m""", """            MAX(pc) AS pre_close,
            MAX(CASE WHEN hms = '145600' THEN c END) AS c1456, MAX(CASE WHEN hms = '150000' THEN c END) AS c1500,
            SUM(CASE WHEN hms = '150000' THEN v ELSE 0 END) AS v1500
        FROM cte_m""")
assert "c1456" in SQL_AUC
SQLS["auc"] = SQL_AUC
AUC = ("    is1500 = (df['date'].dt.hour * 100 + df['date'].dt.minute) == 1500\n"
       "    dd = df.loc[is1500, ['instrument', 'day', 'c1456', 'c1500', 'v1500']].copy()\n"
       "    dd['ra'] = np.log(pos(dd['c1500']) / pos(dd['c1456']))\n"
       "    dv = df.groupby(['instrument', 'day'])['vol'].sum().rename('dvol').reset_index()\n"
       "    dd = dd.merge(dv, on=['instrument', 'day'], how='left')\n"
       "    dd['sh'] = dd['v1500'] / pos(dd['dvol'])\n"
       "    dd = dd.sort_values(['instrument', 'day'])\n"
       "    gi = dd.groupby('instrument')\n"
       "    for k in (1, 5, 20):\n"
       "        dd['ra%d' % k] = gi['ra'].transform(lambda s: s.shift(1).rolling(k, min_periods=max(1, k // 2)).mean())\n"
       "        dd['sh%d' % k] = gi['sh'].transform(lambda s: s.shift(1).rolling(k, min_periods=max(1, k // 2)).mean())\n"
       "    df = df.merge(dd[['instrument', 'day', 'ra1', 'ra5', 'ra20', 'sh5']], on=['instrument', 'day'], how='left')\n"
       "    at1430 = (df['date'].dt.hour * 100 + df['date'].dt.minute) == 1430\n")
B["s37"] = {
    "auc_m5": ("14:30 截面：过去 5 天收盘竞价收益（竞价价 ÷ 14:56 收盘）的均值（竞价惯性）", AUC + "    x = df['ra5'].where(at1430)", 0.0, 40, "auc"),
    "auc_m20": ("14:30 截面：过去 20 天收盘竞价收益的均值", AUC + "    x = df['ra20'].where(at1430)", 0.0, 40, "auc"),
    "auc_m1": ("14:30 截面：昨天的收盘竞价收益", AUC + "    x = df['ra1'].where(at1430)", 0.0, 40, "auc"),
    "auc_m5_neg": ("同 auc_m5 取负（竞价反转）", AUC + "    x = -df['ra5'].where(at1430)", 0.0, 40, "auc"),
    "auc_m5_sh": ("14:30 截面：过去 5 天竞价收益 × 过去 5 天竞价成交占比（竞价份额大的股票竞价影响 VWAP 更多）", AUC + "    x = (df['ra5'] * df['sh5']).where(at1430)", 0.0, 40, "auc"),
}

# s38（10-23）：仿 s37（收盘竞价惯性），11:30 截面的标签是午后开盘的 30 分钟。看个股"11:30 → 午后开盘"的跳动是否有惯性：
# 过去几天的 ln(13:01 开盘 ÷ 11:30 价)、ln(13:05 收盘 ÷ 11:30 价) 的均值，只在 11:30 截面取值。
SQL_LUN = SQL.replace("""            MAX(pc) AS pre_close
        FROM cte_m""", """            MAX(pc) AS pre_close,
            MAX(CASE WHEN hms = '130100' THEN o END) AS o1301, MAX(CASE WHEN hms = '130500' THEN c END) AS c1305
        FROM cte_m""")
assert "o1301" in SQL_LUN
SQLS["lun"] = SQL_LUN
LUN = ("    hm_ = df['date'].dt.hour * 100 + df['date'].dt.minute\n"
       "    a = df.loc[hm_ == 1130, ['instrument', 'day', 'last_px']].rename(columns={'last_px': 'p1130'})\n"
       "    b = df.loc[hm_ == 1330, ['instrument', 'day', 'o1301', 'c1305']]\n"
       "    dd = a.merge(b, on=['instrument', 'day'], how='inner').sort_values(['instrument', 'day'])\n"
       "    dd['g1'] = np.log(pos(dd['o1301']) / pos(dd['p1130']))\n"
       "    dd['g5'] = np.log(pos(dd['c1305']) / pos(dd['p1130']))\n"
       "    gi = dd.groupby('instrument')\n"
       "    for c_ in ('g1', 'g5'):\n"
       "        for k in (5, 20):\n"
       "            dd['%s_%d' % (c_, k)] = gi[c_].transform(lambda s: s.shift(1).rolling(k, min_periods=max(1, k // 2)).mean())\n"
       "    df = df.merge(dd[['instrument', 'day', 'g1_5', 'g1_20', 'g5_5', 'g5_20']], on=['instrument', 'day'], how='left')\n"
       "    at1130 = (df['date'].dt.hour * 100 + df['date'].dt.minute) == 1130\n")
B["s38"] = {
    "lun_g1_m5": ("11:30 截面：过去 5 天午后开盘跳动（13:01 开盘 ÷ 11:30 价）的均值（惯性）", LUN + "    x = df['g1_5'].where(at1130)", 0.0, 40, "lun"),
    "lun_g1_m20": ("11:30 截面：过去 20 天午后开盘跳动的均值", LUN + "    x = df['g1_20'].where(at1130)", 0.0, 40, "lun"),
    "lun_g5_m5": ("11:30 截面：过去 5 天午后前 5 分钟收益（13:05 收盘 ÷ 11:30 价）的均值", LUN + "    x = df['g5_5'].where(at1130)", 0.0, 40, "lun"),
    "lun_g5_m20": ("11:30 截面：过去 20 天午后前 5 分钟收益的均值", LUN + "    x = df['g5_20'].where(at1130)", 0.0, 40, "lun"),
}

# s39（10-23）：收盘竞价惯性（s37 有用）的变体：方向的一致性、按竞价成交占比加权、竞价波动 × 截面前订单流、竞价份额本身
AUC2 = AUC.replace("    df = df.merge(dd[['instrument', 'day', 'ra1', 'ra5', 'ra20', 'sh5']], on=['instrument', 'day'], how='left')\n",
    "    dd['pos10'] = gi['ra'].transform(lambda s: (s.shift(1) > 0).astype(float).rolling(10, min_periods=5).mean() - (s.shift(1) < 0).astype(float).rolling(10, min_periods=5).mean())\n"
    "    dd['vw5'] = gi['ra'].transform(lambda s: s.shift(1)) * dd['sh'].groupby(dd['instrument']).shift(1)\n"
    "    dd['vw5'] = dd.groupby('instrument')['vw5'].transform(lambda s: s.rolling(5, min_periods=3).sum())\n"
    "    dd['abs20'] = gi['ra'].transform(lambda s: s.shift(1).abs().rolling(20, min_periods=10).mean())\n"
    "    df = df.merge(dd[['instrument', 'day', 'ra1', 'ra5', 'ra20', 'sh5', 'pos10', 'vw5', 'abs20']], on=['instrument', 'day'], how='left')\n")
assert AUC2 != AUC
B["s39"] = {
    "auc_pos10": ("14:30 截面：过去 10 天收盘竞价上涨天数比例 − 下跌天数比例（方向的一致性）", AUC2 + "    x = df['pos10'].where(at1430)", 0.0, 40, "auc"),
    "auc_vw5": ("14:30 截面：过去 5 天收盘竞价收益 × 当天竞价成交占比之和（竞价份额大的日子权重大）", AUC2 + "    x = df['vw5'].where(at1430)", 0.0, 40, "auc"),
    "auc_sh5": ("14:30 截面：过去 5 天收盘竞价成交占全天比例（竞价份额本身）", AUC2 + "    x = df['sh5'].where(at1430)", 0.0, 40, "auc"),
    "auc_sh5_neg": ("同上取负", AUC2 + "    x = -df['sh5'].where(at1430)", 0.0, 40, "auc"),
    "auc_m5_t": ("14:30 截面：过去 5 天竞价收益均值 ÷ 过去 20 天竞价收益绝对值均值（惯性的 t 值近似）", AUC2 + "    x = (df['ra5'] / df['abs20'].where(df['abs20'] > 0)).where(at1430)", 0.0, 40, "auc"),
}

# s40（10-23）：分时段的信号。14:30 的标签含收盘竞价，截面前的盘口可能更有用；10:00 是每天第一个截面，隔夜信息最新。
HM = "    hm_ = df['date'].dt.hour * 100 + df['date'].dt.minute\n"
OG = (HM + "    g = df.loc[hm_ == 930, ['instrument', 'day', 's_open', 'pre_close']].copy()\n"
      "    g['gap'] = np.log(pos(g['s_open']) / pos(g['pre_close']))\n"
      "    g = g.sort_values(['instrument', 'day'])\n"
      "    g['gap5'] = g.groupby('instrument')['gap'].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())\n"
      "    df = df.merge(g[['instrument', 'day', 'gap', 'gap5']], on=['instrument', 'day'], how='left')\n"
      "    hm_ = df['date'].dt.hour * 100 + df['date'].dt.minute\n")
B["s40"] = {
    "q14_s1": ("只在 14:30：一档委托规模比取负（截面前 10 分钟）", HM + "    x = (-df['qs1L']).where(hm_ == 1430)", 0.0, 10, "sz"),
    "q14_end": ("只在 14:30：段末快照深档笔数失衡 + 一档规模比取负（各自排名）", HM + "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = (rk(df['qn23E']).fillna(0.5) + rk(-df['qs1E']).fillna(0.5)).where(hm_ == 1430)", 0.0, 10, "q2"),
    "of14": ("只在 14:30：截面前 10 分钟主动买卖失衡", HM + "    x = ((df['vb_l'] - df['vs_l']) / pos(df['vb_l'] + df['vs_l'])).where(hm_ == 1430)", 0.0, 10, "of"),
    "gap10_rev": ("只在 10:00：隔夜跳空（开盘竞价价 ÷ 昨收）取负", OG + "    x = (-df['gap']).where(hm_ == 1000)", 0.0, 15, "base"),
    "gap10_m5": ("只在 10:00：过去 5 天开盘跳空的均值（开盘竞价的惯性）", OG + "    x = df['gap5'].where(hm_ == 1000)", 0.0, 15, "base"),
    "gap10_x": ("只在 10:00：今天跳空 − 过去 5 天跳空均值，取负（异常跳空回吐）", OG + "    x = (-(df['gap'] - df['gap5'])).where(hm_ == 1000)", 0.0, 15, "base"),
}

# s41（10-23）：价格冲击（每单位签名成交量带来的收益）的不对称与分解。基础 SQL 已有 Σr·sv、Σsv²（分买、卖方向）。
#   λ+ = Σ(r·sv | sv>0) / Σ(sv² | sv>0)，λ− 同理；乘以本段成交量化成无量纲。段收益 = 订单流解释的部分 λ·Σsv + 剩余部分。
LAM = ("    lam_p = df['srsvp'] / df['ssv2p'].where(df['ssv2p'] > 0) * df['vol']\n"
       "    lam_n = df['srsvn'] / df['ssv2n'].where(df['ssv2n'] > 0) * df['vol']\n"
       "    lam = df['srsv'] / df['ssv2'].where(df['ssv2'] > 0)\n"
       "    r_flow = (lam * df['ssv']).clip(-0.2, 0.2)\n")
B["s41"] = {
    "lam_asym": ("价格冲击不对称：买方向 λ − 卖方向 λ（买单推价更容易 → ?）", LAM + "    x = lam_p - lam_n", 0.3, 10),
    "lam_asym_neg": ("同上取负", LAM + "    x = -(lam_p - lam_n)", 0.3, 10),
    "lam_tot": ("总价格冲击（Kyle λ × 本段成交量，非流动性的分钟版）", LAM + "    x = lam * df['vol']", 0.3, 10),
    "rev_noflow": ("段收益中订单流解释不了的部分取负（噪声部分的反转）", LAM + "    x = -(r_seg - r_flow)", 0.3, 10),
    "rev_flow": ("段收益中订单流解释的部分取负（信息部分的反转，对照）", LAM + "    x = -r_flow", 0.3, 10),
    "rev_split": ("噪声部分反转 + 0.5·信息部分延续（各自排名）", LAM + "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n    x = rk(-(r_seg - r_flow)).fillna(0.5) + 0.5 * rk(r_flow).fillna(0.5)", 0.3, 10),
}

# s42（10-24 凌晨）：14:30 上"今天异常放量 → 收盘前继续涨"（14:30 扫描的发现）的变体，只在 14:30 取值：
# 放量的方向（今天涨还是跌、最后 30 分钟涨还是跌）、放量集中在最后 10 分钟、今天的放量持续性
V14 = (ABN + "    ca = np.log(pos(gd['amt'].cumsum()))\n"
       "    cab = ca.groupby([inst, slot]).transform(lambda q: q.shift(1).rolling(20, min_periods=8).mean())\n"
       "    abd = ca - cab\n"
       "    dret = np.log(px / pos(day_open))\n"
       "    at14 = slot == '1430'\n"
       "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True)\n")
B["s42"] = {
    "v14_abn": ("只在 14:30：本段成交量相对同时段 20 日常态（对照，等价于 s5_c_abnvol 取反）", V14 + "    x = abn.where(at14)", 0.0, 45),
    "v14_abn_up": ("只在 14:30：放量 × 今天涨（今天跌的记 0）", V14 + "    x = (abn * (dret > 0)).where(at14)", 0.0, 45),
    "v14_abn_dn": ("只在 14:30：放量 × 今天跌（今天涨的记 0）", V14 + "    x = (abn * (dret < 0)).where(at14)", 0.0, 45),
    "v14_abn_x_ret": ("只在 14:30：放量 × 今天收益的方向（放量上涨 → 收盘前继续涨，放量下跌 → 继续跌？）", V14 + "    x = (abn * np.sign(dret)).where(at14)", 0.0, 45),
    "v14_abn_x_seg": ("只在 14:30：放量 × 最后 30 分钟收益的方向", V14 + "    x = (abn * np.sign(r_seg)).where(at14)", 0.0, 45),
    "v14_abd_rank": ("只在 14:30：本段放量与今天至今放量（各自排名相加）", V14 + "    x = (rk(abn).fillna(0.5) + rk(abd).fillna(0.5)).where(at14)", 0.0, 45),
    "v14_abn_vs_day": ("只在 14:30：本段放量 − 今天至今放量（尾盘才开始放量）", V14 + "    x = (abn - abd).where(at14)", 0.0, 45),
}

# s43（10-24）：订单流推动的反转阶梯。s41 发现段反转全部来自订单流解释的那部分价格（λ·Σ签名成交量，IC 0.060），剩余部分 IC ≈ 0；
# 只试过单段。这里做多尺度：最近 1／2／4／8 行（每天 9 行，含 09:30、15:00）与当天至今的订单流推动收益取负，以及多尺度排名相加的阶梯；
# 对照：同样尺度的普通段收益阶梯、剩余（非订单流）部分。
LAM2 = (LAM + "    rf = r_flow.fillna(0.0)\n    rs = r_seg.fillna(0.0)\n    rn = rs - rf\n"
        "    rk = lambda v: pd.Series(np.asarray(v, dtype='float64'), index=df.index).groupby(df['date']).rank(pct=True).fillna(0.5)\n"
        "    lad = lambda s: rk(-s) + rk(-roll(s, 2)) + rk(-roll(s, 4)) + rk(-roll(s, 8))\n")
B["s43"] = {
    "rf1": ("订单流推动的段收益取负（不平滑）", LAM2 + "    x = -rf", 0.0, 10),
    "rf2": ("最近 2 段订单流推动收益之和取负", LAM2 + "    x = -roll(rf, 2)", 0.0, 10),
    "rf4": ("最近 4 段订单流推动收益之和取负", LAM2 + "    x = -roll(rf, 4)", 0.0, 10),
    "rf8": ("最近 8 段（约一天）订单流推动收益之和取负", LAM2 + "    x = -roll(rf, 8)", 0.0, 10),
    "rfday": ("当天至今订单流推动收益之和取负", LAM2 + "    x = -rf.groupby([inst, df['day']]).cumsum()", 0.0, 10),
    "rflad": ("订单流推动收益的多尺度反转阶梯（1／2／4／8 段，各自排名相加）", LAM2 + "    x = lad(rf)", 0.0, 10),
    "rflad_sm": ("同上，跨截面平滑 0.3", LAM2 + "    x = lad(rf)", 0.3, 10),
    "rvlad": ("对照：普通段收益的多尺度反转阶梯（同尺度）", LAM2 + "    x = lad(rs)", 0.0, 10),
    "rnlad": ("对照：剩余（非订单流）部分的多尺度阶梯", LAM2 + "    x = lad(rn)", 0.0, 10),
    "rf_minus_rn": ("订单流部分阶梯 − 剩余部分阶梯（反转推开的、延续没推开的）", LAM2 + "    x = lad(rf) - lad(rn)", 0.0, 10),
}


def main():
    out = sys.argv[1]
    batch = sys.argv[2] if len(sys.argv) > 2 else "s1"
    os.makedirs(out, exist_ok=True)
    for name, spec in B[batch].items():
        doc, formula, smooth, lookback = spec[:4]
        sql = SQLS[spec[4] if len(spec) > 4 else "base"]
        src = TEMPLATE.format(doc=doc, batch=batch, smooth_doc=("，按股票跨截面指数平滑 α=%.1f（过去的权重）" % smooth) if smooth else "",
                              lookback=lookback, sql=sql, formula=formula, smooth=smooth)
        compile(src, name, "exec")
        with open(os.path.join(out, "%s_%s.py" % (batch, name)), "w", encoding="utf-8") as f:
            f.write(src)
    print(batch, len(B[batch]), "个文件 →", out)


if __name__ == "__main__":
    main()
