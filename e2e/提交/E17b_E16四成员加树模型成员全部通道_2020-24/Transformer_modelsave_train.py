# -*- coding: utf-8 -*-
"""Transformer 端到端模型 —— 训练侧脚本（共享定义 + 从零训练并持久化）。

基于官方 demo 改写，接口保持一致（推理 notebook 从本文件导入）：
    MODEL_PATH, BATCH, StockTransformer, build_dataset, pool, train_and_save, load_model
以及推理用的分片预测函数 predict_range。

相对 demo 的改动（只做类型转换、重采样、标准化、切窗，不构造跨字段的衍生特征）：
  1. 数值变换 NORM="rel"：量、额、笔数字段取 log1p；价格字段除以样本最后一个有效 1m close
     后取 log；随后按训练集统计量做全局标准化。
  2. 双尺度输入：最近 SEQ_LEN 根原始 1m bar + 最近 COARSE_LEN 根 30 分钟 bar（由 1m 原样
     重采样：开=首、高=最大、低=最小、量额笔数=求和、其余快照字段=末值），两组 token 拼接后
     进入同一个 Transformer 编码器。
  3. 训练目标 TARGET="cs_z"：vwap_return 在每个截面内做 z-score（截断 ±5）后用 MSE 拟合。
  4. N_MODELS 个随机种子的模型集成，推理取平均。
  5. 推理只使用当前截面及之前的原始行情，不使用模型对之前截面的输出。
  6. 按月分片读取与切窗，样本以 float16 存储；设备优先 cuda，其次 mps，最后 cpu；固定随机种子。

数据层：平台上默认使用 DaiBackend（datasources["bar1m"] + cpt_jyc_2026_instruments +
cpt_jyc_2026_vwap）。本地训练可传入实现了相同接口的 backend；其余逻辑完全共用本文件。

用法：
    python Transformer_modelsave_train.py           # 平台环境，训练并保存 transformer_model.json
"""
import copy
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dai
except ImportError:          # 本地训练环境没有 dai，使用自定义 backend
    dai = None

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)

    class _Logger(object):
        def info(self, msg, **kw):
            logging.info("%s %s", msg, kw)
    logger = _Logger()

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")

# ---------- 配置（写死，不随平台入参变化） ----------
TRAIN_START, TRAIN_END = "2020-01-01", "2024-12-31 23:59:59"   # 训练区间写死，切勿用测试区间训练
SEQ_LEN = 64  # 1m 回看根数
COARSE_LEN = 0  # 30m 回看根数（每天 9 根含集合竞价，约 5 个交易日）；0 = 关闭
SUMMARY = False  # True = 加入截面摘要特征（多尺度反转、波动、盘口等，逐截面 z-score）
SUMMARY_SET = "all"  # all（含慢变量股票特征）/ rev（只保留反转与盘口类，去掉波动、非流动性、价格、成交额、价差）
SUMMARY_POOL = False  # True = 摘要特征投影后加到池化向量上（序列保持 64 个 token，MPS 上快约 3 倍）
LABEL_NEUTRAL = 0.0  # >0：训练标签对四个风格特征（规模、特质波动、非流动性、低价）逐截面回归取残差的力度
OUT_NEUTRAL = 1.0  # >0：推理输出（截面排名）对同样四个特征逐截面中性化的力度（参照因子 104 号取 1.5）
EPOCHS, BATCH, LR, SEED = 6, 512, 0.001, 42
WEIGHT_DECAY = 0.0  # >0 时使用 AdamW
LR_SCHEDULE = "none"  # none / cos
N_MODELS = 4  # 种子集成个数（种子 SEED, SEED+1, ...）
MAX_TRAIN_INSTRUMENTS = 0  # 0 = 不限制
NORM = "rel2"  # raw / log / rel / rel2（量额相对本窗口均值，与计量单位无关）
E04_NORM = "rel2"  # e04 的 1 分钟输入变换：rel2 每个量类字段各自除以本窗口均值；rel3（10-24）买卖两侧同类字段共用一个参照，保留买卖之间的相对水平
NORM_SHARE = [["bid_volume1", "ask_volume1", "bid_volume2", "ask_volume2", "bid_volume3", "ask_volume3"],
              ["bid_num_orders1", "ask_num_orders1", "bid_num_orders2", "ask_num_orders2", "bid_num_orders3", "ask_num_orders3"]]
TARGET = "cs_rank"  # raw / cs_z / cs_rank（逐截面秩 → 正态分位数，与评测的秩 IC 同口径；R2）
LOSS = "mse"  # mse（样本随机打乱）/ ic（每批若干完整截面，最大化截面相关）/ mse+ic
SECS_PER_BATCH = 4  # LOSS 含 ic 时每批的截面数
FIELD_SET = "base"  # base（demo 10 字段）/ full（三档价量 + 委托笔数 + 成交笔数）
BUFFER_DAYS = 20               # 每个分片向前多读的自然日，保证 1m 与 30m 回看
STATS_SAMPLE = 200000          # 计算标准化统计量所用的训练样本数上限
PROGRESS_STEPS = 10            # 每个 epoch 打印进度的次数
CKPT_DIR = None                # 断点目录（本地训练用）；None = 不存断点（平台默认）

FIELDS = {
    "base": ["open", "high", "low", "close", "volume", "amount",
             "bid_price1", "ask_price1", "bid_volume1", "ask_volume1"],
    "ob1": ["open", "high", "low", "close", "volume", "amount",                   # 10-24：base + 一档委托笔数（原始字段）
            "bid_price1", "ask_price1", "bid_volume1", "ask_volume1", "bid_num_orders1", "ask_num_orders1"],
    "full": ["open", "high", "low", "close", "volume", "amount", "deal_number",
             "bid_price1", "bid_price2", "bid_price3", "ask_price1", "ask_price2", "ask_price3",
             "bid_volume1", "bid_volume2", "bid_volume3", "ask_volume1", "ask_volume2", "ask_volume3",
             "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
             "ask_num_orders1", "ask_num_orders2", "ask_num_orders3"],
}
FEATURE_COLS = FIELDS[FIELD_SET]
N_FEAT = len(FEATURE_COLS)
LABEL_TABLE = "cpt_jyc_2026_vwap"
SECTION_TABLE = "cpt_jyc_2026_instruments"
# 30 分钟窗口以结束时刻命名；集合竞价归入 09:30
BUCKETS = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"]

MODEL_CFG = dict(n_feat=N_FEAT, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN,
                 coarse_len=COARSE_LEN, n_summary=0, summary_pool=False)
DATA_ALIGNMENT = "1m grid bars <= section; 30m buckets ending <= section; label=vwap_return"


def set_fields(field_set):
    """切换字段集合（本地实验用）；同步更新依赖字段数的全局量。"""
    global FIELD_SET, FEATURE_COLS, N_FEAT
    FIELD_SET, FEATURE_COLS = field_set, FIELDS[field_set]
    N_FEAT = len(FEATURE_COLS)
    MODEL_CFG["n_feat"] = N_FEAT


def _is_price(c):
    return c in ("open", "high", "low", "close") or "_price" in c


# ---------- 模型 ----------
class StockTransformer(nn.Module):
    """逐 token 线性投影 + 可学习位置编码；1m token 与 30m token 拼接后共用一个 Encoder；均值池化 + 回归头。"""

    def __init__(self, n_feat, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN,
                 coarse_len=0, n_summary=0, summary_pool=False, dropout=0.1):
        super(StockTransformer, self).__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.coarse_len = coarse_len
        if coarse_len:
            self.proj_c = nn.Linear(n_feat, d_model)
            self.pos_c = nn.Parameter(torch.zeros(1, coarse_len, d_model))
        self.n_summary = n_summary
        self.summary_pool = summary_pool
        if n_summary:
            self.proj_s = nn.Linear(n_summary, d_model)
            if not summary_pool:
                self.pos_s = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x, xc=None, xs=None):                     # (B, L, F), (B, Lc, F), (B, K) -> (B,)
        h = self.proj(x) + self.pos
        if self.coarse_len:
            h = torch.cat([h, self.proj_c(xc) + self.pos_c], dim=1)
        if self.n_summary and not self.summary_pool:
            h = torch.cat([h, self.proj_s(xs).unsqueeze(1) + self.pos_s], dim=1)
        z = self.encoder(h).mean(dim=1)
        if self.n_summary and self.summary_pool:
            z = z + self.proj_s(xs)
        return self.head(z).squeeze(-1)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------- 数据层：平台 dai ----------
class DaiBackend(object):
    """平台数据接口。本地 backend 需实现同名方法，返回相同列与口径（价格单位元，缺失为 NaN）。"""

    def __init__(self, table):
        self.table = table

    def pool(self, sd, ed):
        df = dai.query(f"SELECT DISTINCT instrument FROM {SECTION_TABLE}", filters={"date": [sd, ed]}).df()
        return sorted(df["instrument"].dropna().unique().tolist())

    def sections(self, sd, ed, instruments):
        df = dai.query(f"SELECT date, instrument FROM {SECTION_TABLE}",
                       filters={"date": [sd, ed], "instrument": instruments}).df()
        df["date"] = pd.to_datetime(df["date"])
        return df.drop_duplicates(["date", "instrument"])

    def labels(self, sd, ed, instruments):
        df = dai.query(f"SELECT date, instrument, vwap_return AS y FROM {LABEL_TABLE}",
                       filters={"date": [sd, ed], "instrument": instruments}).df()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def bars(self, sd, ed, instruments, cols=None):
        sql = f"SELECT date, instrument, {', '.join(cols or FEATURE_COLS)} FROM {self.table}"   # 含 adjust_factor
        df = dai.query(sql, filters={"date": [sd, ed], "instrument": instruments}).df()
        df["date"] = pd.to_datetime(df["date"])
        return df


_WIN30_SQL = """
WITH cte_raw AS (
    SELECT
        instrument_id,
        strftime(date, '%Y-%m-%d') AS trading_day,
        strftime(date, '%H%M%S') AS hms,
        CAST(COALESCE(bid_price1, 0) AS DOUBLE) AS bp1,
        CAST(COALESCE(ask_price1, 0) AS DOUBLE) AS ap1,
        CAST(COALESCE(high, 0) AS DOUBLE) AS px_high,
        CAST(COALESCE(low, 0) AS DOUBLE) AS px_low,
        CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,
        CAST(COALESCE(pre_close, 0) AS DOUBLE) AS pre_close,
        CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
        CAST(COALESCE(amount, 0) AS DOUBLE) AS amt,
        CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,
        CAST(COALESCE(deal_number, 0) AS DOUBLE) AS deals
    FROM {table}
),
cte_snapshot AS (
    SELECT
        instrument_id, trading_day, hms, pre_close, px_open, px_high, px_low, px_close, deals, amt,
        CASE WHEN vol > 0 AND amt > 0 AND px_low > 0
                  AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02
             THEN amt ELSE 0 END AS amt_c,
        CASE WHEN vol > 0 AND amt > 0 AND px_low > 0
                  AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02
             THEN vol ELSE 0 END AS vol_c,
        CASE WHEN bp1 > 0 AND ap1 > bp1 THEN (bp1 + ap1) / 2.0
             WHEN px_close > 0 THEN px_close ELSE 0 END AS px,
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
        END AS time_segment
    FROM cte_raw
),
cte_window AS (
    SELECT trading_day, time_segment, instrument_id,
           SUM(amt_c) AS sum_amt, SUM(vol_c) AS sum_vol,
           MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment OR (time_segment = 93000 AND hms = '092500') THEN px ELSE 0 END) AS last_px,
           MAX(pre_close) AS pre_close,
           arg_min(CASE WHEN px_open > 0 THEN px_open ELSE NULL END, hms) AS w_open,
           MAX(CASE WHEN px_high > 0 THEN px_high ELSE NULL END) AS w_high,
           MIN(CASE WHEN px_low > 0 THEN px_low ELSE NULL END) AS w_low,
           arg_max(CASE WHEN px_close > 0 THEN px_close ELSE NULL END, hms) AS w_close,
           SUM(amt) AS raw_amt, SUM(deals) AS w_deals
    FROM cte_snapshot
    WHERE time_segment != -1
    GROUP BY instrument_id, trading_day, time_segment
)
SELECT
    CAST(CONCAT(f.trading_day, ' ',
                strftime(strptime(LPAD(f.time_segment, 6, '0'), '%H%M%S'), '%H:%M:%S')) AS DATETIME) AS date,
    all_instruments.instrument,
    f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.w_open, f.w_high, f.w_low, f.w_close, f.raw_amt, f.w_deals
FROM cte_window f
LEFT JOIN all_instruments USING (instrument_id)
"""


def _windows30_dai(table, sd, ed, instruments):
    """30 分钟窗口表（平台）：每个窗口的清洗后成交额 / 成交量、窗口末价格（中间价优先）、昨收。
    SQL 源自传统因子赛道 104 号。2026-10-11 修复：1 分钟数据在 09:25 集合竞价之后直接是 09:31，没有 09:30:00 的 bar，
    原写法（末价取 hms = 窗口终点）使 09:30 窗口末价缺失；现改为 09:30 窗口取 09:25 竞价 bar，与本地 30 分钟数据一致
    （scripts/sql_win30_check.py 用 DuckDB 0.9.2 在本地 1 分钟数据上核对）。"""
    df = dai.query(_WIN30_SQL.format(table=table), filters={"date": [sd, ed]}, compression=True).df()
    df = df.dropna(subset=["instrument"])
    df["instrument"] = df["instrument"].astype(str)
    df = df[df["instrument"].isin(set(instruments))].drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    for c in WIN30_COLS:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)
    return df[["date", "instrument"] + WIN30_COLS]


# 30 分钟窗口表的列：清洗后成交额 / 成交量、窗口末价格（中间价优先）、昨收；窗口开高低收（1 分钟 bar 的首开、最高、最低、末收）、原始成交额、成交笔数
WIN30_COLS = ["sum_amt", "sum_vol", "last_px", "pre_close", "w_open", "w_high", "w_low", "w_close", "raw_amt", "w_deals"]


DaiBackend.windows30 = lambda self, sd, ed, instruments: _windows30_dai(self.table, sd, ed, instruments)

# 30 分钟窗口内 1 分钟盘口的均值（WIN_GROUPS 含 book 时使用；窗口划分与 _WIN30_SQL 相同，只用 1–3 档）。
# 本地用同一段 SQL 在本地 1 分钟数据上预先算好（scripts/build_book30_cache.py），保证两边口径一致。
_BOOK30_SQL = """
WITH cte_raw AS (
    SELECT
        instrument_id,
        strftime(date, '%Y-%m-%d') AS trading_day,
        strftime(date, '%H%M%S') AS hms,
        CAST(COALESCE(bid_price1, 0) AS DOUBLE) AS bp1,
        CAST(COALESCE(ask_price1, 0) AS DOUBLE) AS ap1,
        CAST(COALESCE(bid_volume1, 0) + COALESCE(bid_volume2, 0) + COALESCE(bid_volume3, 0) AS DOUBLE) AS bv,
        CAST(COALESCE(ask_volume1, 0) + COALESCE(ask_volume2, 0) + COALESCE(ask_volume3, 0) AS DOUBLE) AS av,
        CAST(COALESCE(bid_volume1, 0) + COALESCE(ask_volume1, 0) AS DOUBLE) AS v1,
        CAST(COALESCE(bid_volume1, 0) AS DOUBLE) AS bv1,
        CAST(COALESCE(ask_volume1, 0) AS DOUBLE) AS av1,
        CAST(COALESCE(bid_num_orders1, 0) + COALESCE(bid_num_orders2, 0) + COALESCE(bid_num_orders3, 0) AS DOUBLE) AS bn,
        CAST(COALESCE(ask_num_orders1, 0) + COALESCE(ask_num_orders2, 0) + COALESCE(ask_num_orders3, 0) AS DOUBLE) AS an,
        CAST(COALESCE(volume, 0) AS DOUBLE) AS vol
    FROM {table}
),
cte_seg AS (
    SELECT *,
        CASE WHEN bv + av > 0 THEN (bv - av) / (bv + av) ELSE NULL END AS imb,
        CASE WHEN bp1 > 0 AND ap1 > bp1 THEN (ap1 - bp1) / ((ap1 + bp1) / 2.0) ELSE NULL END AS spr,
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
        END AS time_segment
    FROM cte_raw
),
cte_window AS (
    SELECT trading_day, time_segment, instrument_id,
           AVG(CASE WHEN bv + av > 0 THEN (bv - av) / (bv + av) ELSE NULL END) AS b_imb,
           AVG(CASE WHEN bp1 > 0 AND ap1 > bp1 THEN (ap1 - bp1) / ((ap1 + bp1) / 2.0) ELSE NULL END) AS b_spr,
           AVG(CASE WHEN bv + av > 0 THEN v1 / (bv + av) ELSE NULL END) AS b_conc,
           AVG(CASE WHEN bn + an > 0 THEN (bn - an) / (bn + an) ELSE NULL END) AS b_oimb,
           AVG(CASE WHEN bv1 + av1 > 0 THEN (bv1 - av1) / (bv1 + av1) ELSE NULL END) AS b_imb1,
           arg_max(imb, CASE WHEN imb IS NOT NULL THEN hms ELSE NULL END)
               - arg_min(imb, CASE WHEN imb IS NOT NULL THEN hms ELSE NULL END) AS b_dimb,
           STDDEV_POP(spr) AS b_sprv,
           AVG(CASE WHEN bn + an > 0 THEN (bv + av) / (bn + an) ELSE NULL END) AS b_vpo,
           AVG(CASE WHEN bp1 > 0 AND ap1 > bp1 AND bv1 + av1 > 0
                    THEN ((bp1 * av1 + ap1 * bv1) / (bv1 + av1)) / ((ap1 + bp1) / 2.0) - 1 ELSE NULL END) AS b_mp,
           LN((AVG(bv + av) + 1) / (AVG(vol) + 1)) AS b_dep
    FROM cte_seg
    WHERE time_segment != -1
    GROUP BY instrument_id, trading_day, time_segment
)
SELECT
    CAST(CONCAT(f.trading_day, ' ',
                strftime(strptime(LPAD(f.time_segment, 6, '0'), '%H%M%S'), '%H:%M:%S')) AS DATETIME) AS date,
    all_instruments.instrument,
    f.b_imb, f.b_spr, f.b_conc, f.b_oimb, f.b_imb1, f.b_dimb, f.b_sprv, f.b_vpo, f.b_mp, f.b_dep
FROM cte_window f
LEFT JOIN all_instruments USING (instrument_id)
"""
BOOK30_COLS = ["b_imb", "b_spr", "b_conc", "b_oimb"]
BOOK2_COLS = ["b_imb1", "b_dimb", "b_sprv", "b_vpo"]
BOOK3_COLS = ["b_mp", "b_dep"]   # 微观价格相对中间价的偏离、1–3 档总挂单量 ÷ 每分钟成交量（对数）；reports/15 的 T4


def _book30_dai(table, sd, ed, instruments):
    df = dai.query(_BOOK30_SQL.format(table=table), filters={"date": [sd, ed]}, compression=True).df()
    df = df.dropna(subset=["instrument"])
    df["instrument"] = df["instrument"].astype(str)
    df = df[df["instrument"].isin(set(instruments))].drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    for c in BOOK30_COLS + BOOK2_COLS + BOOK3_COLS:
        df[c] = df[c].astype("float64")
    return df[["date", "instrument"] + BOOK30_COLS + BOOK2_COLS + BOOK3_COLS]


DaiBackend.book30 = lambda self, sd, ed, instruments: _book30_dai(self.table, sd, ed, instruments)

# 收盘集合竞价（WIN_GROUPS 含 auc 时使用，10-23）：每个交易日 14:56 收盘、15:00 竞价成交价与成交量、全天成交量，
# 结果挂在当天 15:00 窗口上。日线 token 只取截面所在交易日之前的日子，日内 token 取所在窗口前一交易日的值，当天的竞价不进特征。
# 本地用同一段 SQL 在本地 1 分钟数据上预先算好（scripts/build_auc_cache.py）。CTE 名 cte_window 让本地模拟 dai 用 DuckDB 原样执行。
_AUC_SQL = """
WITH cte_raw AS (
    SELECT
        instrument_id,
        strftime(date, '%Y-%m-%d') AS trading_day,
        strftime(date, '%H%M%S') AS hms,
        CAST(close AS DOUBLE) AS c,
        CAST(COALESCE(volume, 0) AS DOUBLE) AS v
    FROM {table}
),
cte_window AS (
    SELECT instrument_id, trading_day,
           MAX(CASE WHEN hms = '145600' AND c > 0 THEN c ELSE NULL END) AS a_c1456,
           MAX(CASE WHEN hms = '150000' AND c > 0 THEN c ELSE NULL END) AS a_c1500,
           SUM(CASE WHEN hms = '150000' THEN v ELSE 0 END) AS a_v1500,
           SUM(v) AS a_vday
    FROM cte_raw
    GROUP BY instrument_id, trading_day
)
SELECT
    CAST(CONCAT(f.trading_day, ' 15:00:00') AS DATETIME) AS date,
    all_instruments.instrument,
    f.a_c1456, f.a_c1500, f.a_v1500, f.a_vday
FROM cte_window f
LEFT JOIN all_instruments USING (instrument_id)
"""
AUC_COLS = ["a_c1456", "a_c1500", "a_v1500", "a_vday"]


def _auc_dai(table, sd, ed, instruments):
    df = dai.query(_AUC_SQL.format(table=table), filters={"date": [sd, ed]}, compression=True).df()
    df = df.dropna(subset=["instrument"])
    df["instrument"] = df["instrument"].astype(str)
    df = df[df["instrument"].isin(set(instruments))].drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    for c in AUC_COLS:
        df[c] = df[c].astype("float64")
    return df[["date", "instrument"] + AUC_COLS]


DaiBackend.auc_daily = lambda self, sd, ed, instruments: _auc_dai(self.table, sd, ed, instruments)


def _backend(table_or_backend):
    return DaiBackend(table_or_backend) if isinstance(table_or_backend, str) else table_or_backend


def pool(sd, ed, backend=None):
    """区间内中证 1000 成分股代码（同 demo 接口）。"""
    return (backend or DaiBackend(None)).pool(sd, ed)


# ---------- 切窗 ----------
def _cube(df, times_col, inst, cols):
    grid = np.sort(df[times_col].unique())
    ti = np.searchsorted(grid, df[times_col].values)
    ii = np.searchsorted(inst, df["instrument"].values)
    cube = np.full((len(inst), len(grid), len(cols)), np.nan, np.float32)
    cube[ii, ti] = df[cols].to_numpy(np.float32)
    return grid, cube


def resample_30m(bars):
    """1m -> 30 分钟 bar（原样重采样，不引入跨字段运算）。时间戳 = 窗口结束时刻。"""
    hm = bars["date"].dt.strftime("%H:%M").values
    b = np.minimum(np.searchsorted(np.array(BUCKETS), hm, side="left"), len(BUCKETS) - 1)
    off = pd.to_timedelta(np.array([f"{x}:00" for x in BUCKETS])[b])
    w = bars.assign(bucket=bars["date"].dt.normalize() + off).sort_values(["instrument", "date"])
    agg = {}
    for c in FEATURE_COLS:
        if c == "open":
            agg[c] = "first"
        elif c == "high":
            agg[c] = "max"
        elif c == "low":
            agg[c] = "min"
        elif c in ("volume", "amount", "deal_number"):
            agg[c] = "sum"
        else:
            agg[c] = "last"
    return w.groupby(["instrument", "bucket"], sort=False).agg(agg).reset_index()


def _coarse_from_cube(grid, cube):
    """在 1m 网格立方体上按 30 分钟窗口分段聚合（与 resample_30m 等价，numpy 实现）。
    返回 (cgrid 窗口结束时刻, ccube (N_inst, N_bucket, F))。窗口内无任何行的股票记为 NaN。"""
    t = pd.DatetimeIndex(grid)
    hm = t.strftime("%H:%M").values
    b = np.minimum(np.searchsorted(np.array(BUCKETS), hm, side="left"), len(BUCKETS) - 1)
    bend = (t.normalize() + pd.to_timedelta(np.array([f"{x}:00" for x in BUCKETS])[b])).values
    starts = np.flatnonzero(np.r_[True, bend[1:] != bend[:-1]])
    ends = np.r_[starts[1:], len(grid)] - 1
    cgrid = bend[starts]
    n_i, _, n_f = cube.shape
    out = np.full((n_i, len(starts), n_f), np.nan, np.float32)
    has_row = np.add.reduceat(np.isfinite(cube[..., FEATURE_COLS.index("volume")]).astype(np.int32), starts, axis=1) > 0
    pos = np.arange(cube.shape[1])
    for k, c in enumerate(FEATURE_COLS):
        x = cube[..., k]
        if c == "high":
            v = np.fmax.reduceat(x, starts, axis=1)
        elif c == "low":
            v = np.fmin.reduceat(x, starts, axis=1)
        elif c in ("volume", "amount", "deal_number"):
            v = np.add.reduceat(np.nan_to_num(x), starts, axis=1)
        elif c == "open":                                       # 窗口内第一个有效值
            idx = np.where(np.isfinite(x), pos, len(pos))
            idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1][:, starts]
            ok = idx <= ends
            v = np.where(ok, np.take_along_axis(x, np.minimum(idx, len(pos) - 1), 1), np.nan)
        else:                                                   # 窗口内最后一个有效值（快照字段与 close）
            idx = np.where(np.isfinite(x), pos, -1)
            idx = np.maximum.accumulate(idx, axis=1)[:, ends]
            ok = idx >= starts
            v = np.where(ok, np.take_along_axis(x, np.maximum(idx, 0), 1), np.nan)
        out[..., k] = np.where(has_row, v, np.nan)
    return cgrid, out


SUMMARY_BASE = ["r30", "r120", "r240", "r_yclose", "r_dayvwap", "r_vwap240", "r_5d", "vol240",
                "r_dayvwap_vn", "r_vwap240_vn", "pos_day", "spread", "imb1_30", "illiq240", "log_price", "log_amt240"]
SUMMARY_FULL = ["ord_imb3_30", "vol_imb3_30"]


SUMMARY_SLOW = ["vol240", "illiq240", "log_price", "log_amt240", "spread"]   # 慢变量 / 风格类，rev 集合中去掉
CHAR_COLS = ["c_size", "c_ivol", "c_illiq", "c_lowp"]                      # 中性化用的四个风格特征


def summary_all_names():
    return SUMMARY_BASE + (SUMMARY_FULL if "bid_num_orders1" in FEATURE_COLS else [])


def summary_names():
    names = summary_all_names()
    return [n for n in names if n not in SUMMARY_SLOW] if SUMMARY_SET == "rev" else names


def _ffill_t(x):
    """沿时间轴（axis=1）前向填充 NaN。"""
    pos = np.arange(x.shape[1])
    idx = np.maximum.accumulate(np.where(np.isfinite(x), pos, 0), axis=1)
    out = np.take_along_axis(x, idx, 1)
    return out


def _summary_features(grid, cube, si, e, adj=None):
    """截面摘要特征：只用网格下标 <= e（截面时刻及之前）的数据。返回 (N, K) float32（未标准化）。
    cube 中的价格已复权；adj 为复权因子（用于还原名义价格水平）。"""
    f = {c: cube[..., FEATURE_COLS.index(c)] for c in
         ("close", "high", "low", "volume", "amount", "bid_price1", "ask_price1", "bid_volume1", "ask_volume1")}
    c, b1, a1 = f["close"], f["bid_price1"], f["ask_price1"]
    mid = np.where((b1 > 0) & (a1 > b1), (b1 + a1) / 2, np.nan)
    px = _ffill_t(np.where(np.isfinite(mid), mid, np.where(c > 0, c, np.nan)))
    cf = _ffill_t(np.where(c > 0, c, np.nan))
    T_ = len(grid)
    day = pd.DatetimeIndex(grid).normalize().values
    day_start = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
    seg = np.searchsorted(day_start, np.arange(T_), side="right") - 1      # 每个网格点所属交易日序号
    ds = day_start[seg]                                                     # 当日第一根下标
    vol = np.nan_to_num(f["volume"]).astype(np.float64)                     # 累计和用 float64，避免大数相减丢精度
    amt = np.nan_to_num(f["amount"]).astype(np.float64)
    cv = np.concatenate([np.zeros((len(cube), 1)), np.cumsum(vol, 1)], 1)  # cv[:, k] = 前 k 根之和
    ca = np.concatenate([np.zeros((len(cube), 1)), np.cumsum(amt, 1)], 1)
    lr = np.diff(np.log(px.astype(np.float64)), axis=1, prepend=np.nan)
    lr = np.nan_to_num(lr)
    c1 = np.concatenate([np.zeros((len(cube), 1)), np.cumsum(lr, 1)], 1)
    c2 = np.concatenate([np.zeros((len(cube), 1)), np.cumsum(lr * lr, 1)], 1)
    ia = np.nan_to_num(np.abs(lr) / np.where(amt > 0, amt, np.nan))
    ci = np.concatenate([np.zeros((len(cube), 1)), np.cumsum(ia, 1)], 1)
    bv, av = np.nan_to_num(f["bid_volume1"]), np.nan_to_num(f["ask_volume1"])
    imb = np.where(bv + av > 0, (bv - av) / np.maximum(bv + av, 1e-9), 0.0)
    cimb = np.concatenate([np.zeros((len(cube), 1)), np.cumsum(imb, 1)], 1)

    def at(x, k):                                                           # x[si, k]，k<0 记 NaN
        kk = np.maximum(k, 0)
        return np.where(k >= 0, x[si, kk], np.nan)

    def wsum(cs, w):                                                        # 截至 e（含）的最近 w 根之和
        lo = np.maximum(e + 1 - w, 0)
        return cs[si, e + 1] - cs[si, lo], (e + 1 - lo)

    p0 = px[si, e]
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["r30"] = np.log(p0 / at(px, e - 30))
        out["r120"] = np.log(p0 / at(px, e - 120))
        out["r240"] = np.log(p0 / at(px, e - 240))
        out["r_yclose"] = np.log(p0 / at(cf, ds[e] - 1))                    # 前一交易日最后收盘
        dv = cv[si, e + 1] - cv[si, ds[e]]
        da = ca[si, e + 1] - ca[si, ds[e]]
        out["r_dayvwap"] = np.log(p0 / np.where(dv > 0, da / np.maximum(dv, 1e-9), np.nan))
        v240, _ = wsum(cv, 240)
        a240, _ = wsum(ca, 240)
        out["r_vwap240"] = np.log(p0 / np.where(v240 > 0, a240 / np.maximum(v240, 1e-9), np.nan))
        out["r_5d"] = np.log(p0 / at(px, e - 5 * 250))
        s1, n1 = wsum(c1, 240)
        s2, _ = wsum(c2, 240)
        sig = np.sqrt(np.maximum(s2 / n1 - (s1 / n1) ** 2, 0)) * np.sqrt(240)
        out["vol240"] = np.log(sig + 1e-6)
        out["r_dayvwap_vn"] = out["r_dayvwap"] / (sig + 1e-4)
        out["r_vwap240_vn"] = out["r_vwap240"] / (sig + 1e-4)
        hi = np.full(len(si), np.nan)
        lo_ = np.full(len(si), np.nan)
        for d0 in np.unique(ds[e]):                                         # 当日开盘至 e 的最高/最低
            m = ds[e] == d0
            rows, ends = si[m], e[m]
            span = np.arange(d0, ends.max() + 1)
            hh = np.fmax.accumulate(f["high"][rows][:, span], axis=1)
            ll = np.fmin.accumulate(f["low"][rows][:, span], axis=1)
            hi[m] = hh[np.arange(len(rows)), ends - d0]
            lo_[m] = ll[np.arange(len(rows)), ends - d0]
        rng = hi - lo_
        out["pos_day"] = np.where(rng > 0, (cf[si, e] - lo_) / np.where(rng > 0, rng, 1), 0.5)
        out["spread"] = (a1[si, e] - b1[si, e]) / mid[si, e]
        ims, n30 = wsum(cimb, 30)
        out["imb1_30"] = ims / n30
        il, _ = wsum(ci, 240)
        out["illiq240"] = np.log(il * 1e8 + 1e-6)
        out["log_price"] = np.log(cf[si, e] / (adj[si, e] if adj is not None else 1.0))   # 名义价格（未复权）
        out["log_amt240"] = np.log1p(a240)
        if "bid_num_orders1" in FEATURE_COLS:
            def lvl_imb(prefix_b, prefix_a, w):
                nb = sum(w[k] * np.nan_to_num(cube[..., FEATURE_COLS.index(f"{prefix_b}{k + 1}")]) for k in range(3))
                na = sum(w[k] * np.nan_to_num(cube[..., FEATURE_COLS.index(f"{prefix_a}{k + 1}")]) for k in range(3))
                x = np.where(nb + na > 0, (nb - na) / np.maximum(nb + na, 1e-9), 0.0)
                cx = np.concatenate([np.zeros((len(cube), 1)), np.cumsum(x, 1)], 1)
                sm, nn_ = wsum(cx, 30)
                return sm / nn_
            out["ord_imb3_30"] = lvl_imb("bid_num_orders", "ask_num_orders", (1.0, 2.0, 3.0))
            out["vol_imb3_30"] = lvl_imb("bid_volume", "ask_volume", (1.0, 1.0, 1.0))
    return np.stack([out[k] for k in summary_names()], 1).astype(np.float32)


def _roll(x, k=40, min_n=10):
    """沿 axis=1 的滚动和与有效个数（窗口含当前位置）。x 中 NaN 视为缺失。"""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    cs = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(v, 1)], 1)
    cn = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(ok, 1)], 1)
    hi = np.arange(1, x.shape[1] + 1)
    lo = np.maximum(hi - k, 0)
    sm, n = cs[:, hi] - cs[:, lo], cn[:, hi] - cn[:, lo]
    return sm, np.where(n >= min_n, n, np.nan)


def _style_chars(grid, cube, si, t, adj=None):
    """四个风格特征（与因子赛道 neutral4 同口径，在 30 分钟窗口上计算，只用结束时刻 <= 截面的窗口）：
    c_size = -ln(1 + 近 40 窗成交额)，c_ivol = -近 40 窗残差收益标准差（残差 = 收益 - beta x 截面均值，
    beta 为 40 窗滚动、截断 [-1, 4]、不足取 1），c_illiq = 近 40 窗 |收益| / 成交额(亿) 均值，c_lowp = -ln(名义价格)。"""
    cgrid, cc = _coarse_from_cube(grid, cube)
    k = FEATURE_COLS.index
    b1, a1, c = cc[..., k("bid_price1")], cc[..., k("ask_price1")], cc[..., k("close")]
    px = np.where((b1 > 0) & (a1 > b1), (b1 + a1) / 2, np.where(c > 0, c, np.nan)).astype(np.float64)
    px = _ffill_t(px)
    r = np.diff(np.log(px), axis=1, prepend=np.nan)
    amt = cc[..., k("amount")].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.nanmean(r, axis=0, keepdims=True) * np.ones_like(r)
        m = np.where(np.isfinite(r), m, np.nan)
        sx, n = _roll(r)
        sy, _ = _roll(m)
        sxy, _ = _roll(r * m)
        syy, _ = _roll(m * m)
        ex, ey = sx / n, sy / n
        var_m = syy / n - ey ** 2
        beta = np.where(var_m > 1e-12, (sxy / n - ex * ey) / np.where(var_m > 1e-12, var_m, 1), np.nan)
        beta = np.where(np.isfinite(beta), np.clip(beta, -1, 4), 1.0)
        res = r - beta * m
        s1, nr = _roll(res)
        s2, _ = _roll(res * res)
        ivol = np.sqrt(np.maximum((s2 - s1 ** 2 / nr) / np.maximum(nr - 1, 1), 0))
        il = np.abs(r) / np.where(amt > 0, amt / 1e8, np.nan)
        si_, ni = _roll(il)
        sa, na = _roll(np.where(np.isfinite(amt), amt, np.nan))
    cb = np.searchsorted(cgrid, t, side="right") - 1
    ok = cb >= 0
    cbx = np.maximum(cb, 0)
    pick = lambda a: np.where(ok, a[si, cbx], np.nan)  # noqa: E731
    e = np.searchsorted(grid, t, side="right") - 1
    cl = cube[..., k("close")]
    cl = _ffill_t(np.where(cl > 0, cl, np.nan))
    nominal = cl[si, e] / (adj[si, e] if adj is not None else 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.stack([-np.log1p(pick(sa / na * na)), -pick(ivol), pick(si_ / ni), -np.log(nominal)], 1)
    return out.astype(np.float32)


def neutralize(values, chars, dates, frac=1.0, rank_values=True):
    """逐截面中性化（与因子赛道 factor_lib.neutralize 一致）：
    y = 截面居中百分位排名（rank_values=False 时直接用 values），X = 各特征居中百分位排名（缺失记 0），
    逐截面 OLS（含截距），返回 y - frac x 拟合值（不扣截距）。"""
    d = pd.Series(dates)
    v = pd.Series(np.asarray(values, np.float64))
    y = (v.groupby(d.values).rank(pct=True) - 0.5) if rank_values else v
    X = np.column_stack([(pd.Series(chars[:, j]).groupby(d.values).rank(pct=True) - 0.5).fillna(0.0).to_numpy()
                         for j in range(chars.shape[1])])
    ok = y.notna().to_numpy()
    codes, uniq = pd.factorize(d.values)
    kk, G = X.shape[1], len(uniq)
    Xc = np.column_stack([np.ones(len(X)), X]) * ok[:, None]
    yv = np.nan_to_num(y.to_numpy()) * ok
    XtX = np.zeros((G, kk + 1, kk + 1))
    Xty = np.zeros((G, kk + 1))
    for i in range(kk + 1):
        Xty[:, i] = np.bincount(codes, Xc[:, i] * yv, minlength=G)
        for j in range(i, kk + 1):
            w = np.bincount(codes, Xc[:, i] * Xc[:, j], minlength=G)
            XtX[:, i, j] = w
            XtX[:, j, i] = w
    b = np.linalg.solve(XtX + 1e-9 * np.eye(kk + 1), Xty[..., None])[..., 0]
    fit = (X * b[codes, 1:]).sum(axis=1)
    return np.where(ok, y.to_numpy() - frac * fit, np.nan)


def need_chars():
    return LABEL_NEUTRAL > 0 or OUT_NEUTRAL > 0


def cs_zscore(X, dates):
    """逐截面 z-score（截断 ±5，缺失置 0）。X (N, K)。"""
    codes = pd.factorize(dates)[0]
    out = np.zeros_like(X, dtype=np.float32)
    cnt = np.bincount(codes).astype(np.float64)
    for k in range(X.shape[1]):
        x = X[:, k].astype(np.float64)
        ok = np.isfinite(x)
        n = np.bincount(codes, ok.astype(np.float64))
        s1 = np.bincount(codes, np.where(ok, x, 0.0))
        s2 = np.bincount(codes, np.where(ok, x * x, 0.0))
        mu = s1 / np.maximum(n, 1)
        sd = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mu ** 2, 0))
        z = (x - mu[codes]) / (sd[codes] + 1e-12)
        out[:, k] = np.nan_to_num(np.clip(z, -5, 5), nan=0.0, posinf=0.0, neginf=0.0)
    del cnt
    return out


def build_windows(bars, sections, seq_len=None, coarse_len=None, compute_chars=None):
    """为每个 (date, instrument) 截面取：
         1m：全局分钟网格上 date <= 截面时间的最近 seq_len 根（缺行为 NaN）；
         30m：结束时刻 <= 截面时间的最近 coarse_len 根（不足补 NaN）。
    SUMMARY=True 时另返回截面摘要特征 Xs (N, K)（未标准化）。
    返回 Xf (N, seq_len, F), Xc (N, coarse_len, F) 或 None, Xs 或 None, keys。"""
    seq_len = SEQ_LEN if seq_len is None else seq_len
    coarse_len = COARSE_LEN if coarse_len is None else coarse_len
    inst = np.sort(sections["instrument"].unique())
    b = bars[bars["instrument"].isin(inst)]
    has_adj = "adjust_factor" in b.columns
    grid, cube = _cube(b, "date", inst, FEATURE_COLS + (["adjust_factor"] if has_adj else []))
    adj = None
    if has_adj:                                                 # 复权：跨日比较价格时消除除权跳变
        adj = cube[..., -1]
        cube = cube[..., :-1]
        adj = np.where(adj > 0, adj, np.nan)
        adj = _ffill_t(adj)
        adj = np.where(np.isfinite(adj), adj, 1.0).astype(np.float32)
        for k, c in enumerate(FEATURE_COLS):
            if _is_price(c):
                cube[..., k] *= adj                             # 后复权价格
            elif c == "volume" or "_volume" in c:
                cube[..., k] /= adj                             # 股数按同一比例调整；成交额、笔数不变

    s = sections.drop_duplicates(["date", "instrument"])
    end = np.searchsorted(grid, s["date"].values, side="right")
    ok = end >= seq_len
    s, end = s[ok], end[ok]
    si = np.searchsorted(inst, s["instrument"].values)
    idx = end[:, None] - seq_len + np.arange(seq_len)[None, :]
    Xf = cube[si[:, None], idx]
    if len(s) and not (grid[idx[:, -1]] <= s["date"].values).all():
        raise RuntimeError("特征时点泄漏 (1m)")
    Xs = _summary_features(grid, cube, si, end - 1, adj) if SUMMARY else None
    s = s.copy()
    if (need_chars() if compute_chars is None else compute_chars):
        ch = _style_chars(grid, cube, si, s["date"].values, adj)
        for j, cn in enumerate(CHAR_COLS):
            s[cn] = ch[:, j]
    Xc = None
    if coarse_len:
        cgrid, ccube = _coarse_from_cube(grid, cube)
        del cube
        ccube = np.concatenate([np.full((len(inst), coarse_len, len(FEATURE_COLS)), np.nan, np.float32), ccube], 1)
        cend = np.searchsorted(cgrid, s["date"].values, side="right") + coarse_len
        cidx = cend[:, None] - coarse_len + np.arange(coarse_len)[None, :]
        Xc = ccube[si[:, None], cidx]
        last = cidx[:, -1] - coarse_len
        if len(s) and not (cgrid[np.maximum(last, 0)][last >= 0] <= s["date"].values[last >= 0]).all():
            raise RuntimeError("特征时点泄漏 (30m)")
    else:
        del cube
    return Xf, Xc, Xs, s.reset_index(drop=True)


# ---------- 数值变换与标准化 ----------
def transform(Xf, Xc=None, norm=None):
    """逐样本数值变换（两个尺度共用同一组参照值）：
    rel ：量额笔数 log1p；价格除以样本最后一个有效 1m close 后取 log。
    rel2：价格同 rel；量额笔数除以本样本 1m 窗口内该字段的均值后取 log1p（与计量单位无关）。
    rel3：同 rel2，但 NORM_SHARE 里同一组的字段（买卖各档挂单量、买卖各档委托笔数）共用一个参照（组内各字段均值的平均），
          rel2 下每个字段各自除以自己的均值，买一量是卖一量的几倍这类相对水平就丢了。"""
    norm = norm or NORM
    out = []
    ref = sref = None
    if norm in ("rel", "rel2", "rel3"):
        close = Xf[..., FEATURE_COLS.index("close")].astype(np.float32)
        ok = close > 0
        last = np.where(ok.any(1), Xf.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1), 0)
        ref = np.where(ok.any(1), close[np.arange(len(close)), last], np.nan).astype(np.float32)
    si = [i for i, c in enumerate(FEATURE_COLS) if not _is_price(c)]
    pi = [i for i, c in enumerate(FEATURE_COLS) if _is_price(c)]
    if norm in ("rel2", "rel3"):
        with np.errstate(invalid="ignore"):
            sref = np.nanmean(np.clip(Xf[..., si].astype(np.float32), 0, None), axis=1)   # (N, n_size)
        if norm == "rel3":
            names = [FEATURE_COLS[i] for i in si]
            for grp in NORM_SHARE:
                gi = [names.index(c) for c in grp if c in names]
                if len(gi) > 1:
                    with np.errstate(invalid="ignore"):
                        sh = np.nanmean(sref[:, gi], axis=1)
                    sref[:, gi] = sh[:, None]
        sref = np.where(np.isfinite(sref) & (sref > 0), sref, 1.0).astype(np.float32)
    for X in (Xf, Xc):
        if X is None:
            out.append(None)
            continue
        X = X.astype(np.float32)
        X[~np.isfinite(X)] = np.nan
        if norm in ("rel2", "rel3"):
            X[..., si] = np.log1p(np.clip(X[..., si], 0, None) / sref[:, None, :])
        elif norm != "raw":
            X[..., si] = np.log1p(np.clip(X[..., si], 0, None))
        if norm in ("rel", "rel2", "rel3"):
            p = X[..., pi]
            p[p <= 0] = np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                X[..., pi] = np.log(p / ref[:, None, None])
        out.append(X.astype(np.float16))
    return out[0], out[1]


def fit_stats(X):
    flat = X.reshape(-1, X.shape[-1]).astype(np.float32)
    return np.nanmean(flat, 0).astype(np.float32), (np.nanstd(flat, 0) + 1e-6).astype(np.float32)


def _std_into(out, start, X, m, s):
    z = (X.astype(np.float32) - m) / s
    out[start:start + len(X)] = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def standardize(X, m, s, chunk=200000):
    out = np.empty(X.shape, np.float16)
    for i in range(0, len(X), chunk):
        _std_into(out, i, X[i:i + chunk], m, s)
    return out


# ---------- 分片构建 ----------
def month_chunks(sd, ed):
    sd, ed = pd.Timestamp(sd), pd.Timestamp(ed)
    for st in pd.date_range(sd.normalize().replace(day=1), ed, freq="MS"):
        lo = max(st, sd)
        hi = min(st + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59, seconds=59), ed)
        if lo <= hi:
            yield lo, hi


def _fmt(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def iter_chunks(backend, sd, ed, instruments, with_label, norm=None):
    """逐月产出 (Xf, Xc, Xs, y, keys)：Xf/Xc 已做数值变换（float16，未标准化），Xs 已逐截面 z-score。"""
    for lo, hi in month_chunks(sd, ed):
        t0 = time.time()
        sec = backend.sections(_fmt(lo), _fmt(hi), instruments)
        if with_label:
            lab = backend.labels(_fmt(lo), _fmt(hi), instruments)
            lab = lab.replace([np.inf, -np.inf], np.nan).dropna(subset=["y"])
            # 摘要特征的截面 z-score 要在完整截面上计算（与推理一致），因此先保留无标签样本，标准化后再剔除
            sec = sec.merge(lab[["date", "instrument", "y"]], on=["date", "instrument"],
                            how="left" if SUMMARY else "inner")
        if sec.empty:
            continue
        bars = backend.bars(_fmt(lo - pd.Timedelta(days=BUFFER_DAYS)), _fmt(hi),
                            sorted(sec["instrument"].unique().tolist()), FEATURE_COLS + ["adjust_factor"])
        Xf, Xc, Xs, keys = build_windows(bars, sec)
        del bars
        if not len(keys):
            continue
        if Xs is not None:                                      # 分片按月切分，同一截面的全部股票都在本片内
            Xs = cs_zscore(Xs, keys["date"].values).astype(np.float16)
        if with_label and SUMMARY:
            keep = keys["y"].notna().to_numpy()
            Xf, Xs = Xf[keep], Xs[keep]
            Xc = Xc[keep] if Xc is not None else None
            keys = keys[keep].reset_index(drop=True)
            if not len(keys):
                continue
        Xf, Xc = transform(Xf, Xc, norm)
        y = keys["y"].to_numpy(np.float32) if with_label else None
        logger.info("分片完成", start=_fmt(lo)[:10], samples=len(keys), elapsed=round(time.time() - t0, 1))
        yield Xf, Xc, Xs, y, keys[["date", "instrument"] + [c for c in CHAR_COLS if c in keys.columns]]


def _standardize_inplace(X, m, s, chunk=200000):
    """原地分块标准化（float16），NaN/inf 置 0，不额外占用一份数据的内存。"""
    for i in range(0, len(X), chunk):
        z = (X[i:i + chunk].astype(np.float32) - m) / s
        X[i:i + chunk] = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def build_dataset(table, sd, ed, mode, instruments, stats=None):
    """构建 30 分钟截面样本。table 为 bar1m 表名（平台）或 backend 对象（本地）。

    返回 (X, y 或 None, keys, stats)；X = (Xf, Xc 或 None, Xs 或 None)，已标准化为 float16；
    stats = (mean_f, std_f, mean_c, std_c)，只在训练集上计算。
    内存：先按截面数上限预分配最终数组，逐月写入后原地标准化，峰值约为一份样本数据加单月临时量。"""
    if ARCH == "e04":
        return e04_build_dataset(table, sd, ed, mode, instruments, stats)
    backend = _backend(table)
    with_label = mode == "train"
    n_max = sum(len(backend.sections(_fmt(lo), _fmt(hi), instruments)) for lo, hi in month_chunks(sd, ed))
    Xf = Xc = Xs = None
    ys, ks, pos = [], [], 0
    for cf, cc, cs, cy, ck in iter_chunks(backend, sd, ed, instruments, with_label):
        if Xf is None:                                          # 未写入的尾部不会实际占用内存
            Xf = np.empty((n_max,) + cf.shape[1:], np.float16)
            Xc = np.empty((n_max,) + cc.shape[1:], np.float16) if cc is not None else None
            Xs = np.empty((n_max,) + cs.shape[1:], np.float16) if cs is not None else None
        k = len(ck)
        Xf[pos:pos + k] = cf
        if Xc is not None:
            Xc[pos:pos + k] = cc
        if Xs is not None:
            Xs[pos:pos + k] = cs
        pos += k
        ks.append(ck)
        if with_label:
            ys.append(cy)
        del cf, cc, cs
    if not pos:
        raise RuntimeError(f"build_dataset 无样本 (mode={mode}, {sd}~{ed})")
    Xf = Xf[:pos]
    Xc = Xc[:pos] if Xc is not None else None
    Xs = Xs[:pos] if Xs is not None else None
    if stats is None:
        idx = np.sort(np.random.RandomState(SEED).choice(pos, min(pos, STATS_SAMPLE), replace=False))
        mf, sf = fit_stats(Xf[idx])
        mc, sc = fit_stats(Xc[idx]) if Xc is not None else (None, None)
        stats = (mf, sf, mc, sc)
    _standardize_inplace(Xf, stats[0], stats[1])
    if Xc is not None:
        _standardize_inplace(Xc, stats[2], stats[3])
    keys = pd.concat(ks, ignore_index=True)
    del ks
    logger.info(f"{mode} 集构建完成", samples=len(keys))
    return (Xf, Xc, Xs), (np.concatenate(ys) if with_label else None), keys, stats


def make_target(y, keys, target=None):
    """cs_z：每个截面内 z-score（样本标准差），截断 ±5，缺失置 0。用 numpy 分组计算以节省内存。"""
    target = target or TARGET
    y = np.asarray(y, np.float64)
    if target == "raw":
        return y.astype(np.float32)
    codes = pd.factorize(keys["date"])[0]
    if target == "cs_rank":                                     # 秩 → 正态分位数（缺失置 0）
        return rank_gauss(y, codes)
    cnt = np.bincount(codes).astype(np.float64)
    mean = np.bincount(codes, y) / cnt
    var = (np.bincount(codes, y * y) - cnt * mean ** 2) / np.maximum(cnt - 1, 1)
    z = (y - mean[codes]) / (np.sqrt(np.maximum(var, 0))[codes] + 1e-8)
    return np.nan_to_num(np.clip(z, -5, 5), nan=0.0).astype(np.float32)


def rank_gauss(y, codes):
    """逐截面（codes 相同为一组）平均秩 → (秩 − 0.5) ÷ 有效个数 → 标准正态分位数；缺失置 0。"""
    s = pd.Series(np.asarray(y, np.float64))
    r = s.groupby(codes).rank(method="average").to_numpy()
    n = s.notna().groupby(codes).transform("sum").to_numpy().astype(np.float64)
    p = np.clip((r - 0.5) / np.maximum(n, 1.0), 1e-6, 1 - 1e-6)
    z = (np.sqrt(2.0) * torch.erfinv(torch.from_numpy(2.0 * p - 1.0))).numpy()
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


@torch.inference_mode()
def predict_array(models, X, device, batch=4096):
    """models 为单个模型或模型列表（取平均）；X = (Xf, Xc 或 None, Xs 或 None)。"""
    models = models if isinstance(models, (list, tuple)) else [models]
    if device.type == "cpu" and torch.get_num_threads() < (os.cpu_count() or 1):
        torch.set_num_threads(os.cpu_count() or 1)              # CPU 推理用满全部核心
    Xf, Xc = X[0], X[1]
    Xs = X[2] if len(X) > 2 else None
    out = []
    for m in models:
        m.eval()
    for i in range(0, len(Xf), batch):
        xb = torch.from_numpy(np.ascontiguousarray(Xf[i:i + batch])).to(device).float()
        xc = torch.from_numpy(np.ascontiguousarray(Xc[i:i + batch])).to(device).float() if Xc is not None else None
        xs = torch.from_numpy(np.ascontiguousarray(Xs[i:i + batch])).to(device).float() if Xs is not None else None
        out.append(np.mean([m(xb, xc, xs).float().cpu().numpy() for m in models], axis=0))
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def predict_range(models, table, sd, ed, instruments, stats, device, norm=None, out_neutral=None, keep_chars=False, zero=None):
    """推理：逐月切窗、标准化、预测，返回 DataFrame[date, instrument, factor]。每个截面只用自身的输入窗口。
    out_neutral>0 时对每个截面的预测排名做四个风格特征中性化（只用当前截面数据）。"""
    if ARCH == "e04":
        return e04_predict_range(models, table, sd, ed, instruments, stats, device, out_neutral, keep_chars, zero)
    out_neutral = OUT_NEUTRAL if out_neutral is None else out_neutral
    backend = _backend(table)
    parts = []
    for Xf, Xc, Xs, _, keys in iter_chunks(backend, sd, ed, instruments, with_label=False, norm=norm):
        X = (standardize(Xf, stats[0], stats[1]), standardize(Xc, stats[2], stats[3]) if Xc is not None else None, Xs)
        keys = keys.copy()
        keys["factor"] = predict_array(models, X, device).astype(np.float64)
        if out_neutral and out_neutral > 0:                     # 分片按月切分，同一截面的全部股票都在本片内
            keys["factor"] = neutralize(keys["factor"].to_numpy(), keys[CHAR_COLS].to_numpy(),
                                        keys["date"].values, out_neutral)
        parts.append(keys if keep_chars else keys[["date", "instrument", "factor"]])
    if not parts:
        raise RuntimeError(f"predict_range 无样本 ({sd}~{ed})")
    return pd.concat(parts, ignore_index=True)


# ==== 模型存/读：文本 JSON ====
def _tensors(sd):
    out = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        out[k] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape),
                  "data": t.reshape(-1).tolist()}
    return out


def save_model(ckpt, model_path=MODEL_PATH):
    """ckpt["state_dicts"]：模型列表的 state_dict；其余字段原样写入。"""
    payload = {k: v for k, v in ckpt.items() if k != "state_dicts"}
    payload["state_dicts"] = [_tensors(sd) for sd in ckpt["state_dicts"]]
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sds = payload.pop("state_dicts", None) or [payload.pop("state_dict")]
    payload["state_dicts"] = [
        {k: torch.tensor(m["data"], dtype=getattr(torch, m["dtype"])).reshape(m["shape"]).to(map_location)
         for k, m in sd.items()} for sd in sds]
    return payload


def build_models(ckpt, device):
    """由 load_model 的结果重建模型列表。"""
    models = []
    for sd in ckpt["state_dicts"]:
        m = (E04Net if ckpt.get("arch") == "e04" else StockTransformer)(**ckpt["model_cfg"]).to(device)
        m.load_state_dict({k: v.to(device) for k, v in sd.items()})
        m.eval()
        models.append(m)
    return models


def configure_from_ckpt(ckpt):
    """推理前按权重文件里的配置设置字段集合与回看长度，保证与训练一致。"""
    global SEQ_LEN, COARSE_LEN, NORM, SUMMARY, SUMMARY_SET, SUMMARY_POOL, OUT_NEUTRAL, ARCH, WIN_LOOKBACK_DAYS, ZERO_0930, PEER_FROM_DAILY
    set_fields(ckpt.get("field_set", "base"))
    ARCH = ckpt.get("arch", "v1")
    if ARCH == "e04":
        global E04_NORM, GBDT_CK
        E04_NORM = ckpt.get("norm", "rel2") if ckpt.get("norm") in ("rel2", "rel3") else "rel2"
        GBDT_CK = ckpt.get("gbdt")
        SEQ_LEN, COARSE_LEN, SUMMARY, NORM = ckpt["model_cfg"]["seq_len"], 0, False, E04_NORM
        OUT_NEUTRAL = float(ckpt.get("out_neutral", 0.0))
        WIN_LOOKBACK_DAYS = int(ckpt.get("win_lookback_days", WIN_LOOKBACK_DAYS))
        ZERO_0930 = bool(ckpt.get("zero_0930", False))
        PEER_FROM_DAILY = bool(ckpt.get("peer_from_daily", False))
        E04_CFG.update(pool="mean", xscale=False, mgate=False, mtl=False, peer_mode="attn", drop="", mtl_n=2)  # 旧模型没有这几项，按原结构恢复
        E04_CFG.update({k: v for k, v in ckpt["model_cfg"].items() if k in E04_CFG})
        global WIN_GROUPS, E04_DAILY_MIX
        WIN_GROUPS = ckpt.get("win_groups", "shape" if E04_CFG["n_win_feat"] == 7 else "")
        E04_DAILY_MIX = float(ckpt.get("daily_mix", 0.0))
        assert n_win_feat() == E04_CFG["n_win_feat"], (WIN_GROUPS, E04_CFG["n_win_feat"])
        return
    SUMMARY = bool(ckpt["model_cfg"].get("n_summary", 0))
    SUMMARY_SET = ckpt.get("summary_set", "all")
    SUMMARY_POOL = bool(ckpt["model_cfg"].get("summary_pool", False))
    OUT_NEUTRAL = float(ckpt.get("out_neutral", 0.0))
    SEQ_LEN = ckpt["model_cfg"]["seq_len"]
    COARSE_LEN = ckpt["model_cfg"].get("coarse_len", 0)
    NORM = ckpt.get("norm", "rel")


def load_stats(ckpt):
    if ckpt.get("arch") == "e04":
        return tuple(np.asarray(a, np.float32) for a in ckpt["stats"])
    return _load_stats_v1(ckpt)


def _load_stats_v1(ckpt):
    f = lambda k: np.asarray(ckpt[k], np.float32) if ckpt.get(k) is not None else None  # noqa: E731
    return f("mean"), f("std"), f("mean_c"), f("std_c")


# ==== 训练并持久化 ====
def _batches(n, sec_id, gen):
    """产出每批的样本下标：mse 为随机打乱的定长批；含 ic 时为随机顺序的若干完整截面。"""
    if LOSS == "mse":
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            yield perm[i:i + BATCH], None
        return
    order = np.argsort(sec_id, kind="stable")
    bounds = np.flatnonzero(np.diff(sec_id[order])) + 1
    groups = np.split(order, bounds)
    sp = torch.randperm(len(groups), generator=gen).numpy()
    for i in range(0, len(sp), SECS_PER_BATCH):
        gs = [groups[k] for k in sp[i:i + SECS_PER_BATCH]]
        seg = np.concatenate([np.full(len(g), q) for q, g in enumerate(gs)])
        yield torch.from_numpy(np.concatenate(gs)), torch.from_numpy(seg)


def _ic_loss(pred, y, seg):
    """各截面内 Pearson 相关的平均取负。"""
    k = int(seg.max()) + 1
    cnt = torch.zeros(k, device=pred.device).index_add_(0, seg, torch.ones_like(pred))
    pm = torch.zeros(k, device=pred.device).index_add_(0, seg, pred) / cnt
    ym = torch.zeros(k, device=pred.device).index_add_(0, seg, y) / cnt
    pc, yc = pred - pm[seg], y - ym[seg]
    cov = torch.zeros(k, device=pred.device).index_add_(0, seg, pc * yc)
    pv = torch.zeros(k, device=pred.device).index_add_(0, seg, pc * pc)
    yv = torch.zeros(k, device=pred.device).index_add_(0, seg, yc * yc)
    return -(cov / torch.sqrt(pv * yv + 1e-12)).mean()


def _train_one(X, y, seed, device, on_epoch_end=None, member=0, sec_id=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = StockTransformer(**MODEL_CFG).to(device)
    if WEIGHT_DECAY > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_batches = math.ceil(len(y) / BATCH) if LOSS == "mse" else math.ceil(len(np.unique(sec_id)) / SECS_PER_BATCH)
    steps = EPOCHS * n_batches
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, (lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))) if LR_SCHEDULE == "cos" else (lambda s: 1.0))
    loss_fn = nn.MSELoss()
    Xf, Xc = torch.from_numpy(X[0]), (torch.from_numpy(X[1]) if X[1] is not None else None)
    Xs = torch.from_numpy(X[2]) if len(X) > 2 and X[2] is not None else None
    yt = torch.from_numpy(y)
    gen = torch.Generator().manual_seed(seed)
    start_ep = 0
    if CKPT_DIR:                                                # 断点续训：读取该成员最新的 epoch 存档
        os.makedirs(CKPT_DIR, exist_ok=True)
        done = sorted(int(f.split("_ep")[1].split(".")[0]) for f in os.listdir(CKPT_DIR)
                      if f.startswith(f"m{member}_ep") and f.endswith(".pt"))
        if done:
            st = torch.load(os.path.join(CKPT_DIR, f"m{member}_ep{done[-1]}.pt"), map_location="cpu")
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            gen.set_state(st["gen"])
            start_ep = st["epoch"]
            logger.info("从断点恢复", member=member, epoch=start_ep)
    every = max(1, n_batches // max(PROGRESS_STEPS, 1))
    for ep in range(start_ep, EPOCHS):
        model.train()
        t, tot, nb, seen, win, wn = time.time(), 0.0, 0, 0, 0.0, 0
        for j, seg in _batches(len(yt), sec_id, gen):
            xb = Xf[j].to(device).float()
            xc = Xc[j].to(device).float() if Xc is not None else None
            xs = Xs[j].to(device).float() if Xs is not None else None
            yb = yt[j].to(device)
            opt.zero_grad()
            pred = model(xb, xc, xs)
            if LOSS == "mse":
                loss = loss_fn(pred, yb)
            elif LOSS == "ic":
                loss = _ic_loss(pred, yb, seg.to(device))
            else:
                loss = loss_fn(pred, yb) + _ic_loss(pred, yb, seg.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            lv = loss.item()
            tot += lv
            win += lv
            nb += 1
            wn += 1
            seen += len(j)
            if nb % every == 0 and nb < n_batches:
                el = time.time() - t
                logger.info("训练进度", member=member, epoch=f"{ep + 1}/{EPOCHS}", pct=f"{100 * nb / n_batches:.0f}%",
                            loss=round(win / wn, 6), samples_per_s=int(seen / el), eta_s=int(el / nb * (n_batches - nb)))
                win, wn = 0.0, 0
        logger.info("epoch 完成", member=member, epoch=f"{ep + 1}/{EPOCHS}", loss=round(tot / max(nb, 1), 6),
                    elapsed=round(time.time() - t, 1))
        if CKPT_DIR:                                            # 先写临时文件再改名，避免崩溃时留下半个存档
            path = os.path.join(CKPT_DIR, f"m{member}_ep{ep + 1}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "gen": gen.get_state(), "epoch": ep + 1}, path + ".tmp")
            os.replace(path + ".tmp", path)
        if on_epoch_end is not None:
            on_epoch_end(model, device, member, ep + 1)
    return model


def train_and_save(datasources, model_path=MODEL_PATH, backend=None, on_epoch_end=None, on_data_ready=None):
    """在写死的训练区间上从零训练 N_MODELS 个模型，把 权重 + 标准化统计 + 结构超参 一并存盘。

    backend：本地训练时传入数据后端；平台上留空，使用 dai。
    on_data_ready(stats) / on_epoch_end(model, device, member, epoch)：本地验证用回调，平台上留空。"""
    backend = backend or DaiBackend(datasources["bar1m"])
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if ARCH == "e04":
        return e04_train_and_save(backend, model_path, on_epoch_end, on_data_ready)
    device = pick_device()
    logger.info("训练设备", device=str(device))
    # 结构超参中与输入相关的部分按当前配置推导，保证与写死的常量一致
    MODEL_CFG.update(n_feat=N_FEAT, seq_len=SEQ_LEN, coarse_len=COARSE_LEN,
                     n_summary=len(summary_names()) if SUMMARY else 0, summary_pool=SUMMARY_POOL)

    instruments = backend.pool(TRAIN_START, TRAIN_END)
    if MAX_TRAIN_INSTRUMENTS:
        instruments = instruments[:MAX_TRAIN_INSTRUMENTS]
    X, y, keys, stats = build_dataset(backend, TRAIN_START, TRAIN_END, "train", instruments)
    y = make_target(y, keys)
    if LABEL_NEUTRAL > 0:                                       # 只用训练标签，去掉风格部分
        y = neutralize(y, keys[CHAR_COLS].to_numpy(), keys["date"].values, LABEL_NEUTRAL, rank_values=False)
        y = np.nan_to_num(y).astype(np.float32)
    sec_id = pd.factorize(keys["date"])[0].astype(np.int64)
    del keys
    if on_data_ready is not None:
        on_data_ready(stats)
    logger.info("训练样本", samples=len(y), models=N_MODELS,
                n_params=sum(p.numel() for p in StockTransformer(**MODEL_CFG).parameters()))

    models = [_train_one(X, y, SEED + k, device, on_epoch_end, k, sec_id) for k in range(N_MODELS)]
    mf, sf, mc, sc = stats
    tolist = lambda a: None if a is None else np.asarray(a, np.float32).tolist()  # noqa: E731
    save_model({
        "state_dicts": [m.state_dict() for m in models],
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
        "field_set": FIELD_SET,
        "seq_len": SEQ_LEN,
        "coarse_len": COARSE_LEN,
        "summary": summary_names() if SUMMARY else [],
        "summary_set": SUMMARY_SET,
        "label_neutral": LABEL_NEUTRAL,
        "out_neutral": OUT_NEUTRAL,
        "norm": NORM,
        "target": TARGET,
        "loss": LOSS,
        "train_range": [TRAIN_START, TRAIN_END],
        "data_alignment": DATA_ALIGNMENT,
        "mean": tolist(mf), "std": tolist(sf), "mean_c": tolist(mc), "std_c": tolist(sc),
    }, model_path)
    logger.info("模型已保存", path=model_path, bytes=os.path.getsize(model_path))
    return model_path


# ============================== E04：多尺度专家 + 同伴注意力 ==============================
# 三个尺度专家（短线：最近 64 根 1m；日内：最近 18 个 30 分钟窗口；日线：最近 20 个交易日）各自给出打分，
# 同伴编码器从最近 60 个交易日的日收益学出"走势画像"，同伴注意力层让每只股票参考同一截面里画像相似的股票，
# 只保留相对同伴的部分；推理时三个专家各自截面排名后等权平均，再做风格中性化。
# 输入只用原始量价与简单变换（相对当前价的对数价格、成交额相对均值、收益、逐截面标准化），不用不平衡类指标。
ARCH = "e04"  # v1（E01–E03 结构）/ e04
E04_CFG = dict(fuse_hidden=0, fuse_peer=False, tod_route='token', dropout=0.1, d_model=64, nhead=4, n_layers_short=2, n_layers_long=1, dim_ff=128, intra_len=18, daily_len=20, peer_len=60, peer_dim=32, n_win_feat=19, slot_len=0, patch_len=0, xs_layers=0, fuse=True, pool='mean', xscale=False, mgate=False, mtl=True, peer_mode='attn', drop='', mtl_n=2)
# peer_mode（10-21 简化消融）：attn = 同伴注意力（原结构）；mean = 减去同一截面的平均表示（不要同伴编码器）；none = 不做同伴对比
# drop（10-21 简化消融）：去掉的尺度专家（s = 1 分钟、i = 日内、d = 日线），其表示置零、头不参与打分
# mtl（10-20 R22）：多任务辅助头（融合输入 → 2 维：未来 30 分钟已实现波动、量能变化的截面秩目标），只在训练时计损失，推理不用
# mgate（10-20）：市场门控（MASTER 2024 的市场引导特征选择）：截面内各股最近一个 token 的均值作市况向量，
#   线性层 + 2·sigmoid 给三个专家的输入通道逐通道缩放（层初始为 0 → 门控恒为 1，起点等价于原结构）
# pool（10-20）：各专家 token 的池化方式。mean = 平均（原做法）；attn = 可学习查询的注意力池化（查询初始为 0，起点等价于平均）；
#   mix = [平均, 最近一个 token] 拼接后线性投影（显式保留最近一分钟 / 截面所在窗口 / 最近一日）
# xscale（10-20）：三个专家池化后的表示各做一次跨尺度注意力（同时看三个尺度的全部 token），残差 + LayerNorm，再进同伴比较与融合头
# dropout：各 Transformer 层与融合头的 dropout（R1e 加强正则时上调）
# tod_route（需 WIN_GROUPS 以 tod 结尾）：token = 时段 one-hot 拼在每个日内 / 日线 token 上（S21b）；fuse = 从 token 里拿掉，只拼到融合头输入；
#   film = 从 token 里拿掉，按时段对融合头隐层做逐通道缩放与平移（初始为恒等）。目的：保住时段带来的 IC，减少它带来的换手；reports/15 S22
# fuse_hidden > 0：融合头加宽加深（6d → fuse_hidden → d → 1）；fuse_peer：融合头的输入再拼上同伴画像编码（60 日走势）；reports/15 S17
# xs_layers > 0：用可学习的截面 Transformer（同一截面的股票互相注意，xs_layers 层）取代固定相似度的同伴注意力；reports/15 S12 的 A1
# fuse：再加一个跨尺度融合头（三个专家表示拼接），推理只用融合头；三个专家头保留作辅助监督；A2
# patch_len > 0（取 42）：日内专家再接当日开盘至截面的 5 分钟 patch token（09:30–11:30 共 24 个、13:00–14:30 共 18 个，
# 按时刻对齐，截面之后的置空），由 1 分钟 bar 聚合；reports/15 的 T2
# slot_len > 0：日内专家在 18 个窗口 token 之后再接 2·slot_len 个"同时段" token（过去 slot_len 个交易日里，
# 截面所在窗口与下一个窗口，通道定义与日内 token 完全相同），reports/15 的 T1
PEER_FROM_DAILY = False  # True：同伴日收益向量取自一次性构建的日线面板（可取 peer_len=240 等长窗口，不加大逐月窗口表的回看）
E04_IC_WEIGHT = 1.0            # 损失 = MSE + 权重 x (1 - 截面相关)
BOOST_PRED = ""                # R5d 两阶段提升：第一阶段模型在训练区间上的预测（parquet 通配，date／instrument／factor）；空 = 不用
BOOST_BETA = 0.0               # >0：训练目标 = 原目标 − β × 第一阶段原始预测（残差提升）；推理时最终分 = 第一阶段 + 第二阶段
SLOT_W = {}  # 按截面时刻的损失权重，如 {"14:30": 0.5}（各截面 IC 拉平以提高 IC_IR；R3）；与 YEAR_W 相乘后归一化到均值 1
DISP_W = 0.0                   # >0：按截面标签离散度加权，权重 = (截面标签标准差 ÷ 中位数)^DISP_W，截断 0.25～4（10-20：压力 = 高、低离散两半中较弱一半的 IC_IR，弱的是高离散半）
YEAR_W = {}  # 按年份的训练样本权重，如 {2020: 0.5, 2021: 0.83, ...}；截面权重归一化到均值 1；空 = 等权；reports/15 S18
E04_MTL_W = 0.5  # >0：多任务辅助损失权重（R22；需 E04_CFG["mtl"]）
E04_MTL_SET = "rv,lv"  # 多任务目标（10-21 R26）：rv 已实现波动、lv 量能变化、r60／r10 之后 60／10 分钟收益、hi／lo 之后 30 分钟最高／最低收盘相对现价、sv 之后 30 分钟签名成交量占比
E04_SAM_RHO = 0.0              # >0：SAM 锐度感知最小化的扰动半径（10-20 结构探索；每步两次前向反向）
E04_WEIGHT_EMA = 0.995  # >0：训练时维护权重的指数滑动平均（衰减系数），保存与推理用平均后的权重（降低单模型的种子噪声）；reports/15 S20
E04_DAILY_MIX = 0.0  # >0（仅融合模型）：推理最终分 = (1 − w)·截面秩(融合头) + w·截面秩(日线专家头)。日线专家只用截面之前的完整交易日，
#                                同一交易日内几乎不变，作"日内不变锚"稳定相邻截面的排序、压换手；只改推理，不改训练；reports/15 S23
E04_ANCHOR_TRAIN = 0.0  # >0（仅融合模型）：训练时的合成分 = (1 − w)·z(融合头) + w·z(日线专家头)，IC 损失直接作用在混合分上，
#                                让融合头学"日内变化的部分"、把稳定部分留给日线头；推理应同时设 E04_DAILY_MIX = w；reports/15 S24
E04_HEAD_W = 1.0  # 融合模型里三个专家头的损失权重（融合头为 1）；<1 让共享表示更偏向融合头；reports/15 S17
E04_TAIL_WEIGHT = 0.0          # >0：再加 权重 x (1 − 尾部加权相关)，权重 = |合成分的截面 z|（截断到 3，不回传梯度）；A4
E04_LISTNET_WEIGHT = 0.0       # >0：再加 权重 x 列表式排序损失（ListNet，头尾对称：KL(softmax(z(y)) ‖ softmax(合成分)) + 同样对 −y、−合成分）；R1a
WIN_LOOKBACK_DAYS = 130  # 30 分钟窗口表向前多取的自然日（60 + 20 个交易日 + 余量）
E04_BUFFER_DAYS = 7            # 1m 数据向前多取的自然日（64 根回看）
E04_EXPERTS = ["short", "intraday", "daily"]
# 训练时的标签与损失剔除（E07 起只用行情数据；端到端线上环境不再提供官方风格暴露表，2026-10-12 主办方通知）：
#   none      不剔除；
#   proxy     对行情代理特征（规模、规模平方、波动、beta、非流动性、20/120 日动量、相对放量、低价）逐截面回归取残差；
#   proxy_ind 再加伪行业哑变量（过去 PIND_DAYS 个交易日的超额日收益做球面 k-means 聚成 PIND_K 类，每月月初只用过去数据重算）。
# 损失里的 IC 也按同一剔除后的口径计算。推理不需要这些特征。
TRAIN_NEUTRAL = "proxy_ind"
PIND_K = 30
PIND_DAYS = 120
DAILY_LOOKBACK_DAYS = 400      # 日线面板向前多取的自然日（240 个交易日的相对放量 + 余量）
P_COLS = ["p_size", "p_size2", "p_vol", "p_beta", "p_illiq", "p_mom20", "p_mom120", "p_rvol", "p_lowp"]
ZERO_0930 = True  # True：推理时 09:30 截面输出常数 0（平台跳过常数截面；09:30 是最弱的截面）
# 训练目标与五项评分对齐（E07 S3）：
LOSS_SKIP_0930 = True  # True：09:30 截面不参与训练（推理时它输出常数，不计分）
DAY_BAG = 0.0                  # >0：每个成员随机去掉这一比例的交易日（按种子固定；R16 抽样集成）；0 = 用全部交易日
DAY_BATCH = 0  # >0：每批取若干个完整交易日（按时间顺序的全部截面），以便计算相邻截面的换手惩罚；0 = 随机截面
PAIR_UNITS = False  # 10-25：每个训练单元 = 同一交易日两个相邻截面（按 epoch 交替配对），批内仍跨日随机，用于腿换手损失
E04_LEG_TURN = 0.0  # 10-25：腿换手损失权重：风格残差后的合成分在相邻截面的"软腿成员"（前 10%／后 10%，sigmoid）之差的平方，只作用在腿边界附近
E04_LEG_TAU = 0.15  # 软腿成员的温度（标准化合成分单位）
TURN_WEIGHT = 0.0  # 换手惩罚：同一交易日相邻截面、同一股票的合成得分（截面标准化后）之差的平方均值
ICIR_WEIGHT = 0.0  # IC 稳定性：一批内各截面 IC 的标准差（对应 IC_IR 与压力）
TARGET_AUX_W = 0.5  # 多周期目标（只在训练时）：目标 = z(本截面标签) + 权重 x z(同一交易日之后 TARGET_AUX_N 个截面标签之和)
TARGET_AUX_N = 2
EXTRA_TIMES = ['09:45', '10:15', '10:45', '11:15', '13:15', '13:45', '14:15']  # 只在训练时加入的中间时刻截面（如 ["09:45", "10:15", ...]），标签按官方口径由 1 分钟 bar 计算；推理不变；reports/15 S19
GBDT_CK = None                 # 推理时由 configure_from_ckpt 从权重文件读入的树模型成员
E04_GBDT = {'bags': 1, 'trees': 800, 'depth': 8, 'lr': 0.03, 'rows': 0.5, 'cols': 0.7, 'min_leaf': 1000, 'lam': 1.0, 'bins': 64, 'fields': 'full', 'norm': 'rel3', 'rich': True, 'weight': 1.0, 'win': 'shape,book,book2,book3,tod,pos,state,cls,mkt,lead,dow,auc'}  # 10-25：集成里的树模型成员（空 = 不用）。键见 _GB_DEFAULT；树在三档全字段 + rel3 的输入上训练，与 Transformer 同一目标
E04_EXTRA_MEMBERS = [2, 3]  # 10-23 混合集成：非空时只有这些成员（序号 k，种子 SEED + k）用中间时刻截面，其余成员训练时跳过它们（同一份数据，不重复构建）
TARGET_AUX_DECAY = 1.0  # 第 k 个之后截面的权重 = DECAY^(k-1)；1.0 = 等权求和（E09 及以前）
# 日内 / 日线专家的输入通道（E08 S7 起按组叠加；基础 4 通道 = 相对当前价的对数价格、对数 VWAP、相对成交额、收益）：
#   shape  上影线占比、实体占比、平均单笔成交额（3）
#   book   窗口内 1 分钟盘口的均值：1–3 档挂单量不平衡、相对价差、一档挂单量占 1–3 档的比例、1–3 档委托笔数不平衡（4）
#   state  股票状态，每个 token 相同：价位、20 日均成交额、20 日日收益波动、20 日非流动性、20 日平均单笔成交额，逐截面排名（5）
#   mkt    市场状态，同一截面相同：当日至今收益、最近一个窗口收益各自的截面离散度（对数）（2）
WIN_GROUPS = "shape,book,tod"  #   pos    价格位置（6）：每个日内 / 日线 token 加窗口（当日）最高、最低价相对当前价（2）；另加每个 token 相同、逐截面排名的
#          当日至今最低价、最高价、累计 VWAP 相对当前价，与 3 日区间位置（4）。对应因子赛道 F 方向（1133、1200）
#   slot   同时段异常量（1）：窗口成交额 ÷ 同一时段过去 20 个交易日均值（只用之前的日子）；日线 token 用日成交额 ÷ 之前 20 日均值。对应 C 方向
#   lead   领先滞后（2）：全市场成交额前 20% 的股票上一窗口 / 上两窗口的平均收益 − 本股，同一截面计算。对应 B 方向
#   xrank  （放在最后）对日内 / 日线 token 的全部通道再给一份逐截面排名（同一 token 位置跨股票，−0.5～0.5），通道数翻倍；reports/15 的 T5
#   peer   同伴走势（4，放在所有通道最后）：同一截面里按过去 60 日收益画像最相似的 20 只股票，在同一个窗口 / 交易日上
#          前 4 个通道（收盘与 VWAP 相对各自当前价、相对成交额、收益）的平均；同时段与 patch token 置空。不可与 xrank 同用；P1
#   tod    当前时段 one-hot（09:30 … 14:30 共 8 个，每个 token 相同）；reports/15 S12 的 A3
#   book3  微观价格偏离、深度 ÷ 成交量（2，reports/15 T4）
#   book2  盘口细化（4）：一档挂单量不平衡、窗口内 1–3 档不平衡的变化（末减初）、相对价差在窗口内的波动、每笔委托平均挂单量（对数）
#   cls    尾盘段习惯（1，reports/15 S25）：日内 token 用该窗口 收盘÷VWAP（对数），日线 token 用当日最后一个窗口（14:30–15:00，收盘集合竞价收尾）的 收盘÷VWAP
WIN_GROUP_DIM = {"shape": 3, "book": 4, "book2": 4, "book3": 2, "state": 5, "mkt": 2, "pos": 6, "slot": 1, "lead": 2, "xrank": 0, "tod": 8, "peer": 4, "cls": 1, "dow": 5,
                 "auc": 2}
BOOK_LOOKBACK_DAYS = 45        # 盘口窗口表向前多取的自然日（20 个交易日 + 余量；只有日内 / 日线 token 用到）


def win_groups():
    return [g for g in WIN_GROUPS.split(",") if g]


def n_win_feat():
    n = 4 + sum(WIN_GROUP_DIM[g] for g in win_groups() if g != "xrank")
    return 2 * n if "xrank" in win_groups() else n


def _segments(dates):
    """按日期把一组截面时间切成连续段，返回各段的 (起点, 终点)。dates 需已按时间排序。"""
    d = np.asarray(dates)
    cut = np.flatnonzero(d[1:] != d[:-1]) + 1
    st = np.r_[0, cut]
    return list(zip(st, np.r_[cut, len(d)]))


def daily_z(backend, sd, ed, instruments):
    """同伴用的日线截面标准化日收益矩阵（股票 x 交易日），只在 PEER_FROM_DAILY 时构建，覆盖 [sd - 回看, ed]。"""
    lb = int(E04_CFG["peer_len"] * 1.6) + 30
    dp = daily_panel(backend, _fmt(pd.Timestamp(sd) - pd.Timedelta(days=lb)), _fmt(ed), instruments)
    close = dp.pivot(index="day", columns="instrument", values="close").sort_index()
    pre = dp.pivot(index="day", columns="instrument", values="pre").reindex_like(close)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log(close / pre).where(lambda x: np.isfinite(x) & (x.abs() < 0.5))
    z = r.sub(r.mean(1), axis=0).div(r.std(1) + 1e-12, axis=0)
    return {"days": z.index.values, "inst": z.columns.values.astype(str), "z": z.to_numpy(np.float32).T}


def _patch_cube(bars, keys, P=42):
    """T2：当日开盘至截面的 5 分钟 patch（与 keys 行对齐），通道 = 收盘相对截面价（对数）、VWAP 相对最后一个 patch 的 VWAP、
    成交额 ÷ 当日已有 patch 均值（log1p）、patch 收益、上影线占比、实体占比；截面之后的 patch 为 NaN。
    只用比值，与价格 / 量的计量单位无关；只用截面时刻（含）之前的 1 分钟 bar。"""
    b = bars[bars["instrument"].isin(set(keys["instrument"]))][["date", "instrument", "open", "high", "low", "close", "volume", "amount"]]
    hm = b["date"].dt.hour.to_numpy() * 100 + b["date"].dt.minute.to_numpy()
    mins = np.where(hm <= 1130, (hm // 100) * 60 + hm % 100 - 570, 120 + (hm // 100) * 60 + hm % 100 - 780)
    keep = (((hm > 930) & (hm <= 1130)) | ((hm > 1300) & (hm <= 1430))) & (b["close"].to_numpy() > 0)
    b = b[keep].assign(k=(mins[keep] - 1) // 5, day=b["date"][keep].dt.normalize()).sort_values(["instrument", "date"])
    pt = b.groupby(["instrument", "day", "k"], sort=True).agg(o=("open", "first"), h=("high", "max"), l=("low", "min"),
                                                               c=("close", "last"), v=("volume", "sum"), a=("amount", "sum")).reset_index()
    pairs = pt[["instrument", "day"]].drop_duplicates().reset_index(drop=True)
    pairs["pid"] = np.arange(len(pairs))
    pt = pt.merge(pairs, on=["instrument", "day"])
    cube = np.full((len(pairs), P, 6), np.nan)
    kk = pt["k"].to_numpy()
    okk = kk < P
    cube[pt["pid"].to_numpy()[okk], kk[okk]] = pt[["o", "h", "l", "c", "v", "a"]].to_numpy(np.float64)[okk]
    kd = pd.DataFrame({"instrument": keys["instrument"].values, "day": pd.DatetimeIndex(keys["date"].values).normalize()})
    pid = kd.merge(pairs, on=["instrument", "day"], how="left")["pid"].to_numpy()
    th = pd.DatetimeIndex(keys["date"].values)
    tm = th.hour * 100 + th.minute
    kmax = np.where(tm <= 1130, ((tm // 100) * 60 + tm % 100 - 570) // 5, 24 + ((tm // 100) * 60 + tm % 100 - 780) // 5)
    kmax = np.clip(np.where(tm <= 930, 0, kmax), 0, P)                         # 有效 patch 个数
    N = len(keys)
    X = np.full((N, P, 6), np.nan)
    okp = np.isfinite(pid)
    X[okp] = cube[pid[okp].astype(np.int64)]
    X[np.arange(P)[None, :] >= kmax[:, None]] = np.nan
    o, h, l_, c, v, a = (X[..., j] for j in range(6))
    with np.errstate(invalid="ignore", divide="ignore"):
        lastk = np.clip(kmax - 1, 0, P - 1)
        ref_c = c[np.arange(N), lastk]
        vw = np.where(v > 0, a / np.where(v > 0, v, 1), np.nan)
        ref_v = vw[np.arange(N), lastk]
        prev = np.concatenate([o[:, :1], c[:, :-1]], 1)
        prev = np.where(np.isfinite(prev), prev, o)
        rng = h - l_
        okr = np.isfinite(rng) & (rng > 0)
        ch = [np.log(c / ref_c[:, None]), np.log(vw / ref_v[:, None]),
              np.log1p(a / (np.nanmean(a, 1, keepdims=True) + 1e-9)), np.log(c / prev),
              np.where(okr, (h - np.fmax(o, c)) / np.where(okr, rng, 1), np.where(np.isfinite(c), 0.0, np.nan)),
              np.where(okr, np.abs(c - o) / np.where(okr, rng, 1), np.where(np.isfinite(c), 0.0, np.nan))]
    out = np.stack(ch, -1)
    return np.where(np.isfinite(out), out, np.nan)


def _e04_window_feats(win, keys, dz=None, patches=None):
    """由 30 分钟窗口表构造日内 / 日线 token、同伴日收益向量与四个风格特征。只用截面时刻及之前的窗口。
    复权：用 昨收 / 前一交易日收盘 得到除权缺口，把历史价格换算到当前交易日口径。"""
    inst = np.sort(keys["instrument"].unique())
    w = win[win["instrument"].isin(inst)]
    times = np.sort(w["date"].unique())
    ti = np.searchsorted(times, w["date"].values)
    ii = np.searchsorted(inst, w["instrument"].values)
    I, T = len(inst), len(times)

    def cube(col):
        a = np.full((I, T), np.nan)
        a[ii, ti] = w[col].to_numpy(np.float64)
        return a
    px_raw, amt, vol, pc = cube("last_px"), cube("sum_amt"), cube("sum_vol"), cube("pre_close")
    grp = win_groups()
    ext = "shape" in grp                                                        # 扩展通道：上影线占比、实体占比、平均单笔成交额
    if ext or "state" in grp or "pos" in grp:
        wo, wh, wl, wc, ra, wd = (cube(c) for c in ("w_open", "w_high", "w_low", "w_close", "raw_amt", "w_deals"))
        auc = pd.DatetimeIndex(times).strftime("%H%M") == "0930"                 # 集合竞价窗口：1 分钟竞价 bar 的开高低不可靠
        for a_ in (wo, wh, wl, wc):                                             # （平台 SQL 与本地 30 分钟数据在此处不一致），形态不用它
            a_[:, auc] = np.nan
    px = _ffill_t(px_raw)
    day = pd.DatetimeIndex(times).normalize().values
    d_start = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
    d_end = np.r_[d_start[1:], T] - 1
    D = len(d_start)
    dot = np.searchsorted(d_start, np.arange(T), side="right") - 1             # 每个窗口所属交易日序号
    close_d = px[:, d_end]
    amt0 = np.nan_to_num(amt)
    amt_d = np.add.reduceat(amt0, d_start, axis=1)
    vol_d = np.add.reduceat(np.nan_to_num(vol), d_start, axis=1)
    pre_d = np.fmax.reduceat(pc, d_start, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = pre_d[:, 1:] / close_d[:, :-1]
        lg = np.where(np.isfinite(gap) & (gap > 0), np.log(gap), 0.0)
        G = np.concatenate([np.zeros((I, 1)), np.cumsum(lg, 1)], 1)             # 累计除权缺口（对数）
        L = np.log(px) - G[:, dot]                                              # 复权后的对数价格
        Lv = np.log(np.where(vol > 0, amt / np.where(vol > 0, vol, 1), np.nan)) - G[:, dot]
        Ld = np.log(close_d) - G
        Lvd = np.log(np.where(vol_d > 0, amt_d / np.where(vol_d > 0, vol_d, 1), np.nan)) - G
        r_w = np.diff(L, axis=1, prepend=np.nan)
        r_d = np.log(close_d / pre_d)
        if "cls" in grp:                                                        # S25：窗口收盘相对窗口 VWAP；日线取当日最后一个窗口（尾盘段）
            lab_w = L - Lv
            cls_d = lab_w[:, d_end]
        zr = (r_d - np.nanmean(r_d, 0, keepdims=True)) / (np.nanstd(r_d, 0, keepdims=True) + 1e-12)
        if ext:
            def shape3(o, h, l_, c, a, n):                                      # 上影线占比、实体占比、ln 平均单笔成交额
                rng = h - l_
                ok = np.isfinite(rng) & (rng > 0)
                us = np.where(ok, (h - np.fmax(o, c)) / np.where(ok, rng, 1), 0.0)
                bd = np.where(ok, np.abs(c - o) / np.where(ok, rng, 1), 0.0)
                at = np.log(np.where((n > 0) & (a > 0), a / np.where(n > 0, n, 1), np.nan))
                return us, bd, at
            us_w, bd_w, at_w = shape3(wo, wh, wl, wc, ra, wd)
            o_d = wo[:, np.minimum(d_start + 1, d_end)]                         # 当日开盘 = 第一个连续竞价窗口的开盘
            c_d = np.take_along_axis(_ffill_t(wc), np.broadcast_to(d_end, (I, D)), 1)
            us_d, bd_d, at_d = shape3(o_d, np.fmax.reduceat(wh, d_start, axis=1), np.fmin.reduceat(wl, d_start, axis=1), c_d,
                                      np.add.reduceat(np.nan_to_num(ra), d_start, axis=1),
                                      np.add.reduceat(np.nan_to_num(wd), d_start, axis=1))

        if "pos" in grp:                                                        # 价格位置：复权对数最高 / 最低价，当日累计量
            Lh = np.log(np.where(wh > 0, wh, np.nan)) - G[:, dot]
            Ll = np.log(np.where(wl > 0, wl, np.nan)) - G[:, dot]
            Lhd = np.log(np.where(np.fmax.reduceat(wh, d_start, axis=1) > 0, np.fmax.reduceat(wh, d_start, axis=1), np.nan)) - G
            Lld = np.log(np.where(np.fmin.reduceat(wl, d_start, axis=1) > 0, np.fmin.reduceat(wl, d_start, axis=1), np.nan)) - G
            cmin, cmax = np.full_like(Ll, np.nan), np.full_like(Lh, np.nan)
            camt, cvol = np.zeros_like(amt0), np.zeros_like(amt0)
            vol0 = np.nan_to_num(vol)
            for s_, e_ in zip(d_start, d_end):                                  # 当日至今（含截面所在窗口）
                cmin[:, s_:e_ + 1] = np.fmin.accumulate(Ll[:, s_:e_ + 1], axis=1)
                cmax[:, s_:e_ + 1] = np.fmax.accumulate(Lh[:, s_:e_ + 1], axis=1)
                camt[:, s_:e_ + 1] = np.cumsum(amt0[:, s_:e_ + 1], axis=1)
                cvol[:, s_:e_ + 1] = np.cumsum(vol0[:, s_:e_ + 1], axis=1)
            Lcv = np.log(np.where(cvol > 0, camt / np.where(cvol > 0, cvol, 1), np.nan)) - G[:, dot]
        if "slot" in grp:                                                       # 同时段：同一 HH:MM 的窗口按日排列，取之前 20 日均值
            def prev_mean_ratio(a, n=20, min_n=5):
                f = np.isfinite(a) & (a > 0)
                cs = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(np.where(f, a, 0.0), 1)], 1)
                cn = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(f.astype(np.float64), 1)], 1)
                k = np.arange(a.shape[1])
                lo_ = np.maximum(k - n, 0)
                sm, cnt = cs[:, k] - cs[:, lo_], cn[:, k] - cn[:, lo_]           # 严格早于当前的 n 个
                mu = np.where(cnt >= min_n, sm / np.maximum(cnt, 1), np.nan)
                return np.log1p(np.where(f, a, 0.0) / mu)
            hm = pd.DatetimeIndex(times).strftime("%H%M").values
            slot_w = np.full_like(amt0, np.nan)
            for h_ in np.unique(hm):
                cols = np.flatnonzero(hm == h_)
                slot_w[:, cols] = prev_mean_ratio(amt[:, cols])
            slot_d = prev_mean_ratio(amt_d)

    si = np.searchsorted(inst, keys["instrument"].values)
    tt = np.searchsorted(times, keys["date"].values, side="right") - 1          # 截面时刻（含）之前的最后一个窗口
    if len(tt) and ((tt < 0) | (times[np.maximum(tt, 0)] > keys["date"].values)).any():
        raise RuntimeError("特征时点泄漏 (30m 窗口)")
    Dt = dot[np.maximum(tt, 0)]
    Lt = L[si, np.maximum(tt, 0)]

    def gather(a, idx):                                                         # a (I, X)，idx (N, K)，越界记 NaN
        ok = (idx >= 0) & (idx < a.shape[1])
        return np.where(ok, a[si[:, None], np.clip(idx, 0, a.shape[1] - 1)], np.nan)
    li, ld, lp = E04_CFG["intra_len"], E04_CFG["daily_len"], E04_CFG["peer_len"]
    iw = tt[:, None] - li + 1 + np.arange(li)[None, :]
    a_i = gather(amt0, iw)
    with np.errstate(invalid="ignore", divide="ignore"):
        Xi = [gather(L, iw) - Lt[:, None], gather(Lv, iw) - Lt[:, None],
              np.log1p(a_i / (np.nanmean(a_i, 1, keepdims=True) + 1e-9)), gather(r_w, iw)]
        if ext:
            t_i = gather(at_w, iw)
            Xi += [gather(us_w, iw), gather(bd_w, iw), t_i - np.nanmean(t_i, 1, keepdims=True)]
        if "pos" in grp:
            Xi += [gather(Lh, iw) - Lt[:, None], gather(Ll, iw) - Lt[:, None]]
        if "slot" in grp:
            Xi += [gather(slot_w, iw)]
        if "cls" in grp:
            Xi += [gather(lab_w, iw)]
        Xi = np.stack(Xi, -1)
        idd = Dt[:, None] - ld + np.arange(ld)[None, :]                         # 截面所在交易日之前的完整交易日
        a_d = gather(amt_d, idd)
        Xd = [gather(Ld, idd) - Lt[:, None], gather(Lvd, idd) - Lt[:, None],
              np.log1p(a_d / (np.nanmean(a_d, 1, keepdims=True) + 1e-9)), gather(r_d, idd)]
        if ext:
            t_d = gather(at_d, idd)
            Xd += [gather(us_d, idd), gather(bd_d, idd), t_d - np.nanmean(t_d, 1, keepdims=True)]
        if "pos" in grp:
            Xd += [gather(Lhd, idd) - Lt[:, None], gather(Lld, idd) - Lt[:, None]]
        if "slot" in grp:
            Xd += [gather(slot_d, idd)]
        if "cls" in grp:
            Xd += [gather(cls_d, idd)]
        Xd = np.stack(Xd, -1)
        bcols = (BOOK30_COLS if "book" in grp else []) + (BOOK2_COLS if "book2" in grp else []) + (BOOK3_COLS if "book3" in grp else [])
        bcube = {}
        if bcols:                                                               # 盘口：窗口均值；日线 token 用当日各窗口的均值
            bi, bdy = [], []
            for c in bcols:
                b = cube(c)
                bcube[c] = b
                if c == "b_vpo":
                    b = np.log(np.where(b > 0, b, np.nan))
                fin = np.isfinite(b)
                s_ = np.add.reduceat(np.where(fin, b, 0.0), d_start, axis=1)
                n_ = np.add.reduceat(fin.astype(np.float64), d_start, axis=1)
                bi.append(gather(b, iw))
                bdy.append(gather(np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan), idd))
            Xi = np.concatenate([Xi, np.stack(bi, -1)], -1)
            Xd = np.concatenate([Xd, np.stack(bdy, -1)], -1)
        if "auc" in grp:                                                        # 10-23：收盘集合竞价——ln(15:00 竞价价 ÷ 14:56 收盘)、竞价成交量占全天比例
            c56, c00, v00, vdy = cube("a_c1456"), cube("a_c1500"), cube("a_v1500"), cube("a_vday")
            with np.errstate(invalid="ignore", divide="ignore"):
                a_r = np.where((c56 > 0) & (c00 > 0), np.log(np.where(c00 > 0, c00, 1.0) / np.where(c56 > 0, c56, 1.0)), np.nan)
                a_s = np.where(vdy > 0, v00 / np.where(vdy > 0, vdy, 1.0), np.nan)

            def _dayv(a):                                                       # 每个交易日一个值（只有当天 15:00 窗口有值）
                fin = np.isfinite(a)
                s_ = np.add.reduceat(np.where(fin, a, 0.0), d_start, axis=1)
                n_ = np.add.reduceat(fin.astype(np.float64), d_start, axis=1)
                return np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)
            ar_d, as_d = _dayv(a_r), _dayv(a_s)
            pdi = dot - 1                                                       # 日内 token：所在窗口前一交易日的竞价（当天的竞价在收盘后，不能用）
            okp = (pdi >= 0)[None, :]
            ar_w = np.where(okp, ar_d[:, np.maximum(pdi, 0)], np.nan)
            as_w = np.where(okp, as_d[:, np.maximum(pdi, 0)], np.nan)
            Xi = np.concatenate([Xi, np.stack([gather(ar_w, iw), gather(as_w, iw)], -1)], -1)
            Xd = np.concatenate([Xd, np.stack([gather(ar_d, idd), gather(as_d, idd)], -1)], -1)   # 日线 token：截面所在交易日之前的日子
        sl = E04_CFG.get("slot_len", 0)
        if sl:                                                                   # T1：同时段 token（只用截面所在交易日之前的日子）
            hm = pd.DatetimeIndex(times).strftime("%H%M").values
            su = np.unique(hm)                                                   # 当日窗口标签按时间排序（0930…1500）
            sid = np.searchsorted(su, hm)
            Mi = np.full((D, len(su)), -1, dtype=np.int64)
            Mi[dot, sid] = np.arange(T)
            cs = sid[np.maximum(tt, 0)]                                          # 截面所在窗口的时段
            ns = np.where(cs + 1 < len(su), cs + 1, -1)                         # 下一个窗口的时段（1500 之后没有）
            days = Dt[:, None] - sl + np.arange(sl)[None, :]                     # 之前 sl 个交易日（旧 → 新）
            okd = days >= 0
            dd = np.clip(days, 0, D - 1)
            ic = np.where(okd, Mi[dd, cs[:, None]], -1)
            inx = np.where(okd & (ns[:, None] >= 0), Mi[dd, np.maximum(ns, 0)[:, None]], -1)
            iws = np.stack([ic, inx], 2).reshape(len(tt), 2 * sl)              # 交替：同时段、下一时段
            iws = np.where(iws >= 0, iws, -10 ** 9)                              # 缺失 → 越界 → NaN
            a_s = gather(amt0, iws)
            Xs = [gather(L, iws) - Lt[:, None], gather(Lv, iws) - Lt[:, None],
                  np.log1p(a_s / (np.nanmean(a_s, 1, keepdims=True) + 1e-9)), gather(r_w, iws)]
            if ext:
                t_s = gather(at_w, iws)
                Xs += [gather(us_w, iws), gather(bd_w, iws), t_s - np.nanmean(t_s, 1, keepdims=True)]
            if "pos" in grp:
                Xs += [gather(Lh, iws) - Lt[:, None], gather(Ll, iws) - Lt[:, None]]
            if "slot" in grp:
                Xs += [gather(slot_w, iws)]
            for c in bcols:
                b = bcube[c]
                Xs.append(gather(np.log(np.where(b > 0, b, np.nan)) if c == "b_vpo" else b, iws))
            if "auc" in grp:                                                     # 竞价通道在同时段 token 上留空（对齐通道数）
                Xs += [np.full(iws.shape, np.nan), np.full(iws.shape, np.nan)]
            Xi = np.concatenate([Xi, np.stack(Xs, -1)], 1)
        if patches is not None:                                                  # T2：5 分钟 patch，通道对齐日内 token（缺的通道置空）
            nb = Xi.shape[2]
            pt_ = [patches[..., 0], patches[..., 1], patches[..., 2], patches[..., 3]]
            if ext:
                pt_ += [patches[..., 4], patches[..., 5], np.full(patches.shape[:2], np.nan)]
            pt_ = np.stack(pt_, -1)
            pt_ = np.concatenate([pt_, np.full(pt_.shape[:2] + (nb - pt_.shape[2],), np.nan)], -1)
            Xi = np.concatenate([Xi, pt_], 1)
        extra = []
        sec_t = keys["date"].values
        if "state" in grp:                                                      # 股票状态：截面所在交易日之前 20 个完整交易日
            ra_d = np.add.reduceat(np.nan_to_num(ra), d_start, axis=1)
            wd_d = np.add.reduceat(np.nan_to_num(wd), d_start, axis=1)
            g_c, g_a, g_r = gather(close_d, idd), gather(amt_d, idd), gather(r_d, idd)
            g_ra, g_wd = gather(ra_d, idd), gather(wd_d, idd)
            st = np.stack([np.log(g_c[:, -1]),                                  # 价位（前一交易日收盘，未复权）
                           np.log(np.nanmean(g_a, 1)),                          # 20 日均成交额
                           np.nanstd(g_r, 1),                                   # 20 日日收益波动
                           np.log(np.nanmean(np.abs(g_r) / np.where(g_a > 0, g_a, np.nan), 1)),   # 20 日非流动性
                           np.log(np.nansum(g_ra, 1) / np.nansum(g_wd, 1))], 1)   # 20 日平均单笔成交额
            st = np.where(np.isfinite(st), st, np.nan)
            extra.append(pd.DataFrame(st).groupby(sec_t).rank(pct=True).to_numpy() - 0.5)
        if "mkt" in grp:                                                        # 市场状态：截面离散度
            rd_now = Lt - gather(Ld, (Dt - 1)[:, None])[:, 0]
            r_last = r_w[si, np.maximum(tt, 0)]
            mk = pd.DataFrame({"a": rd_now, "b": r_last}).groupby(sec_t).transform("std").to_numpy()
            extra.append(np.log(mk + 1e-6))
        t0_ = np.maximum(tt, 0)
        if "pos" in grp:                                                        # 当日至今最低 / 最高 / 累计 VWAP 与 3 日区间位置
            lo_t, hi_t = cmin[si, t0_], cmax[si, t0_]
            d1 = (Dt - 1)[:, None]
            lo3 = np.fmin(lo_t, np.fmin(gather(Lld, d1)[:, 0], gather(Lld, d1 - 1)[:, 0]))
            hi3 = np.fmax(hi_t, np.fmax(gather(Lhd, d1)[:, 0], gather(Lhd, d1 - 1)[:, 0]))
            rg = hi3 - lo3
            ps = np.stack([Lt - lo_t, hi_t - Lt, Lt - Lcv[si, t0_],
                           np.where(rg > 0, (Lt - lo3) / np.where(rg > 0, rg, 1), np.nan)], 1)
            ps = np.where(np.isfinite(ps), ps, np.nan)
            extra.append(pd.DataFrame(ps).groupby(sec_t).rank(pct=True).to_numpy() - 0.5)
        if "lead" in grp:                                                       # 全市场成交额前 20%（前 20 个完整交易日均值）
            sz = np.log(np.nanmean(gather(amt_d, idd), 1))
            r1 = r_w[si, t0_]
            r2 = L[si, t0_] - L[si, np.maximum(t0_ - 2, 0)]
            df_ = pd.DataFrame({"t": sec_t, "s": sz, "r1": r1, "r2": r2})
            top = df_.groupby("t")["s"].rank(pct=True).to_numpy() >= 0.8
            ld_ = [df_[c].where(top).groupby(df_["t"]).transform("mean").to_numpy() - df_[c].to_numpy() for c in ("r1", "r2")]
            extra.append(np.stack(ld_, 1))
        if "dow" in grp:                                                        # R1d：星期几（截面所在日期，周一～周五 one-hot；放在 tod 之前，保证 tod 仍是最后 8 个通道）
            dw = pd.DatetimeIndex(sec_t).dayofweek.values
            extra.append(np.eye(5)[np.clip(dw, 0, 4)])
        if "tod" in grp:                                                        # A3：当前时段
            th = pd.DatetimeIndex(sec_t)
            tod = np.searchsorted([930, 1000, 1030, 1100, 1130, 1330, 1400, 1430], th.hour * 100 + th.minute)
            extra.append(np.eye(8)[np.clip(tod, 0, 7)])
        if "xrank" in grp:                                                      # T5：逐截面排名（与已有通道并列）
            def xr(X):
                n_, k_, f_ = X.shape
                r = pd.DataFrame(X.reshape(n_, k_ * f_)).groupby(sec_t).rank(pct=True).to_numpy() - 0.5
                return np.concatenate([X, r.reshape(n_, k_, f_)], -1)
            if extra:                                                           # 广播通道先拼上，再一起排名
                E_ = np.concatenate(extra, 1)[:, None, :]
                Xi = np.concatenate([Xi, np.broadcast_to(E_, Xi.shape[:2] + E_.shape[2:])], -1)
                Xd = np.concatenate([Xd, np.broadcast_to(E_, Xd.shape[:2] + E_.shape[2:])], -1)
                extra = []
            Xi, Xd = xr(Xi), xr(Xd)
        if extra:                                                               # 每个 token 相同，拼在日内 / 日线 token 的通道上
            E_ = np.concatenate(extra, 1)[:, None, :]
            Xi = np.concatenate([Xi, np.broadcast_to(E_, Xi.shape[:2] + E_.shape[2:])], -1)
            Xd = np.concatenate([Xd, np.broadcast_to(E_, Xd.shape[:2] + E_.shape[2:])], -1)
        if dz is None:
            Xp = np.nan_to_num(gather(zr, Dt[:, None] - lp + np.arange(lp)[None, :]), nan=0.0)
        else:                                                                   # 日线面板：截面所在交易日之前的 lp 个交易日
            kday = pd.DatetimeIndex(keys["date"].values).normalize().values
            di = np.searchsorted(dz["days"], kday, side="left")                 # 第一个不早于当天的位置 = 之前的日子数
            pi = np.searchsorted(dz["inst"], keys["instrument"].values.astype(str))
            pin = (pi < len(dz["inst"])) & (dz["inst"][np.minimum(pi, len(dz["inst"]) - 1)] == keys["instrument"].values.astype(str))
            cols = di[:, None] - lp + np.arange(lp)[None, :]
            okc = (cols >= 0) & pin[:, None]
            Xp = np.where(okc, dz["z"][np.minimum(pi, len(dz["inst"]) - 1)[:, None], np.clip(cols, 0, dz["z"].shape[1] - 1)], 0.0)
            Xp = np.nan_to_num(Xp, nan=0.0)
        # 四个风格特征（与因子赛道 neutral4 同口径：近 40 个窗口）
        m = np.where(np.isfinite(r_w), np.nanmean(r_w, 0, keepdims=True) * np.ones_like(r_w), np.nan)
        sx, n = _roll(r_w)
        sy, _ = _roll(m)
        sxy, _ = _roll(r_w * m)
        syy, _ = _roll(m * m)
        ex, ey = sx / n, sy / n
        var_m = syy / n - ey ** 2
        beta = np.where(var_m > 1e-12, (sxy / n - ex * ey) / np.where(var_m > 1e-12, var_m, 1), np.nan)
        beta = np.where(np.isfinite(beta), np.clip(beta, -1, 4), 1.0)
        res = r_w - beta * m
        s1, nr = _roll(res)
        s2, _ = _roll(res * res)
        ivol = np.sqrt(np.maximum((s2 - s1 ** 2 / nr) / np.maximum(nr - 1, 1), 0))
        il = np.abs(r_w) / np.where(amt > 0, amt / 1e8, np.nan)
        sil, nil = _roll(il)
        sa, na = _roll(np.where(np.isfinite(amt), amt, np.nan))
        t0 = np.maximum(tt, 0)
        ch = np.stack([-np.log1p((sa / na * na)[si, t0]), -ivol[si, t0], (sil / nil)[si, t0],
                       -np.log(_ffill_t(px_raw)[si, t0])], 1)
    if "peer" in grp:                                                           # P1：同伴走势通道（按截面找最相似的 20 只）
        assert "xrank" not in grp, "peer 与 xrank 不能同用"
        K = 20
        P = Xp - Xp.mean(1, keepdims=True)
        P = P / (np.sqrt((P * P).sum(1, keepdims=True)) + 1e-9)
        pi_ = np.full((len(P), K), -1, dtype=np.int64)
        for _, ii in pd.Series(np.arange(len(P))).groupby(keys["date"].values).indices.items():
            if len(ii) <= K:
                continue
            S_ = P[ii] @ P[ii].T
            np.fill_diagonal(S_, -np.inf)
            nb = np.argpartition(-S_, K, axis=1)[:, :K]
            pi_[ii] = ii[nb]
        okp = (pi_ >= 0).all(1)
        def peer_avg(X, ntok):
            out = np.full((len(X), X.shape[1], 4), np.nan)
            src = X[:, :ntok, :4]
            with np.errstate(invalid="ignore"):
                g = src[np.where(okp[:, None], pi_, 0)]                         # (N, K, ntok, 4)
                out[:, :ntok] = np.where(okp[:, None, None], np.nanmean(g, 1), np.nan)
            return out
        Xi = np.concatenate([Xi, peer_avg(Xi, E04_CFG["intra_len"])], -1)
        Xd = np.concatenate([Xd, peer_avg(Xd, Xd.shape[1])], -1)
    return (Xi.astype(np.float32), Xd.astype(np.float32), Xp.astype(np.float32), ch.astype(np.float32))


_OFFICIAL_HM = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")


def _tmin(hm):
    """时刻 HHMM（整数数组）-> 当日该时刻及之前的连续竞价分钟数（09:31–11:30 为 1–120，13:01–15:00 为 121–240）。"""
    hm = np.asarray(hm)
    return np.where(hm <= 1130, (hm // 100 - 9) * 60 + hm % 100 - 30, 120 + (hm // 100 - 13) * 60 + hm % 100)


def _labels_from_bars(bars, ex):
    """S19：中间时刻截面的训练标签，与官方同口径——截面之后 30 个连续竞价分钟（11:30 之后接 13:01）的
    收盘 ÷ 同窗口 VWAP − 1，VWAP 须在窗口 [最低 × 0.98, 最高 × 1.02] 内。ex：DataFrame[date, instrument]。"""
    b = bars[["date", "instrument", "high", "low", "close", "volume", "amount"]]
    hm = (b["date"].dt.hour * 100 + b["date"].dt.minute).to_numpy()
    keep = ((hm > 930) & (hm <= 1130)) | ((hm > 1300) & (hm <= 1500))
    b, hm = b[keep], hm[keep]
    m = _tmin(hm) - 1                                                            # 分钟序号 0..239
    pairs = ex[["instrument"]].assign(day=ex["date"].dt.normalize()).drop_duplicates().reset_index(drop=True)
    pid = pd.MultiIndex.from_frame(pairs[["instrument", "day"]])
    pi = pid.get_indexer(pd.MultiIndex.from_arrays([b["instrument"].values, b["date"].dt.normalize().values]))
    ok = pi >= 0
    P = len(pairs)
    cube = {}
    for c in ("high", "low", "close", "volume", "amount"):
        a = np.full((P, 240), np.nan, np.float64)
        a[pi[ok], m[ok]] = b[c].to_numpy(np.float64)[ok]
        cube[c] = a
    out = []
    t_hm = (ex["date"].dt.hour * 100 + ex["date"].dt.minute).to_numpy()
    e_pi = pid.get_indexer(pd.MultiIndex.from_arrays([ex["instrument"].values, ex["date"].dt.normalize().values]))
    for t in np.unique(t_hm):
        k = int(_tmin(t))
        sl = slice(k, min(k + 30, 240))
        with np.errstate(all="ignore"):
            vol, amt = np.nansum(cube["volume"][:, sl], 1), np.nansum(cube["amount"][:, sl], 1)
            hi, lo = np.nanmax(cube["high"][:, sl], 1), np.nanmin(cube["low"][:, sl], 1)
            cc = cube["close"][:, sl]
            has = np.isfinite(cc)
            last = cc[np.arange(P), cc.shape[1] - 1 - np.argmax(has[:, ::-1], 1)]
            last = np.where(has.any(1), last, np.nan)
            vwap = np.where(vol > 0, amt / np.where(vol > 0, vol, 1.0), np.nan)
            vwap = np.where((vwap >= lo * 0.98) & (vwap <= hi * 1.02), vwap, np.nan)
            y = np.where(last > 0, last / vwap - 1.0, np.nan)
        sel = np.flatnonzero(t_hm == t)
        out.append(pd.DataFrame({"date": ex["date"].values[sel], "instrument": ex["instrument"].values[sel], "y": y[e_pi[sel]]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "instrument", "y"])


AUX_COLS = ["aux_" + c_ for c_ in E04_MTL_SET.split(",")]                        # 训练入口按 E04_MTL_SET 重设（10-24：默认也由 E04_MTL_SET 推出）


def _future_aux(bars, keys):
    """R22 多任务辅助目标（只在训练时用，推理不需要）：截面之后的连续竞价分钟（11:30 之后接 13:01，与官方标签同口径）。
    返回 {名字: 与 keys 行对齐的数组}：
      rv 之后 30 分钟已实现波动 log(Σr²)；lv 量能变化 log(之后 30 分钟成交量 + 1) − log(之前 30 分钟 + 1)；
      r60／r10 之后 60／10 分钟收益（收盘对截面收盘，取对数；超出当天的置缺失）；
      hi／lo 之后 30 分钟最高／最低收盘相对截面收盘（对数）；sv 之后 30 分钟签名成交量占比 Σ量·sign(r) / Σ量（10-21 R26）；
      au／dr 只在 14:30 截面（10-24 R33）：au 收盘竞价跳动 ln(15:00 收盘 ÷ 14:57 收盘)，dr 连续竞价漂移 ln(14:57 收盘 ÷ 14:31–14:57 量加权收盘)，
      其余截面缺失（rank_gauss 置 0）。14:30 的官方标签 ≈ 两者之和，模型在竞价部分明显弱于漂移部分。"""
    b = bars[["date", "instrument", "close", "volume"]]
    hm = (b["date"].dt.hour * 100 + b["date"].dt.minute).to_numpy()
    keep = ((hm > 930) & (hm <= 1130)) | ((hm > 1300) & (hm <= 1500))
    b, hm = b[keep], hm[keep]
    m = _tmin(hm) - 1                                                            # 分钟序号 0..239
    pairs = keys[["instrument"]].assign(day=keys["date"].dt.normalize()).drop_duplicates().reset_index(drop=True)
    pid = pd.MultiIndex.from_frame(pairs[["instrument", "day"]])
    pi = pid.get_indexer(pd.MultiIndex.from_arrays([b["instrument"].values, b["date"].dt.normalize().values]))
    ok = pi >= 0
    lc, vv = np.full((len(pairs), 240), np.nan), np.full((len(pairs), 240), np.nan)
    c = b["close"].to_numpy(np.float64)[ok]
    lc[pi[ok], m[ok]] = np.log(np.where(c > 0, c, np.nan))
    vv[pi[ok], m[ok]] = b["volume"].to_numpy(np.float64)[ok]
    r = np.diff(lc, axis=1)                                                      # r[:, j] = 序号 j+1 相对序号 j
    t_hm = (keys["date"].dt.hour * 100 + keys["date"].dt.minute).to_numpy()
    e_pi = pid.get_indexer(pd.MultiIndex.from_arrays([keys["instrument"].values, keys["date"].dt.normalize().values]))
    rv, lv = np.full(len(keys), np.nan), np.full(len(keys), np.nan)
    ex = {c_: np.full(len(keys), np.nan) for c_ in ("r60", "r10", "hi", "lo", "sv", "au", "dr")}
    for t in np.unique(t_hm):
        k = int(_tmin(t))                                                         # 截面收盘 = 序号 k−1；之后 30 分钟 = 序号 k..k+29
        if k <= 0 or k >= 240:
            continue
        sel = np.flatnonzero(t_hm == t)
        with np.errstate(all="ignore"):
            rr = r[:, k - 1:min(k + 29, 239)]
            n = np.isfinite(rr).sum(1)
            rvk = np.where(n >= 10, np.log(np.nansum(rr ** 2, 1) + 1e-10), np.nan)
            lvk = np.log1p(np.nansum(vv[:, k:min(k + 30, 240)], 1)) - np.log1p(np.nansum(vv[:, max(k - 30, 0):k], 1))
            c0 = lc[:, k - 1]
            fut = {"r60": lc[:, k + 59] - c0 if k + 59 <= 239 else np.full(len(lc), np.nan),
                   "r10": lc[:, k + 9] - c0 if k + 9 <= 239 else np.full(len(lc), np.nan),
                   "hi": np.nanmax(lc[:, k:min(k + 30, 240)], 1) - c0, "lo": np.nanmin(lc[:, k:min(k + 30, 240)], 1) - c0}
            vf = vv[:, k:min(k + 29, 239)]
            fut["sv"] = np.nansum(vf * np.sign(rr[:, :vf.shape[1]]), 1) / np.where(np.nansum(vf, 1) > 0, np.nansum(vf, 1), np.nan)
            if t == 1430:                                                      # 序号 236 = 14:57，239 = 15:00（收盘集合竞价）
                l57 = pd.DataFrame(lc[:, k:237]).ffill(axis=1).to_numpy()[:, -1]
                fut["au"] = lc[:, 239] - l57
                vc, cc = vv[:, k:237], np.exp(lc[:, k:237])
                vw = np.nansum(np.where(np.isfinite(cc), vc * cc, 0), 1) / np.where(np.nansum(np.where(np.isfinite(cc), vc, 0), 1) > 0,
                                                                                 np.nansum(np.where(np.isfinite(cc), vc, 0), 1), np.nan)
                fut["dr"] = l57 - np.log(np.where(vw > 0, vw, np.nan))
        rv[sel], lv[sel] = rvk[e_pi[sel]], lvk[e_pi[sel]]
        for c_, v_ in fut.items():
            ex[c_][sel] = v_[e_pi[sel]]
    return dict(rv=rv, lv=lv, **ex)


def e04_iter_chunks(backend, sd, ed, instruments, with_label, dz=None):
    """逐月产出 (Xf, Xi, Xd, Xp, y, keys)；Xf 已做 rel2 变换，Xi/Xd 为原始变换值（float16，未标准化），
    Xp 已逐日截面标准化；keys 含四个风格特征列。"""
    for lo, hi in month_chunks(sd, ed):
        t0 = time.time()
        sec = backend.sections(_fmt(lo), _fmt(hi), instruments)
        if with_label:
            lab = backend.labels(_fmt(lo), _fmt(hi), instruments)
            lab = lab.replace([np.inf, -np.inf], np.nan).dropna(subset=["y"])
            sec = sec.merge(lab[["date", "instrument", "y"]], on=["date", "instrument"], how="left")
        if sec.empty:
            continue
        sec = sec.sort_values(["date", "instrument"]).reset_index(drop=True)
        insts = sorted(sec["instrument"].unique().tolist())
        bars = backend.bars(_fmt(lo - pd.Timedelta(days=E04_BUFFER_DAYS)), _fmt(hi), insts, FEATURE_COLS + ["adjust_factor"])
        if with_label and EXTRA_TIMES:                                          # S19：同一 (股票, 交易日) 再加中间时刻截面
            pr = sec[["instrument"]].assign(day=sec["date"].dt.normalize()).drop_duplicates()
            ex = pd.concat([pr.assign(date=pr["day"] + pd.Timedelta(f"{t}:00")) for t in EXTRA_TIMES], ignore_index=True)
            ex = ex[(ex["date"] >= lo) & (ex["date"] <= hi)][["date", "instrument"]]
            ex = ex.merge(_labels_from_bars(bars, ex), on=["date", "instrument"], how="left").dropna(subset=["y"])
            sec = pd.concat([sec, ex], ignore_index=True).sort_values(["date", "instrument"]).reset_index(drop=True)
        Xf, _, _, keys = build_windows(bars, sec, coarse_len=0, compute_chars=False)
        mtl = with_label and E04_MTL_W > 0
        if mtl and len(keys):                                                    # R22：多任务辅助目标
            _fa = _future_aux(bars, keys)
            for c_ in AUX_COLS:
                keys[c_] = _fa[c_[4:]]
        pcube = _patch_cube(bars, keys, E04_CFG["patch_len"]) if E04_CFG.get("patch_len", 0) else None
        del bars
        if not len(keys):
            continue
        win = backend.windows30(_fmt(lo - pd.Timedelta(days=WIN_LOOKBACK_DAYS)), _fmt(hi), insts)
        if {"book", "book2", "book3"} & set(win_groups()):
            blb = max(BOOK_LOOKBACK_DAYS, int(E04_CFG["daily_len"] * 1.6) + 20)        # 日线专家加长时盘口也要多取
            bk = backend.book30(_fmt(lo - pd.Timedelta(days=blb)), _fmt(hi), insts)
            win = win.merge(bk, on=["date", "instrument"], how="left")
            del bk
        if "auc" in win_groups():                                                # 10-23：收盘集合竞价（只用前一交易日及更早，见 _e04_window_feats）
            alb = max(BOOK_LOOKBACK_DAYS, int(E04_CFG["daily_len"] * 1.6) + 20)
            au = backend.auc_daily(_fmt(lo - pd.Timedelta(days=alb)), _fmt(hi), insts)
            win = win.merge(au, on=["date", "instrument"], how="left")
            del au
        Xi, Xd, Xp, ch = _e04_window_feats(win, keys, dz, pcube)
        del pcube
        del win
        for j, cn in enumerate(CHAR_COLS):
            keys[cn] = ch[:, j]
        if with_label:
            keep = keys["y"].notna().to_numpy()
            Xf, Xi, Xd, Xp = Xf[keep], Xi[keep], Xd[keep], Xp[keep]
            keys = keys[keep].reset_index(drop=True)
            if not len(keys):
                continue
        Xf, _ = transform(Xf, None, E04_NORM)
        y = keys["y"].to_numpy(np.float32) if with_label else None
        logger.info("分片完成", start=_fmt(lo)[:10], samples=len(keys), elapsed=round(time.time() - t0, 1))
        yield (Xf, Xi.astype(np.float16), Xd.astype(np.float16), Xp.astype(np.float16), y,
               keys[["date", "instrument"] + CHAR_COLS + (AUX_COLS if mtl else [])])


def e04_build_dataset(table, sd, ed, mode, instruments, stats=None):
    """返回 (X=(Xf, Xi, Xd, Xp), y, keys, stats)；stats = (mf, sf, mi, si, md, sd_)，只在训练集上计算。"""
    backend = _backend(table)
    with_label = mode == "train"
    dz = daily_z(backend, sd, ed, instruments) if PEER_FROM_DAILY else None
    chunks = list(e04_iter_chunks(backend, sd, ed, instruments, with_label, dz))
    if not chunks:
        raise RuntimeError(f"e04_build_dataset 无样本 ({sd}~{ed})")
    n = sum(len(c[5]) for c in chunks)
    X = [np.empty((n,) + chunks[0][j].shape[1:], np.float16) for j in range(4)]
    ys, ks, pos = [], [], 0
    while chunks:
        c = chunks.pop(0)
        k = len(c[5])
        for j in range(4):
            X[j][pos:pos + k] = c[j]
        ks.append(c[5])
        if with_label:
            ys.append(c[4])
        pos += k
    if stats is None:
        idx = np.sort(np.random.RandomState(SEED).choice(n, min(n, STATS_SAMPLE), replace=False))
        stats = fit_stats(X[0][idx]) + fit_stats(X[1][idx]) + fit_stats(X[2][idx])
    for j in range(3):
        _standardize_inplace(X[j], stats[2 * j], stats[2 * j + 1])
    keys = pd.concat(ks, ignore_index=True)
    logger.info(f"{mode} 集构建完成", samples=len(keys))
    return tuple(X), (np.concatenate(ys) if with_label else None), keys, stats


def daily_panel(backend, sd, ed, instruments):
    """日线面板（由 30 分钟窗口表逐月汇总）：每只股票每个交易日的收盘（最后一个窗口末价）、昨收、成交额、成交量。"""
    parts = []
    for lo, hi in month_chunks(sd, ed):
        w = backend.windows30(_fmt(lo), _fmt(hi), instruments)
        if w.empty:
            continue
        w = w.sort_values("date").assign(day=w["date"].dt.normalize())
        g = w.groupby(["instrument", "day"], sort=False)
        parts.append(pd.DataFrame({"close": g["last_px"].last(), "pre": g["pre_close"].max(),
                                   "amt": g["sum_amt"].sum(), "vol": g["sum_vol"].sum()}).reset_index())
    if not parts:
        raise RuntimeError(f"daily_panel 无数据 ({sd}~{ed})")
    return pd.concat(parts, ignore_index=True).drop_duplicates(["instrument", "day"], keep="last")


def _spherical_kmeans(Z, k, seed, iters=25):
    """Z (n, d) 行已单位化；k-means++ 初始化（固定种子），余弦相似度分配。返回标签 (n,)。"""
    rs = np.random.RandomState(seed)
    n = len(Z)
    k = min(k, n)
    c = [rs.randint(n)]
    d2 = np.full(n, np.inf)
    for _ in range(1, k):
        d2 = np.minimum(d2, 2 - 2 * Z @ Z[c[-1]])
        p = np.maximum(d2, 0)
        c.append(rs.choice(n, p=p / p.sum()) if p.sum() > 0 else rs.randint(n))
    C = Z[c].copy()
    lab = np.zeros(n, np.int64)
    for _ in range(iters):
        new = np.argmax(Z @ C.T, axis=1)
        if _ and (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = Z[lab == j]
            if len(m):
                v = m.sum(0)
                C[j] = v / (np.linalg.norm(v) + 1e-12)
    return lab


def proxy_table(backend, sd, ed, instruments):
    """日级行情代理特征与伪行业（只用严格早于当天的交易日数据），返回 DataFrame[day, instrument, P_COLS..., pind]。
    规模 = ln(近 20 日日均成交额)；规模平方在逐截面标准化后计算；波动 = 近 20 日日收益标准差；beta = 近 60 日对截面均值；
    非流动性 = ln(近 20 日 |日收益| / 成交额 的均值)；动量 = 近 20 日、第 21–120 日累计收益；相对放量 = ln(近 20 日 / 近 240 日日均成交额)；
    低价 = -ln(前一日收盘)。伪行业：每月第一个交易日，用之前 PIND_DAYS 个交易日的超额日收益（减截面均值后逐股标准化）聚类。"""
    lo = pd.Timestamp(sd) - pd.Timedelta(days=DAILY_LOOKBACK_DAYS)
    dp = daily_panel(backend, _fmt(lo), _fmt(ed), instruments)
    piv = lambda c: dp.pivot(index="day", columns="instrument", values=c).sort_index()  # noqa: E731
    close, pre, amt = piv("close"), piv("pre"), piv("amt")
    amt = amt.where(amt > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log(close / pre).where(lambda x: np.isfinite(x) & (x.abs() < 0.5))
    m = r.mean(axis=1)
    ex = r.sub(m, axis=0)
    a20, a240 = amt.rolling(20, min_periods=10).mean(), amt.rolling(240, min_periods=60).mean()
    cov = r.mul(m, axis=0).rolling(60, min_periods=30).mean() - r.rolling(60, min_periods=30).mean().mul(
        m.rolling(60, min_periods=30).mean(), axis=0)
    var_m = (m * m).rolling(60, min_periods=30).mean() - m.rolling(60, min_periods=30).mean() ** 2
    s20, s120 = r.rolling(20, min_periods=10).sum(), r.rolling(120, min_periods=60).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        feats = {"p_size": np.log(a20), "p_vol": r.rolling(20, min_periods=10).std(),
                 "p_beta": cov.div(var_m.where(var_m > 1e-12), axis=0).clip(-1, 4),
                 "p_illiq": np.log((r.abs() / amt * 1e8).rolling(20, min_periods=10).mean()),
                 "p_mom20": s20, "p_mom120": s120 - s20, "p_rvol": np.log(a20 / a240), "p_lowp": -np.log(close)}
    feats = {k: v.shift(1) for k, v in feats.items()}                        # 只用前一交易日及更早
    days = close.index
    keep = days >= pd.Timestamp(sd).normalize()
    dk, ins = days[keep], close.columns                                        # 展开成长表（不用 stack，兼容新旧 pandas）
    out = pd.DataFrame({"day": np.repeat(dk.values, len(ins)), "instrument": np.tile(ins.values, len(dk))})
    for k, v in feats.items():
        out[k] = v[keep].reindex(columns=ins).to_numpy(np.float64).ravel()
    out["p_size2"] = np.nan                                                     # 逐截面标准化后再平方（见 neutral_matrix）
    pind = pd.Series(-1, index=pd.MultiIndex.from_arrays([out["day"], out["instrument"]]), dtype=np.int64)
    if PIND_K > 0:
        months = pd.Series(days[keep]).groupby(pd.Series(days[keep]).dt.to_period("M")).first()
        exv = ex.to_numpy(np.float64)
        for mo, d0 in months.items():
            i0 = days.get_loc(d0)
            if i0 < PIND_DAYS // 2:
                continue
            blk = exv[max(0, i0 - PIND_DAYS):i0]                              # 严格早于本月第一个交易日
            ok = np.isfinite(blk).sum(0) >= int(PIND_DAYS * 0.6)
            if ok.sum() < PIND_K * 5:
                continue
            Z = blk[:, ok]
            Z = (Z - np.nanmean(Z, 0)) / (np.nanstd(Z, 0) + 1e-12)
            Z = np.nan_to_num(Z).T
            Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
            lab = _spherical_kmeans(Z, PIND_K, SEED + mo.year * 12 + mo.month)
            mdays = days[keep][(days[keep].to_period("M") == mo)]
            insts = close.columns[ok]
            idx = pd.MultiIndex.from_product([mdays, insts])
            pind.loc[idx] = np.tile(lab, len(mdays))
    out["pind"] = pind.to_numpy()
    return out[["day", "instrument"] + P_COLS + ["pind"]]


def _official_matrix(keys):
    """仅本地诊断（10-25，上界实验）：官方 10 风格（当天，与评分机同口径）+ 申万一级行业哑变量，读本地 数据/exposure。
    平台线上环境不提供暴露表，这个选项不能用于交付（找不到文件时直接报错）。"""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "数据", "exposure")
    assert os.path.exists(os.path.join(root, "id_map.csv")), "TRAIN_NEUTRAL=official 只能本地诊断用"
    sty = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
    yrs = sorted(set(pd.DatetimeIndex(keys["date"]).year))
    e = pd.concat([pd.read_parquet(os.path.join(root, f"exposure_{y}.parquet")) for y in yrs if os.path.exists(os.path.join(root, f"exposure_{y}.parquet"))])
    e["day"] = pd.to_datetime(e["date"]).dt.normalize()
    mp = pd.read_csv(os.path.join(root, "id_map.csv"), dtype=str)
    mp = mp[mp["ok"].str.lower() == "true"]
    e = e.merge(mp[["instrument", "local_id"]], on="instrument").drop(columns=["instrument"]).rename(columns={"local_id": "instrument"})
    inds = sorted(e["industry_level1_code"].dropna().astype(str).unique())
    k = keys[["date", "instrument"]].assign(day=keys["date"].dt.normalize()).merge(
        e[["day", "instrument", "industry_level1_code"] + sty].drop_duplicates(["day", "instrument"]), on=["day", "instrument"], how="left")
    Z = cs_zscore(k[sty].to_numpy(np.float64), keys["date"].values)
    ic = k["industry_level1_code"].astype(str).to_numpy()
    oh = np.stack([(ic == c).astype(np.float32) for c in inds], 1)
    logger.info("训练剔除特征（官方，仅本地诊断）", cov=round(float(np.isfinite(k["SIZE"]).mean()), 4), industries=len(inds))
    return np.concatenate([np.nan_to_num(Z), oh], 1).astype(np.float32)


def neutral_matrix(keys, ptab):
    """训练剔除用的设计矩阵（不含截距）：代理特征逐截面 z-score（截断 ±5，缺失记 0）+ 规模平方 [+ 伪行业哑变量]。"""
    if TRAIN_NEUTRAL == "official":
        return _official_matrix(keys)
    k = keys[["date", "instrument"]].assign(day=keys["date"].dt.normalize()).merge(ptab, on=["day", "instrument"], how="left")
    cols = [c for c in P_COLS if c != "p_size2"]
    Z = cs_zscore(k[cols].to_numpy(np.float64), keys["date"].values)
    sz = Z[:, cols.index("p_size")]
    sq = cs_zscore((sz.astype(np.float64) ** 2)[:, None], keys["date"].values)
    B = [Z, sq]
    if TRAIN_NEUTRAL == "proxy_ind":
        pi = k["pind"].fillna(-1).to_numpy(np.int64)
        oh = np.zeros((len(k), PIND_K), np.float32)
        ok = pi >= 0
        oh[np.flatnonzero(ok), pi[ok]] = 1.0
        B.append(oh)
    logger.info("训练剔除特征", mode=TRAIN_NEUTRAL, cov=round(float(np.isfinite(k["p_size"]).mean()), 4),
                pind_cov=round(float((k["pind"].fillna(-1) >= 0).mean()), 4))
    return np.concatenate(B, 1).astype(np.float32)


class E04Net(nn.Module):
    """三个尺度专家 + 同伴编码器 + 同伴注意力；forward 输入需按截面连续排列，counts 为各截面样本数。
    输出 (B, 3)：三个专家各自的打分。"""

    def __init__(self, n_feat, d_model=64, nhead=4, n_layers_short=2, n_layers_long=1, dim_ff=128, seq_len=64,
                 intra_len=18, daily_len=20, peer_len=60, peer_dim=32, n_win_feat=4, dropout=0.1, slot_len=0, patch_len=0,
                 xs_layers=0, fuse=False, fuse_hidden=0, fuse_peer=False, tod_route="token", pool="mean", xscale=False, mgate=False, mtl=False,
                 peer_mode="attn", drop="", mtl_n=2):
        super(E04Net, self).__init__()
        self.peer_mode = peer_mode
        self.drop_k = [k for k, c in enumerate("sid") if c in (drop or "")]
        self.tod_route = tod_route if fuse else "token"
        n_tok = n_win_feat - 8 if self.tod_route != "token" else n_win_feat     # S22：时段通道（最后 8 个）不进 token

        def enc(nl):
            return nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True,
                                                                    activation="gelu"), nl)
        self.p_s, self.pos_s, self.enc_s = nn.Linear(n_feat, d_model), nn.Parameter(torch.zeros(1, seq_len, d_model)), enc(n_layers_short)
        self.p_i, self.pos_i, self.enc_i = nn.Linear(n_tok, d_model), nn.Parameter(torch.zeros(1, intra_len + 2 * slot_len + patch_len, d_model)), enc(n_layers_long)
        self.p_d, self.pos_d, self.enc_d = nn.Linear(n_tok, d_model), nn.Parameter(torch.zeros(1, daily_len, d_model)), enc(n_layers_long)
        self.peer = nn.Sequential(nn.Linear(peer_len, 64), nn.GELU(), nn.Linear(64, peer_dim))
        self.peer_dim = peer_dim
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 1)) for _ in range(3)])
        self.pool, self.xscale, self.mgate = pool, bool(xscale), bool(mgate)
        if self.mgate:
            md = n_feat + 2 * n_tok
            self.mg = nn.ModuleList([nn.Linear(md, n_feat), nn.Linear(md, n_tok), nn.Linear(md, n_tok)])
            for l_ in self.mg:
                nn.init.zeros_(l_.weight)
                nn.init.zeros_(l_.bias)
        if pool == "attn":
            self.pool_q = nn.Parameter(torch.zeros(3, 1, d_model))
            self.pool_att = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) for _ in range(3)])
        elif pool == "mix":
            self.pool_mix = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(3)])
        if self.xscale:
            self.xsc_type = nn.Parameter(torch.zeros(3, d_model))
            self.xsc_att = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.xsc_norm = nn.LayerNorm(d_model)
        self.mtl = bool(mtl)
        if self.mtl:
            self.aux_head = nn.Sequential(nn.LayerNorm(6 * d_model), nn.Linear(6 * d_model, d_model), nn.GELU(), nn.Linear(d_model, mtl_n))
        self.xs_layers, self.fuse = xs_layers, fuse
        if xs_layers:                                                   # A1：截面 Transformer（股票之间互相注意，无位置编码）
            self.xs_peer = nn.Linear(peer_dim, d_model)
            self.xs_type = nn.Parameter(torch.zeros(3, d_model))
            self.xs = enc(xs_layers)
        self.fuse_peer = bool(fuse and fuse_peer)
        if fuse:                                                        # A2：跨尺度融合头
            fi = 6 * d_model + (peer_dim if self.fuse_peer else 0) + (8 if self.tod_route == "fuse" else 0)
            if self.tod_route == "film":                                # S22：融合头拆成两段，中间按时段缩放 / 平移
                assert not fuse_hidden
                self.fuse_in = nn.Sequential(nn.LayerNorm(fi), nn.Linear(fi, d_model), nn.GELU())
                self.fuse_out = nn.Linear(d_model, 1)
                self.tod_gamma = nn.Embedding(8, d_model)
                self.tod_beta = nn.Embedding(8, d_model)
                nn.init.zeros_(self.tod_gamma.weight)
                nn.init.zeros_(self.tod_beta.weight)
            elif fuse_hidden:                                             # S17：加宽加深
                self.fuse_head = nn.Sequential(nn.LayerNorm(fi), nn.Linear(fi, fuse_hidden), nn.GELU(), nn.Dropout(dropout),
                                               nn.Linear(fuse_hidden, d_model), nn.GELU(), nn.Linear(d_model, 1))
            else:
                self.fuse_head = nn.Sequential(nn.LayerNorm(fi), nn.Linear(fi, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, xf, xi, xd, xp, counts):
        if self.tod_route != "token":                                   # S22：时段 one-hot（标准化后）取出，token 只留其余通道
            tod = xd[:, :, -8:].mean(1).argmax(-1)
            xi, xd = xi[..., :-8], xd[..., :-8]
        if self.mgate:                                                   # 10-20：市场门控（截面内最近 token 的均值 → 逐通道缩放）
            mv = torch.cat([xf[:, -1], xi[:, -1], xd[:, -1]], -1)
            ms = torch.cat([g.mean(0, keepdim=True).expand(len(g), -1) for g in torch.split(mv, counts)])
            gf, gi, gd = [2 * torch.sigmoid(l_(ms)) for l_ in self.mg]
            xf, xi, xd = xf * gf.unsqueeze(1), xi * gi.unsqueeze(1), xd * gd.unsqueeze(1)
        H = [self.enc_s(self.p_s(xf) + self.pos_s), self.enc_i(self.p_i(xi) + self.pos_i), self.enc_d(self.p_d(xd) + self.pos_d)]
        E = [self._pool(k, h) for k, h in enumerate(H)]
        for k in self.drop_k:                                            # 10-21 消融：去掉的尺度专家表示置零
            E[k] = torch.zeros_like(E[k])
        if self.xscale:                                                  # 10-20：跨尺度注意力（查询 = 三个专家的池化表示）
            kv = torch.cat([h + self.xsc_type[k] for k, h in enumerate(H)], 1)
            qv = torch.stack(E, 1)
            a = self.xsc_att(qv, kv, kv)[0]
            E = [self.xsc_norm(E[k] + a[:, k]) for k in range(3)]
        q = self.peer(xp)
        if self.xs_layers:                                              # A1：每个专家的表示在截面内过一遍 Transformer
            Z = []
            for k in range(3):
                tok = E[k] + self.xs_peer(q) + self.xs_type[k]
                Z.append(torch.cat([self.xs(t.unsqueeze(0)).squeeze(0) for t in torch.split(tok, counts)]))
            pair = [torch.cat([E[k], Z[k]], -1) for k in range(3)]
        elif self.peer_mode == "mean":                                  # 10-21 消融：减去同一截面的平均表示
            pair = [torch.cat([E[k], E[k] - torch.cat([e.mean(0, keepdim=True).expand(len(e), -1) for e in torch.split(E[k], counts)])], -1) for k in range(3)]
        elif self.peer_mode == "none":                                  # 10-21 消融：不做同伴对比
            pair = [torch.cat([E[k], torch.zeros_like(E[k])], -1) for k in range(3)]
        else:
            ctx = [[] for _ in E]
            for qs, es in zip(torch.split(q, counts), zip(*[torch.split(e, counts) for e in E])):
                a = qs @ qs.t() / math.sqrt(self.peer_dim)
                a = a.masked_fill(torch.eye(len(qs), dtype=torch.bool, device=a.device), float("-inf"))
                a = torch.softmax(a, -1) if len(qs) > 1 else torch.zeros_like(a)
                for k in range(3):
                    ctx[k].append(a @ es[k])
            pair = [torch.cat([E[k], E[k] - torch.cat(ctx[k])], -1) for k in range(3)]
        out = [self.heads[k](pair[k]).squeeze(-1) for k in range(3)]
        if self.mtl:                                                     # R22：辅助头输出存在属性里，返回值不变
            self.aux_out = self.aux_head(torch.cat(pair, -1))
        if self.fuse:                                                   # 第 4 列 = 融合头
            fin = torch.cat(pair + ([q] if self.fuse_peer else []) + ([F.one_hot(tod, 8).to(q.dtype)] if self.tod_route == "fuse" else []), -1)
            if self.tod_route == "film":
                h = self.fuse_in(fin)
                out.append(self.fuse_out(h * (1 + self.tod_gamma(tod)) + self.tod_beta(tod)).squeeze(-1))
            else:
                out.append(self.fuse_head(fin).squeeze(-1))
        return torch.stack(out, 1)

    def _pool(self, k, h):
        if self.pool == "attn":
            return self.pool_att[k](self.pool_q[k].expand(h.shape[0], -1, -1), h, h)[0].squeeze(1)
        if self.pool == "mix":
            return self.pool_mix[k](torch.cat([h.mean(1), h[:, -1]], -1))
        return h.mean(1)


def _e04_tensors(X, idx, device):
    return [torch.from_numpy(np.ascontiguousarray(X[j][idx])).to(device).float() for j in range(4)]


@torch.inference_mode()
def e04_predict_array(models, X, keys, device):
    """逐截面前向；每个模型的三个专家各自截面排名（居中）后等权平均，多个模型再取平均。"""
    models = models if isinstance(models, (list, tuple)) else [models]
    if device.type == "cpu" and torch.get_num_threads() < (os.cpu_count() or 1):
        torch.set_num_threads(os.cpu_count() or 1)
    for m in models:
        m.eval()
    order = np.argsort(keys["date"].values, kind="stable")
    dates = keys["date"].values[order]
    out = np.zeros(len(keys))
    for a, b in _segments(dates):
        idx = order[a:b]
        xs = _e04_tensors(X, idx, device)
        acc = np.zeros(b - a)
        for m in models:
            S = m(*xs, [b - a]).float().cpu().numpy()
            if getattr(m, "fuse", False):                                   # 融合模型只用融合头（可再混日线专家头，S23）
                if E04_DAILY_MIX > 0:
                    r = pd.DataFrame(S[:, [3, 2]]).rank(pct=True).to_numpy() - 0.5
                    acc += (1 - E04_DAILY_MIX) * r[:, 0] + E04_DAILY_MIX * r[:, 1]
                    continue
                S = S[:, 3:4]
            r = pd.DataFrame(S).rank(pct=True).to_numpy() - 0.5
            acc += r.mean(1)
        out[idx] = acc / len(models)
    return out


def style_residual_np(v, B, dates):
    """逐截面 OLS（含截距）：v 对风格暴露 B 回归取残差（训练前已排除缺失暴露的样本；推理不使用）。"""
    v = np.asarray(v, np.float64)
    out = v.copy()
    Bz = np.nan_to_num(np.asarray(B, np.float64))
    for a, b in _segments(np.asarray(dates)):
        X = np.column_stack([np.ones(b - a), Bz[a:b]])
        if not np.any(Bz[a:b]):
            continue
        coef, *_ = np.linalg.lstsq(X, v[a:b], rcond=None)
        out[a:b] = v[a:b] - X @ coef
    return out


def _resid_t(f, X):
    """可导的截面残差：f 对 [1, X] 回归取残差（小矩阵，在 CPU 上求解）。"""
    Xc = torch.cat([torch.ones(len(X), 1, device=X.device), X], 1)
    A = (Xc.t() @ Xc).cpu() + 1e-4 * torch.eye(Xc.shape[1])
    b = (Xc.t() @ f.unsqueeze(1)).cpu()
    coef = torch.linalg.solve(A, b).to(f.device)
    return f - (Xc @ coef).squeeze(1)


def _corr_t(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    return (xc * yc).sum() / torch.sqrt((xc * xc).sum() * (yc * yc).sum() + 1e-12)


def _e04_train_one(X, y, sec_id, seed, device, on_epoch_end=None, member=0, style=None, inst=None, sec_day=None, sec_w=None, ya=None, sec_skip=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = dict(E04_CFG, n_feat=N_FEAT, seq_len=SEQ_LEN)
    model = E04Net(**cfg).to(device)
    ema = copy.deepcopy(model) if E04_WEIGHT_EMA > 0 else None             # S20：权重滑动平均
    if ema is not None:
        for p_ in ema.parameters():
            p_.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) if WEIGHT_DECAY > 0 \
        else torch.optim.Adam(model.parameters(), lr=LR)
    order = np.argsort(sec_id, kind="stable")
    groups = [order[a:b] for a, b in _segments(sec_id[order])]           # sec_id 按时间递增，groups 按时间排列
    gw = np.array([sec_w[sec_id[g[0]]] for g in groups]) if sec_w is not None else None   # S18：每个截面的年份权重
    if DAY_BAG > 0:                                                       # R16：按交易日抽样，每个成员随机去掉 DAY_BAG 比例的交易日（成员之间更不相关）
        gd0 = np.array([sec_day[sec_id[g[0]]] for g in groups])
        ud = np.unique(gd0)
        drop = ud[np.random.default_rng(seed + 7919).random(len(ud)) < DAY_BAG]
        keep = np.flatnonzero(~np.isin(gd0, drop))
        groups = [groups[k] for k in keep]
        gw = gw[keep] if gw is not None else None
        logger.info("按交易日抽样", member=member, days=len(ud), dropped=len(drop), sections=len(groups))
    if sec_skip is not None:                                             # 10-23 混合集成：这个成员跳过中间时刻截面
        keep = np.flatnonzero(~sec_skip[np.array([sec_id[g[0]] for g in groups])])
        groups = [groups[k] for k in keep]
        gw = gw[keep] if gw is not None else None
        logger.info("成员跳过中间时刻截面", member=member, sections=len(groups))
    gday_all = np.array([sec_day[sec_id[g[0]]] for g in groups])
    day_seg = list(_segments(gday_all))

    def _pair_units(off):                                                # 同一交易日内 (off, off+1)、(off+2, off+3)…，其余单独成单元
        us = []
        for a, b in day_seg:
            ks = list(range(a, b))
            us += [ks[:off]] if off and ks[:off] else []
            us += [ks[i:i + 2] for i in range(off, len(ks), 2)]
        return [u for u in us if u]
    if PAIR_UNITS:
        units = _pair_units(0)
        per = max(1, SECS_PER_BATCH // 2)
    elif DAY_BATCH > 0:                                                  # 以交易日为单位：units[k] = 该日按时间排列的截面
        units = [list(range(a, b)) for a, b in day_seg]
        per = DAY_BATCH
    else:
        units = [[k] for k in range(len(groups))]
        per = SECS_PER_BATCH
    n_batches = math.ceil(len(units) / per)
    steps = EPOCHS * n_batches
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, (lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))) if LR_SCHEDULE == "cos" else (lambda s: 1.0))
    yt = torch.from_numpy(y)
    yat = torch.from_numpy(ya) if ya is not None else None                  # R22：辅助目标 (N, 2)
    gen = torch.Generator().manual_seed(seed)
    start_ep = 0
    if CKPT_DIR:
        os.makedirs(CKPT_DIR, exist_ok=True)
        done = sorted(int(f.split("_ep")[1].split(".")[0]) for f in os.listdir(CKPT_DIR)
                      if f.startswith(f"m{member}_ep") and f.endswith(".pt"))
        if done:
            st = torch.load(os.path.join(CKPT_DIR, f"m{member}_ep{done[-1]}.pt"), map_location="cpu")
            model.load_state_dict(st["model"])
            if ema is not None and "ema" in st:
                ema.load_state_dict(st["ema"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            gen.set_state(st["gen"])
            start_ep = st["epoch"]
            logger.info("从断点恢复", member=member, epoch=start_ep)
    every = max(1, n_batches // max(PROGRESS_STEPS, 1))
    for ep in range(start_ep, EPOCHS):
        model.train()
        t, tot, nb, seen, win, wn = time.time(), 0.0, 0, 0, 0.0, 0
        if PAIR_UNITS:
            units = _pair_units(ep % 2)
        perm = torch.randperm(len(units), generator=gen).numpy()
        for bi in range(0, len(perm), per):
            ul = [units[u] for u in perm[bi:bi + per]]
            gs = [groups[k] for u in ul for k in u]
            gwb = [float(gw[k]) for u in ul for k in u] if gw is not None else [1.0] * len(gs)
            pairs, pos = [], 0                                            # 同一交易日内相邻截面在 gs 里的位置
            for u in ul:
                pairs += [(pos + q, pos + q + 1) for q in range(len(u) - 1)]
                pos += len(u)
            idx = np.concatenate(gs)
            counts = [len(g) for g in gs]
            xs = _e04_tensors(X, idx, device)
            yb = yt[idx].to(device)
            Bb = torch.from_numpy(np.nan_to_num(style[idx]).astype(np.float32)).to(device) if style is not None else None
            opt.zero_grad()

            def _batch_loss():
                S = model(*xs, counts)
                loss = 0.0
                aux_l = ((model.aux_out - yat[idx].to(device)) ** 2).mean() if yat is not None else None
                Bs = torch.split(Bb, counts) if Bb is not None else [None] * len(counts)
                combs, ics, combs_r = [], [], []
                for gi, (Sg, yg, Bg) in enumerate(zip(torch.split(S, counts), torch.split(yb, counts), Bs)):
                    zs = (Sg - Sg.mean(0)) / (Sg.std(0) + 1e-6)
                    comb = zs[:, 3] if Sg.shape[1] > 3 else zs.mean(1)            # 融合模型以融合头为合成分
                    if Sg.shape[1] > 3 and E04_ANCHOR_TRAIN > 0:                # S24：训练时就以"融合头 + 日线锚"的混合分为合成分
                        comb = (1 - E04_ANCHOR_TRAIN) * zs[:, 3] + E04_ANCHOR_TRAIN * zs[:, 2]
                    if Bg is not None and bool((Bg != 0).any()):        # IC 按"风格剔除后"的口径计算，与评测一致
                        ic = lambda f: _corr_t(_resid_t(f, Bg), yg)  # noqa: E731
                    else:
                        ic = lambda f: _corr_t(f, yg)  # noqa: E731
                    nk = Sg.shape[1]
                    hw = [E04_HEAD_W] * 3 + [1.0] if nk > 3 else [1.0] * nk       # S17：融合模型可降低三个专家头的权重
                    lk = sum(hw[k] * (((Sg[:, k] - yg) ** 2).mean() + E04_IC_WEIGHT * (1 - ic(Sg[:, k]))) for k in range(nk)) / sum(hw)
                    ic_c = ic(comb)
                    loss = loss + gwb[gi] * (lk + E04_IC_WEIGHT * (1 - ic_c))
                    if E04_TAIL_WEIGHT > 0:                                     # A4：尾部加权相关（权重 = |合成分 z|，不回传）
                        wt = comb.detach().abs().clamp(max=3.0)
                        cw = comb - (wt * comb).sum() / wt.sum()
                        yw = yg - (wt * yg).sum() / wt.sum()
                        wc = (wt * cw * yw).sum() / torch.sqrt((wt * cw * cw).sum() * (wt * yw * yw).sum() + 1e-12)
                        loss = loss + gwb[gi] * E04_TAIL_WEIGHT * (1 - wc)
                    if E04_LISTNET_WEIGHT > 0:                                  # R1a：列表式排序损失，头尾对称（多空、压力只看两端）
                        yz = (yg - yg.mean()) / (yg.std() + 1e-6)
                        for sg in (1.0, -1.0):
                            pt = torch.softmax(sg * yz, 0)
                            kl = (pt * (torch.log(pt + 1e-12) - torch.log_softmax(sg * comb, 0))).sum()
                            loss = loss + gwb[gi] * E04_LISTNET_WEIGHT * kl
                    combs.append((comb - comb.mean()) / (comb.std() + 1e-6))
                    if E04_LEG_TURN > 0:                                        # 腿换手损失用风格残差后的合成分（评分机按残差排名定腿）
                        cr = _resid_t(comb, Bg) if (Bg is not None and bool((Bg != 0).any())) else comb
                        combs_r.append((cr - cr.mean()) / (cr.std() + 1e-6))
                    ics.append(ic_c)
                loss = loss / len(gs)
                if TURN_WEIGHT > 0 and pairs:                                 # 换手惩罚：相邻截面同一股票的标准化合成得分之差
                    tp = []
                    for a, b in pairs:
                        _, pa, pb = np.intersect1d(inst[gs[a]], inst[gs[b]], assume_unique=True, return_indices=True)
                        if len(pa):
                            tp.append(((combs[a][torch.from_numpy(pa).to(device)] - combs[b][torch.from_numpy(pb).to(device)]) ** 2).mean())
                    if tp:
                        loss = loss + TURN_WEIGHT * torch.stack(tp).mean()
                if E04_LEG_TURN > 0 and pairs and combs_r:                     # 腿换手：相邻截面同一股票的软腿成员之差（阈值不回传）
                    lt = []
                    for a, b in pairs:
                        _, pa, pb = np.intersect1d(inst[gs[a]], inst[gs[b]], assume_unique=True, return_indices=True)
                        if len(pa) < 20:
                            continue
                        ca, cb = combs_r[a], combs_r[b]
                        sa, sb = ca.detach().sort().values, cb.detach().sort().values      # 分位数用排序取（MPS 不支持 torch.quantile）
                        ha, hb = sa[int(0.9 * (len(sa) - 1))], sb[int(0.9 * (len(sb) - 1))]
                        la, lb = sa[int(0.1 * (len(sa) - 1))], sb[int(0.1 * (len(sb) - 1))]
                        ia, ib = torch.from_numpy(pa).to(device), torch.from_numpy(pb).to(device)
                        xa, xb = ca[ia], cb[ib]
                        sl = torch.sigmoid((xa - ha) / E04_LEG_TAU) - torch.sigmoid((xb - hb) / E04_LEG_TAU)
                        ss = torch.sigmoid((la - xa) / E04_LEG_TAU) - torch.sigmoid((lb - xb) / E04_LEG_TAU)
                        lt.append((sl ** 2 + ss ** 2).mean())
                    if lt:
                        loss = loss + E04_LEG_TURN * torch.stack(lt).mean()
                if ICIR_WEIGHT > 0 and len(ics) > 2:                          # IC 稳定性：一批内截面 IC 的标准差
                    loss = loss + ICIR_WEIGHT * torch.stack(ics).std()
                if aux_l is not None:
                    loss = loss + E04_MTL_W * aux_l
                return loss
            loss = _batch_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if E04_SAM_RHO > 0:                                           # 10-20 SAM：沿梯度方向扰动 ρ 后的权重上重算梯度，再在原权重上更新
                with torch.no_grad():
                    ps = [p_ for p_ in model.parameters() if p_.grad is not None]
                    gn = torch.norm(torch.stack([p_.grad.norm() for p_ in ps])) + 1e-12
                    eps = [E04_SAM_RHO * p_.grad / gn for p_ in ps]
                    for p_, e_ in zip(ps, eps):
                        p_.add_(e_)
                opt.zero_grad()
                _batch_loss().backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                with torch.no_grad():
                    for p_, e_ in zip(ps, eps):
                        p_.sub_(e_)
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(E04_WEIGHT_EMA).add_(pm.detach(), alpha=1.0 - E04_WEIGHT_EMA)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            lv = loss.item()
            tot += lv
            win += lv
            nb += 1
            wn += 1
            seen += len(idx)
            if nb % every == 0 and nb < n_batches:
                el = time.time() - t
                logger.info("训练进度", member=member, epoch=f"{ep + 1}/{EPOCHS}", pct=f"{100 * nb / n_batches:.0f}%",
                            loss=round(win / wn, 6), samples_per_s=int(seen / el), eta_s=int(el / nb * (n_batches - nb)))
                win, wn = 0.0, 0
        logger.info("epoch 完成", member=member, epoch=f"{ep + 1}/{EPOCHS}", loss=round(tot / max(nb, 1), 6),
                    elapsed=round(time.time() - t, 1))
        if CKPT_DIR:
            path = os.path.join(CKPT_DIR, f"m{member}_ep{ep + 1}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "gen": gen.get_state(), "epoch": ep + 1, **({"ema": ema.state_dict()} if ema is not None else {})}, path + ".tmp")
            os.replace(path + ".tmp", path)
        if on_epoch_end is not None:
            on_epoch_end(ema if ema is not None else model, device, member, ep + 1)
            model.train()
    return ema if ema is not None else model, cfg


def aux_target(y, keys):
    """同一股票、同一交易日之后 TARGET_AUX_N 个截面的标签加权和（权重 DECAY^(k-1)；只用训练集标签，训练时使用），逐截面 z 分数，缺失置 0。"""
    d = pd.DataFrame({"date": keys["date"].values, "inst": keys["instrument"].values, "y": np.asarray(y, np.float64)})
    d["day"] = d["date"].values.astype("datetime64[D]")
    d["fam"] = ~pd.DatetimeIndex(d["date"]).strftime("%H:%M").isin(_OFFICIAL_HM)   # S19：中间时刻自成一组，官方截面的"之后 N 个"不变
    d = d.sort_values(["inst", "date"])
    g = d.groupby(["inst", "day", "fam"], sort=False)["y"]
    nx = [g.shift(-k) for k in range(1, TARGET_AUX_N + 1)]                  # 收盘前不足 N 个截面时用现有的（14:30 没有）
    d["a"] = sum(TARGET_AUX_DECAY ** k * v.fillna(0.0) for k, v in enumerate(nx)).where(sum(v.notna().astype(int) for v in nx) > 0)
    if TARGET == "cs_rank":                                                      # R2：辅助目标同样取秩 → 正态分位数
        d = d.sort_index()
        return rank_gauss(d["a"].to_numpy(), pd.factorize(d["date"])[0])
    s = d.groupby("date")["a"]
    z = ((d["a"] - s.transform("mean")) / (s.transform("std") + 1e-8)).clip(-5, 5).fillna(0.0)
    return z.sort_index().to_numpy(np.float32)                                # d 的行号即 keys 的原始顺序



_GB_DEFAULT = {"bags": 3, "trees": 500, "depth": 8, "lr": 0.03, "rows": 0.5, "cols": 0.7, "min_leaf": 1000, "lam": 1.0, "bins": 64,
               "fields": "full", "norm": "rel3", "rich": True, "weight": 1.0,
               "win": "shape,book,book2,book3,tod,pos,state"}                  # 树另用端到端自己的 30 分钟通道（Transformer 用不上，G6 验证）


def _gb_flatten(Xf, Xi, Xd, Xp, rich=True):
    """四个张量 → 平铺特征（float32）。只做通用池化（末值、短／长窗口均值、标准差、最近几个 token），不加人工公式。
    树对单个特征的单调变换不敏感，所以不需要标准化。"""
    out = []
    f = Xf.astype(np.float32)
    with np.errstate(all="ignore"):
        last = (1, 2, 3, 5, 10, 20, 30) if rich else (1,)
        out += [f[:, -k, :] for k in last] + [np.nanmean(f[:, -5:, :], 1), np.nanmean(f[:, -15:, :], 1), np.nanmean(f, 1), np.nanstd(f, 1)]
        i = Xi.astype(np.float32)
        out += [i[:, -k, :] for k in ((1, 2, 3, 4, 5, 6, 7, 8) if rich else (1, 2, 3, 4))] + [np.nanmean(i, 1)]
        d = Xd.astype(np.float32)
        out += [d[:, -k, :] for k in ((1, 2, 3, 4, 5) if rich else (1, 2))] + [np.nanmean(d, 1)]
    out.append(Xp.astype(np.float32))
    F = np.concatenate(out, 1)
    F[~np.isfinite(F)] = np.nan
    return F


def _gb_binize(F, edges):
    """分箱：缺失记 0，其余 1..len(edges)+1。"""
    B = np.zeros(F.shape, np.uint8)
    for j in range(F.shape[1]):
        x = F[:, j]
        ok = np.isfinite(x)
        B[ok, j] = (np.searchsorted(edges[j], x[ok], side="right") + 1).astype(np.uint8)
    return B


class _GBTrees:
    """逐层生长的直方图梯度提升（平方损失），torch 实现（CUDA／MPS／CPU）。树 = 每层节点的 (特征, 阈值箱) + 叶子值。"""

    def __init__(self, nb, depth, lr, lam, min_leaf):
        self.nb, self.depth, self.lr, self.lam, self.min_leaf = nb, depth, lr, lam, min_leaf
        self.trees = []

    def _route(self, B, feat, thr):
        nid = torch.zeros(B.shape[0], dtype=torch.long, device=B.device)
        for lvl in range(self.depth):
            base = (1 << lvl) - 1
            f = feat[base + nid]
            x = B.gather(1, f.clamp(min=0).unsqueeze(1)).squeeze(1).long()
            nid = 2 * nid + ((x > thr[base + nid]) & (f >= 0)).long()
        return nid

    def fit_tree(self, B, r, cols):
        n, D, nb, Fs = B.shape[0], self.depth, self.nb, len(cols)
        nid = torch.zeros(n, dtype=torch.long, device=B.device)
        feat = torch.full((2 ** D - 1,), -1, dtype=torch.long, device=B.device)
        thr = torch.zeros(2 ** D - 1, dtype=torch.long, device=B.device)
        Xs = B[:, cols].long()
        fidx = torch.arange(Fs, device=B.device).unsqueeze(0)
        for lvl in range(D):
            nn_ = 1 << lvl
            G = torch.zeros(nn_ * Fs * nb, device=B.device)
            C = torch.zeros(nn_ * Fs * nb, device=B.device)
            step = max(1, 40_000_000 // Fs)
            for s0 in range(0, n, step):
                sl = slice(s0, s0 + step)
                idx = ((nid[sl].unsqueeze(1) * Fs + fidx) * nb + Xs[sl]).reshape(-1)
                G.index_add_(0, idx, r[sl].unsqueeze(1).expand(-1, Fs).reshape(-1))
                C.index_add_(0, idx, torch.ones(idx.shape[0], device=B.device))
            G, C = G.view(nn_, Fs, nb), C.view(nn_, Fs, nb)
            GL, CL = G.cumsum(2), C.cumsum(2)
            Gt, Ct = GL[:, :, -1:], CL[:, :, -1:]
            GR, CR = Gt - GL, Ct - CL
            gain = GL ** 2 / (CL + self.lam) + GR ** 2 / (CR + self.lam) - Gt ** 2 / (Ct + self.lam)
            gain = torch.where((CL >= self.min_leaf) & (CR >= self.min_leaf), gain, torch.full_like(gain, -1.0))
            best = gain.view(nn_, -1).max(1)
            bf, bb = best.indices // nb, best.indices % nb
            base = nn_ - 1
            feat[base:base + nn_] = torch.where(best.values > 0, cols[bf], torch.full_like(bf, -1))
            thr[base:base + nn_] = bb
            f = feat[base + nid]
            x = B.gather(1, f.clamp(min=0).unsqueeze(1)).squeeze(1).long()
            nid = 2 * nid + ((x > thr[base + nid]) & (f >= 0)).long()
        L = 2 ** D
        sG = torch.zeros(L, device=B.device).index_add_(0, nid, r)
        sC = torch.zeros(L, device=B.device).index_add_(0, nid, torch.ones_like(r))
        self.trees.append((feat, thr, self.lr * sG / (sC + self.lam)))

    def predict(self, B):
        p = torch.zeros(B.shape[0], device=B.device)
        for feat, thr, leaf in self.trees:
            p += leaf[self._route(B, feat, thr)]
        return p

    def to_json(self):
        return [[t[0].tolist(), t[1].tolist(), [float(v) for v in t[2].tolist()]] for t in self.trees]

    @classmethod
    def from_json(cls, trees, depth, device):
        m = cls(0, depth, 0.0, 0.0, 0)
        m.trees = [(torch.tensor(f, dtype=torch.long, device=device), torch.tensor(t, dtype=torch.long, device=device),
                    torch.tensor(l, dtype=torch.float32, device=device)) for f, t, l in trees]
        return m


def _gb_chunks(backend, sd, ed, instruments, with_label, cfg):
    """按树成员的字段集合与输入变换逐月产出 (平铺特征, y, keys)；只用官方截面（不加中间时刻）。"""
    global E04_NORM, EXTRA_TIMES, WIN_GROUPS
    saved = (FIELD_SET, E04_NORM, list(EXTRA_TIMES), WIN_GROUPS)
    set_fields(cfg["fields"])
    E04_NORM, EXTRA_TIMES, WIN_GROUPS = cfg["norm"], [], cfg.get("win", WIN_GROUPS)
    try:
        for Xf, Xi, Xd, Xp, y, keys in e04_iter_chunks(backend, sd, ed, instruments, with_label):
            yield _gb_flatten(Xf, Xi, Xd, Xp, cfg["rich"]).astype(np.float16), y, keys
    finally:
        set_fields(saved[0])
        E04_NORM, EXTRA_TIMES, WIN_GROUPS = saved[1], saved[2], saved[3]


def gbdt_train(backend, instruments, device):
    """树模型成员：与 Transformer 同一训练目标（秩目标 + 多周期辅助 + 代理风格／伪行业残差），09:30 不参与；E04_GBDT 给参数。"""
    cfg = dict(_GB_DEFAULT, **E04_GBDT)
    Fs, ys, ks = [], [], []
    for F, y, k in _gb_chunks(backend, TRAIN_START, TRAIN_END, instruments, True, cfg):
        Fs.append(F); ys.append(y); ks.append(k[["date", "instrument"] + CHAR_COLS])
    F = np.concatenate(Fs); del Fs
    y = np.concatenate(ys).astype(np.float64)
    keys = pd.concat(ks, ignore_index=True)
    yt = make_target(y, keys)
    if TARGET_AUX_W > 0:
        yt = (yt + TARGET_AUX_W * aux_target(y, keys)).astype(np.float32)
    if TRAIN_NEUTRAL != "none":
        yt = style_residual_np(yt, neutral_matrix(keys, proxy_table(backend, TRAIN_START, TRAIN_END, instruments)), keys["date"].values).astype(np.float32)
    keep = (keys["date"].dt.strftime("%H%M") != "0930").to_numpy() & np.isfinite(y)
    F, yt = F[keep], np.nan_to_num(np.asarray(yt, np.float32)[keep])
    rs0 = np.random.RandomState(SEED)
    smp = F[rs0.choice(len(F), min(len(F), 400000), replace=False)].astype(np.float32)
    qs = np.linspace(0, 1, cfg["bins"])[1:-1]
    edges = np.stack([np.nanquantile(smp[:, j], qs) if np.isfinite(smp[:, j]).any() else np.zeros(len(qs)) for j in range(F.shape[1])]).astype(np.float32)
    B = np.zeros(F.shape, np.uint8)
    for s0 in range(0, len(F), 500000):
        B[s0:s0 + 500000] = _gb_binize(F[s0:s0 + 500000].astype(np.float32), edges)
    del F, smp
    B = torch.from_numpy(B).to(device)
    tgt = torch.from_numpy(yt).float().to(device)
    n, nf = B.shape
    logger.info("树模型成员训练", samples=n, features=nf, bags=cfg["bags"], trees=cfg["trees"], depth=cfg["depth"], device=str(device))
    models = []
    for b in range(cfg["bags"]):
        rs = np.random.RandomState(SEED + 1000 * b)
        m = _GBTrees(cfg["bins"], cfg["depth"], cfg["lr"], cfg["lam"], cfg["min_leaf"])
        pred = torch.zeros(n, device=device)
        for k in range(cfg["trees"]):
            rows = torch.from_numpy(rs.choice(n, int(n * cfg["rows"]), replace=False)).to(device)
            cols = torch.from_numpy(np.sort(rs.choice(nf, max(1, int(nf * cfg["cols"])), replace=False))).to(device)
            m.fit_tree(B[rows], tgt[rows] - pred[rows], cols)
            feat, thr, leaf = m.trees[-1]
            pred += leaf[m._route(B, feat, thr)]
            if (k + 1) % 100 == 0:
                logger.info("树模型成员进度", bag=b, trees=k + 1, mse=round(float(((tgt - pred) ** 2).mean()), 4))
        models.append(m.to_json())
    return {"cfg": cfg, "edges": edges.tolist(), "models": models}


def gbdt_predict(gb, backend, sd, ed, instruments, device):
    """树模型成员推理：返回 DataFrame[date, instrument, gb]（各袋平均）。"""
    cfg = gb["cfg"]
    edges = np.asarray(gb["edges"], np.float32)
    ms = [_GBTrees.from_json(t, cfg["depth"], device) for t in gb["models"]]
    parts = []
    for F, _, keys in _gb_chunks(backend, sd, ed, instruments, False, cfg):
        B = torch.from_numpy(_gb_binize(F.astype(np.float32), edges)).to(device)
        p = sum(m.predict(B) for m in ms) / len(ms)
        parts.append(pd.DataFrame({"date": keys["date"].values, "instrument": keys["instrument"].values, "gb": p.cpu().numpy()}))
    return pd.concat(parts, ignore_index=True)


def e04_train_and_save(backend, model_path, on_epoch_end=None, on_data_ready=None):
    device = pick_device()
    logger.info("训练设备", device=str(device), arch="e04", win_groups=WIN_GROUPS)
    assert n_win_feat() == E04_CFG["n_win_feat"], (WIN_GROUPS, E04_CFG["n_win_feat"])
    instruments = backend.pool(TRAIN_START, TRAIN_END)
    if MAX_TRAIN_INSTRUMENTS:
        instruments = instruments[:MAX_TRAIN_INSTRUMENTS]
    X, y, keys, stats = e04_build_dataset(backend, TRAIN_START, TRAIN_END, "train", instruments)
    ya = None
    if E04_MTL_W > 0:                                           # R22：辅助目标逐截面秩 → 正态分位数（缺失置 0）
        codes = pd.factorize(keys["date"])[0]
        ya = np.column_stack([rank_gauss(keys[c_].to_numpy(), codes) for c_ in AUX_COLS]).astype(np.float32)
        logger.info("多任务辅助目标", w=E04_MTL_W, cover=[round(float(keys[c_].notna().mean()), 3) for c_ in AUX_COLS])
    y_raw = y
    disp = pd.Series(np.asarray(y_raw, np.float64)).groupby(keys["date"].values).std() if DISP_W > 0 else None
    y = make_target(y, keys)
    if TARGET_AUX_W > 0:
        y = (y + TARGET_AUX_W * aux_target(y_raw, keys)).astype(np.float32)
    del y_raw
    if BOOST_PRED and BOOST_BETA > 0:                           # R5d：残差提升，第二阶段学第一阶段没学到的部分
        import glob as _glob
        bp = pd.concat([pd.read_parquet(f, columns=["date", "instrument", "factor"]) for q in BOOST_PRED.split(",") for f in sorted(_glob.glob(q))], ignore_index=True)
        bp["date"] = pd.to_datetime(bp["date"]); bp["instrument"] = bp["instrument"].astype(str)
        kk = pd.DataFrame({"date": pd.to_datetime(keys["date"].values), "instrument": keys["instrument"].astype(str).values})
        f1 = kk.merge(bp, on=["date", "instrument"], how="left")["factor"].to_numpy(np.float64)
        logger.info("残差提升", beta=BOOST_BETA, matched=float(np.isfinite(f1).mean()), f1_std=float(np.nanstd(f1)), y_std=float(np.std(y)))
        y = (y - BOOST_BETA * np.nan_to_num(f1, nan=0.0)).astype(np.float32)
    style = None
    if TRAIN_NEUTRAL != "none":                                 # 标签对行情代理特征（与伪行业）逐截面取残差，损失同口径
        style = neutral_matrix(keys, proxy_table(backend, TRAIN_START, TRAIN_END, instruments))
        y = style_residual_np(y, style, keys["date"].values).astype(np.float32)
    elif LABEL_NEUTRAL > 0:
        y = np.nan_to_num(neutralize(y, keys[CHAR_COLS].to_numpy(), keys["date"].values, LABEL_NEUTRAL,
                                     rank_values=False)).astype(np.float32)
    if LOSS_SKIP_0930:                                          # 09:30 推理输出常数、不计分，训练也不用它
        keep = (keys["date"].dt.strftime("%H%M") != "0930").to_numpy()
        X = tuple(a[keep] for a in X)
        y, keys = y[keep], keys[keep].reset_index(drop=True)
        style = style[keep] if style is not None else None
        ya = ya[keep] if ya is not None else None
    sec_id, sec_dates = pd.factorize(keys["date"])
    sec_id = sec_id.astype(np.int64)
    sec_day = pd.factorize(pd.DatetimeIndex(sec_dates).normalize())[0]
    sec_w = None
    if YEAR_W:                                                  # S18：按年份加权（截面权重归一化到均值 1）
        sec_w = np.array([float(YEAR_W.get(int(v), YEAR_W.get(str(int(v)), 1.0))) for v in pd.DatetimeIndex(sec_dates).year])
        sec_w = sec_w / sec_w.mean()
        logger.info("按年份加权", year_w=str(YEAR_W), w_min=round(float(sec_w.min()), 3), w_max=round(float(sec_w.max()), 3))
    if SLOT_W:                                                  # R3：按截面时刻加权
        hm = pd.DatetimeIndex(sec_dates).strftime("%H:%M")
        sw = np.array([float(SLOT_W.get(h, 1.0)) for h in hm])
        sec_w = sw if sec_w is None else sec_w * sw
        sec_w = sec_w / sec_w.mean()
        logger.info("按时刻加权", slot_w=str(SLOT_W), w_min=round(float(sec_w.min()), 3), w_max=round(float(sec_w.max()), 3))
    if DISP_W > 0:                                              # 10-20：按截面标签离散度加权（训练时用标签，推理不需要）
        sd = disp.reindex(pd.DatetimeIndex(sec_dates)).to_numpy(np.float64)
        sw = np.nan_to_num(np.clip((sd / np.nanmedian(sd)) ** DISP_W, 0.25, 4.0), nan=1.0)
        sec_w = sw if sec_w is None else sec_w * sw
        sec_w = sec_w / sec_w.mean()
        logger.info("按离散度加权", disp_w=DISP_W, w_min=round(float(sec_w.min()), 3), w_max=round(float(sec_w.max()), 3))
    inst = pd.factorize(keys["instrument"])[0].astype(np.int64)
    del keys
    if on_data_ready is not None:
        on_data_ready(stats)
    logger.info("训练样本", samples=len(y), models=N_MODELS, skip_0930=LOSS_SKIP_0930, day_batch=DAY_BATCH,
                turn_w=TURN_WEIGHT, icir_w=ICIR_WEIGHT, aux_w=TARGET_AUX_W, aux_n=TARGET_AUX_N,
                n_params=sum(p.numel() for p in E04Net(**dict(E04_CFG, n_feat=N_FEAT, seq_len=SEQ_LEN)).parameters()))
    sec_extra = None
    if EXTRA_TIMES and E04_EXTRA_MEMBERS:                                # 10-23 混合集成：哪些截面是中间时刻
        sec_extra = np.isin(pd.DatetimeIndex(sec_dates).strftime("%H:%M"), list(EXTRA_TIMES))
        logger.info("混合集成", extra_members=str(E04_EXTRA_MEMBERS), extra_sections=int(sec_extra.sum()), sections=len(sec_extra))
    res = [_e04_train_one(X, y, sec_id, SEED + k, device, on_epoch_end, k, style, inst, sec_day, sec_w, ya,
                          sec_skip=sec_extra if (sec_extra is not None and k not in E04_EXTRA_MEMBERS) else None) for k in range(N_MODELS)]
    tolist = lambda a: np.asarray(a, np.float32).tolist()  # noqa: E731
    gb = None
    if E04_GBDT:                                                         # 10-25：树模型成员（Transformer 训完、释放数据后再训）
        del X, y
        import gc
        gc.collect()
        gb = gbdt_train(backend, instruments, device)
    save_model({
        "arch": "e04",
        "gbdt": gb,
        "state_dicts": [m.state_dict() for m, _ in res],
        "model_cfg": res[0][1],
        "feature_cols": FEATURE_COLS, "field_set": FIELD_SET, "norm": E04_NORM,
        "label_neutral": LABEL_NEUTRAL, "out_neutral": OUT_NEUTRAL, "target": TARGET,
        "train_neutral": TRAIN_NEUTRAL, "pind_k": PIND_K, "pind_days": PIND_DAYS, "zero_0930": ZERO_0930,
        "peer_from_daily": PEER_FROM_DAILY, "loss_skip_0930": LOSS_SKIP_0930, "day_batch": DAY_BATCH, "day_bag": DAY_BAG, "disp_w": DISP_W, "turn_weight": TURN_WEIGHT, "icir_weight": ICIR_WEIGHT, "pair_units": PAIR_UNITS, "leg_turn": E04_LEG_TURN, "leg_tau": E04_LEG_TAU,
        "target_aux_w": TARGET_AUX_W, "target_aux_n": TARGET_AUX_N, "target_aux_decay": TARGET_AUX_DECAY, "head_w": E04_HEAD_W, "extra_times": list(EXTRA_TIMES), "extra_members": list(E04_EXTRA_MEMBERS), "weight_ema": E04_WEIGHT_EMA, "sam_rho": E04_SAM_RHO, "mtl_w": E04_MTL_W, "mtl_set": E04_MTL_SET, "daily_mix": E04_DAILY_MIX, "anchor_train": E04_ANCHOR_TRAIN,
        "year_w": {str(k): float(v) for k, v in YEAR_W.items()}, "slot_w": {str(k): float(v) for k, v in SLOT_W.items()}, "boost_pred": BOOST_PRED, "boost_beta": BOOST_BETA,
        "train_range": [TRAIN_START, TRAIN_END], "experts": E04_EXPERTS,
        "win_lookback_days": WIN_LOOKBACK_DAYS, "win_groups": WIN_GROUPS,
        "stats": [tolist(a) for a in stats],
    }, model_path)
    logger.info("模型已保存", path=model_path, bytes=os.path.getsize(model_path))
    return model_path


def e04_predict_range(models, table, sd, ed, instruments, stats, device, out_neutral=None, keep_chars=False, zero=None):
    backend = _backend(table)
    out_neutral = OUT_NEUTRAL if out_neutral is None else out_neutral
    zero = ZERO_0930 if zero is None else zero
    parts = []
    dz = daily_z(backend, sd, ed, instruments) if PEER_FROM_DAILY else None
    for Xf, Xi, Xd, Xp, _, keys in e04_iter_chunks(backend, sd, ed, instruments, with_label=False, dz=dz):
        X = (standardize(Xf, stats[0], stats[1]), standardize(Xi, stats[2], stats[3]),
             standardize(Xd, stats[4], stats[5]), Xp)
        keys = keys.copy()
        keys["factor"] = e04_predict_array(models, X, keys, device)
        parts.append(keys)
    if not parts:
        raise RuntimeError(f"e04_predict_range 无样本 ({sd}~{ed})")
    out = pd.concat(parts, ignore_index=True)
    if GBDT_CK:                                                          # 10-25：树模型成员，截面居中排名后按权重并入（Transformer 每个成员权重 1）
        g = gbdt_predict(GBDT_CK, backend, sd, ed, instruments, device)
        out = out.merge(g, on=["date", "instrument"], how="left")
        rg = (out.groupby("date")["gb"].rank(pct=True) - 0.5).fillna(0.0).to_numpy()
        n_m, w = len(models) if isinstance(models, (list, tuple)) else 1, float(GBDT_CK["cfg"].get("weight", 1.0))
        out["factor"] = (n_m * out["factor"].to_numpy() + w * rg) / (n_m + w)
        out = out.drop(columns=["gb"])
    if out_neutral and out_neutral > 0:
        out["factor"] = neutralize(out["factor"].to_numpy(), out[CHAR_COLS].to_numpy(), out["date"].values, out_neutral)
    if zero:
        out["factor"] = zero_0930(out["factor"].to_numpy(), out["date"])
    return out if keep_chars else out[["date", "instrument", "factor"]]


def zero_0930(factor, dates):
    """09:30 截面输出常数 0：平台对常数截面不计 IC、不计换手（因子赛道已验证）。"""
    return np.where(pd.DatetimeIndex(dates).strftime("%H%M") == "0930", 0.0, np.asarray(factor, np.float64))


def predict_any(models, X, keys, device):
    """统一的数组推理入口（本地验证用）：E04 需要按截面分组。"""
    return e04_predict_array(models, X, keys, device) if ARCH == "e04" else predict_array(models, X, device)


if __name__ == "__main__":
    train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"})
