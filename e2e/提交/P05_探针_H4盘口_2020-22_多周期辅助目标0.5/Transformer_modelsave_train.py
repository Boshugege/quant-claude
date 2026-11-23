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
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
TRAIN_START, TRAIN_END = "2020-01-01", "2022-12-31 23:59:59"   # 训练区间写死，切勿用测试区间训练
SEQ_LEN = 64  # 1m 回看根数
COARSE_LEN = 0  # 30m 回看根数（每天 9 根含集合竞价，约 5 个交易日）；0 = 关闭
SUMMARY = False  # True = 加入截面摘要特征（多尺度反转、波动、盘口等，逐截面 z-score）
SUMMARY_SET = "all"  # all（含慢变量股票特征）/ rev（只保留反转与盘口类，去掉波动、非流动性、价格、成交额、价差）
SUMMARY_POOL = False  # True = 摘要特征投影后加到池化向量上（序列保持 64 个 token，MPS 上快约 3 倍）
LABEL_NEUTRAL = 0.0  # >0：训练标签对四个风格特征（规模、特质波动、非流动性、低价）逐截面回归取残差的力度
OUT_NEUTRAL = 1.0  # >0：推理输出（截面排名）对同样四个特征逐截面中性化的力度（参照因子 104 号取 1.5）
EPOCHS, BATCH, LR, SEED = 3, 512, 0.001, 42
WEIGHT_DECAY = 0.0  # >0 时使用 AdamW
LR_SCHEDULE = "none"  # none / cos
N_MODELS = 1  # 种子集成个数（种子 SEED, SEED+1, ...）
MAX_TRAIN_INSTRUMENTS = 0  # 0 = 不限制
NORM = "rel2"  # raw / log / rel / rel2（量额相对本窗口均值，与计量单位无关）
TARGET = "cs_z"  # raw / cs_z
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
        CAST(COALESCE(ask_num_orders1, 0) + COALESCE(ask_num_orders2, 0) + COALESCE(ask_num_orders3, 0) AS DOUBLE) AS an
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
           AVG(CASE WHEN bn + an > 0 THEN (bv + av) / (bn + an) ELSE NULL END) AS b_vpo
    FROM cte_seg
    WHERE time_segment != -1
    GROUP BY instrument_id, trading_day, time_segment
)
SELECT
    CAST(CONCAT(f.trading_day, ' ',
                strftime(strptime(LPAD(f.time_segment, 6, '0'), '%H%M%S'), '%H:%M:%S')) AS DATETIME) AS date,
    all_instruments.instrument,
    f.b_imb, f.b_spr, f.b_conc, f.b_oimb, f.b_imb1, f.b_dimb, f.b_sprv, f.b_vpo
FROM cte_window f
LEFT JOIN all_instruments USING (instrument_id)
"""
BOOK30_COLS = ["b_imb", "b_spr", "b_conc", "b_oimb"]
BOOK2_COLS = ["b_imb1", "b_dimb", "b_sprv", "b_vpo"]


def _book30_dai(table, sd, ed, instruments):
    df = dai.query(_BOOK30_SQL.format(table=table), filters={"date": [sd, ed]}, compression=True).df()
    df = df.dropna(subset=["instrument"])
    df["instrument"] = df["instrument"].astype(str)
    df = df[df["instrument"].isin(set(instruments))].drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    for c in BOOK30_COLS + BOOK2_COLS:
        df[c] = df[c].astype("float64")
    return df[["date", "instrument"] + BOOK30_COLS + BOOK2_COLS]


DaiBackend.book30 = lambda self, sd, ed, instruments: _book30_dai(self.table, sd, ed, instruments)


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
    rel2：价格同 rel；量额笔数除以本样本 1m 窗口内该字段的均值后取 log1p（与计量单位无关）。"""
    norm = norm or NORM
    out = []
    ref = sref = None
    if norm in ("rel", "rel2"):
        close = Xf[..., FEATURE_COLS.index("close")].astype(np.float32)
        ok = close > 0
        last = np.where(ok.any(1), Xf.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1), 0)
        ref = np.where(ok.any(1), close[np.arange(len(close)), last], np.nan).astype(np.float32)
    si = [i for i, c in enumerate(FEATURE_COLS) if not _is_price(c)]
    pi = [i for i, c in enumerate(FEATURE_COLS) if _is_price(c)]
    if norm == "rel2":
        with np.errstate(invalid="ignore"):
            sref = np.nanmean(np.clip(Xf[..., si].astype(np.float32), 0, None), axis=1)   # (N, n_size)
        sref = np.where(np.isfinite(sref) & (sref > 0), sref, 1.0).astype(np.float32)
    for X in (Xf, Xc):
        if X is None:
            out.append(None)
            continue
        X = X.astype(np.float32)
        X[~np.isfinite(X)] = np.nan
        if norm == "rel2":
            X[..., si] = np.log1p(np.clip(X[..., si], 0, None) / sref[:, None, :])
        elif norm != "raw":
            X[..., si] = np.log1p(np.clip(X[..., si], 0, None))
        if norm in ("rel", "rel2"):
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
    cnt = np.bincount(codes).astype(np.float64)
    mean = np.bincount(codes, y) / cnt
    var = (np.bincount(codes, y * y) - cnt * mean ** 2) / np.maximum(cnt - 1, 1)
    z = (y - mean[codes]) / (np.sqrt(np.maximum(var, 0))[codes] + 1e-8)
    return np.nan_to_num(np.clip(z, -5, 5), nan=0.0).astype(np.float32)


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
        SEQ_LEN, COARSE_LEN, SUMMARY, NORM = ckpt["model_cfg"]["seq_len"], 0, False, "rel2"
        OUT_NEUTRAL = float(ckpt.get("out_neutral", 0.0))
        WIN_LOOKBACK_DAYS = int(ckpt.get("win_lookback_days", WIN_LOOKBACK_DAYS))
        ZERO_0930 = bool(ckpt.get("zero_0930", False))
        PEER_FROM_DAILY = bool(ckpt.get("peer_from_daily", False))
        E04_CFG.update({k: v for k, v in ckpt["model_cfg"].items() if k in E04_CFG})
        global WIN_GROUPS
        WIN_GROUPS = ckpt.get("win_groups", "shape" if E04_CFG["n_win_feat"] == 7 else "")
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
E04_CFG = dict(d_model=64, nhead=4, n_layers_short=2, n_layers_long=1, dim_ff=128, intra_len=18, daily_len=20, peer_len=60, peer_dim=32, n_win_feat=11)
PEER_FROM_DAILY = False  # True：同伴日收益向量取自一次性构建的日线面板（可取 peer_len=240 等长窗口，不加大逐月窗口表的回看）
E04_IC_WEIGHT = 1.0            # 损失 = MSE + 权重 x (1 - 截面相关)
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
DAY_BATCH = 0  # >0：每批取若干个完整交易日（按时间顺序的全部截面），以便计算相邻截面的换手惩罚；0 = 随机截面
TURN_WEIGHT = 0.0  # 换手惩罚：同一交易日相邻截面、同一股票的合成得分（截面标准化后）之差的平方均值
ICIR_WEIGHT = 0.0  # IC 稳定性：一批内各截面 IC 的标准差（对应 IC_IR 与压力）
TARGET_AUX_W = 0.5  # 多周期目标（只在训练时）：目标 = z(本截面标签) + 权重 x z(同一交易日之后 TARGET_AUX_N 个截面标签之和)
TARGET_AUX_N = 2  # 日内 / 日线专家的输入通道（E08 S7 起按组叠加；基础 4 通道 = 相对当前价的对数价格、对数 VWAP、相对成交额、收益）：
#   shape  上影线占比、实体占比、平均单笔成交额（3）
#   book   窗口内 1 分钟盘口的均值：1–3 档挂单量不平衡、相对价差、一档挂单量占 1–3 档的比例、1–3 档委托笔数不平衡（4）
#   state  股票状态，每个 token 相同：价位、20 日均成交额、20 日日收益波动、20 日非流动性、20 日平均单笔成交额，逐截面排名（5）
#   mkt    市场状态，同一截面相同：当日至今收益、最近一个窗口收益各自的截面离散度（对数）（2）
WIN_GROUPS = "shape,book"  #   book2  盘口细化（4）：一档挂单量不平衡、窗口内 1–3 档不平衡的变化（末减初）、相对价差在窗口内的波动、每笔委托平均挂单量（对数）
WIN_GROUP_DIM = {"shape": 3, "book": 4, "book2": 4, "state": 5, "mkt": 2}
BOOK_LOOKBACK_DAYS = 45        # 盘口窗口表向前多取的自然日（20 个交易日 + 余量；只有日内 / 日线 token 用到）


def win_groups():
    return [g for g in WIN_GROUPS.split(",") if g]


def n_win_feat():
    return 4 + sum(WIN_GROUP_DIM[g] for g in win_groups())


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


def _e04_window_feats(win, keys, dz=None):
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
    if ext or "state" in grp:
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
        Xi = np.stack(Xi, -1)
        idd = Dt[:, None] - ld + np.arange(ld)[None, :]                         # 截面所在交易日之前的完整交易日
        a_d = gather(amt_d, idd)
        Xd = [gather(Ld, idd) - Lt[:, None], gather(Lvd, idd) - Lt[:, None],
              np.log1p(a_d / (np.nanmean(a_d, 1, keepdims=True) + 1e-9)), gather(r_d, idd)]
        if ext:
            t_d = gather(at_d, idd)
            Xd += [gather(us_d, idd), gather(bd_d, idd), t_d - np.nanmean(t_d, 1, keepdims=True)]
        Xd = np.stack(Xd, -1)
        bcols = (BOOK30_COLS if "book" in grp else []) + (BOOK2_COLS if "book2" in grp else [])
        if bcols:                                                               # 盘口：窗口均值；日线 token 用当日各窗口的均值
            bi, bdy = [], []
            for c in bcols:
                b = cube(c)
                if c == "b_vpo":
                    b = np.log(np.where(b > 0, b, np.nan))
                fin = np.isfinite(b)
                s_ = np.add.reduceat(np.where(fin, b, 0.0), d_start, axis=1)
                n_ = np.add.reduceat(fin.astype(np.float64), d_start, axis=1)
                bi.append(gather(b, iw))
                bdy.append(gather(np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan), idd))
            Xi = np.concatenate([Xi, np.stack(bi, -1)], -1)
            Xd = np.concatenate([Xd, np.stack(bdy, -1)], -1)
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
    return (Xi.astype(np.float32), Xd.astype(np.float32), Xp.astype(np.float32), ch.astype(np.float32))


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
        Xf, _, _, keys = build_windows(bars, sec, coarse_len=0, compute_chars=False)
        del bars
        if not len(keys):
            continue
        win = backend.windows30(_fmt(lo - pd.Timedelta(days=WIN_LOOKBACK_DAYS)), _fmt(hi), insts)
        if {"book", "book2"} & set(win_groups()):
            bk = backend.book30(_fmt(lo - pd.Timedelta(days=BOOK_LOOKBACK_DAYS)), _fmt(hi), insts)
            win = win.merge(bk, on=["date", "instrument"], how="left")
            del bk
        Xi, Xd, Xp, ch = _e04_window_feats(win, keys, dz)
        del win
        for j, cn in enumerate(CHAR_COLS):
            keys[cn] = ch[:, j]
        if with_label:
            keep = keys["y"].notna().to_numpy()
            Xf, Xi, Xd, Xp = Xf[keep], Xi[keep], Xd[keep], Xp[keep]
            keys = keys[keep].reset_index(drop=True)
            if not len(keys):
                continue
        Xf, _ = transform(Xf, None, "rel2")
        y = keys["y"].to_numpy(np.float32) if with_label else None
        logger.info("分片完成", start=_fmt(lo)[:10], samples=len(keys), elapsed=round(time.time() - t0, 1))
        yield (Xf, Xi.astype(np.float16), Xd.astype(np.float16), Xp.astype(np.float16), y,
               keys[["date", "instrument"] + CHAR_COLS])


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


def neutral_matrix(keys, ptab):
    """训练剔除用的设计矩阵（不含截距）：代理特征逐截面 z-score（截断 ±5，缺失记 0）+ 规模平方 [+ 伪行业哑变量]。"""
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
                 intra_len=18, daily_len=20, peer_len=60, peer_dim=32, n_win_feat=4, dropout=0.1):
        super(E04Net, self).__init__()

        def enc(nl):
            return nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True,
                                                                    activation="gelu"), nl)
        self.p_s, self.pos_s, self.enc_s = nn.Linear(n_feat, d_model), nn.Parameter(torch.zeros(1, seq_len, d_model)), enc(n_layers_short)
        self.p_i, self.pos_i, self.enc_i = nn.Linear(n_win_feat, d_model), nn.Parameter(torch.zeros(1, intra_len, d_model)), enc(n_layers_long)
        self.p_d, self.pos_d, self.enc_d = nn.Linear(n_win_feat, d_model), nn.Parameter(torch.zeros(1, daily_len, d_model)), enc(n_layers_long)
        self.peer = nn.Sequential(nn.Linear(peer_len, 64), nn.GELU(), nn.Linear(64, peer_dim))
        self.peer_dim = peer_dim
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 1)) for _ in range(3)])

    def forward(self, xf, xi, xd, xp, counts):
        E = [self.enc_s(self.p_s(xf) + self.pos_s).mean(1),
             self.enc_i(self.p_i(xi) + self.pos_i).mean(1),
             self.enc_d(self.p_d(xd) + self.pos_d).mean(1)]
        q = self.peer(xp)
        ctx = [[] for _ in E]
        for qs, es in zip(torch.split(q, counts), zip(*[torch.split(e, counts) for e in E])):
            a = qs @ qs.t() / math.sqrt(self.peer_dim)
            a = a.masked_fill(torch.eye(len(qs), dtype=torch.bool, device=a.device), float("-inf"))
            a = torch.softmax(a, -1) if len(qs) > 1 else torch.zeros_like(a)
            for k in range(3):
                ctx[k].append(a @ es[k])
        out = [self.heads[k](torch.cat([E[k], E[k] - torch.cat(ctx[k])], -1)).squeeze(-1) for k in range(3)]
        return torch.stack(out, 1)


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


def _e04_train_one(X, y, sec_id, seed, device, on_epoch_end=None, member=0, style=None, inst=None, sec_day=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = dict(E04_CFG, n_feat=N_FEAT, seq_len=SEQ_LEN)
    model = E04Net(**cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) if WEIGHT_DECAY > 0 \
        else torch.optim.Adam(model.parameters(), lr=LR)
    order = np.argsort(sec_id, kind="stable")
    groups = [order[a:b] for a, b in _segments(sec_id[order])]           # sec_id 按时间递增，groups 按时间排列
    if DAY_BATCH > 0:                                                     # 以交易日为单位：units[k] = 该日按时间排列的截面
        gday = np.array([sec_day[sec_id[g[0]]] for g in groups])
        units = [list(range(a, b)) for a, b in _segments(gday)]
        per = DAY_BATCH
    else:
        units = [[k] for k in range(len(groups))]
        per = SECS_PER_BATCH
    n_batches = math.ceil(len(units) / per)
    steps = EPOCHS * n_batches
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, (lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))) if LR_SCHEDULE == "cos" else (lambda s: 1.0))
    yt = torch.from_numpy(y)
    gen = torch.Generator().manual_seed(seed)
    start_ep = 0
    if CKPT_DIR:
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
        perm = torch.randperm(len(units), generator=gen).numpy()
        for bi in range(0, len(perm), per):
            ul = [units[u] for u in perm[bi:bi + per]]
            gs = [groups[k] for u in ul for k in u]
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
            S = model(*xs, counts)
            loss = 0.0
            Bs = torch.split(Bb, counts) if Bb is not None else [None] * len(counts)
            combs, ics = [], []
            for Sg, yg, Bg in zip(torch.split(S, counts), torch.split(yb, counts), Bs):
                zs = (Sg - Sg.mean(0)) / (Sg.std(0) + 1e-6)
                comb = zs.mean(1)
                if Bg is not None and bool((Bg != 0).any()):        # IC 按"风格剔除后"的口径计算，与评测一致
                    ic = lambda f: _corr_t(_resid_t(f, Bg), yg)  # noqa: E731
                else:
                    ic = lambda f: _corr_t(f, yg)  # noqa: E731
                lk = sum(((Sg[:, k] - yg) ** 2).mean() + E04_IC_WEIGHT * (1 - ic(Sg[:, k])) for k in range(3)) / 3
                ic_c = ic(comb)
                loss = loss + lk + E04_IC_WEIGHT * (1 - ic_c)
                combs.append((comb - comb.mean()) / (comb.std() + 1e-6))
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
            if ICIR_WEIGHT > 0 and len(ics) > 2:                          # IC 稳定性：一批内截面 IC 的标准差
                loss = loss + ICIR_WEIGHT * torch.stack(ics).std()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
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
                        "gen": gen.get_state(), "epoch": ep + 1}, path + ".tmp")
            os.replace(path + ".tmp", path)
        if on_epoch_end is not None:
            on_epoch_end(model, device, member, ep + 1)
    return model, cfg


def aux_target(y, keys):
    """同一股票、同一交易日之后 TARGET_AUX_N 个截面的标签之和（只用训练集标签，训练时使用），逐截面 z 分数，缺失置 0。"""
    d = pd.DataFrame({"date": keys["date"].values, "inst": keys["instrument"].values, "y": np.asarray(y, np.float64)})
    d["day"] = d["date"].values.astype("datetime64[D]")
    d = d.sort_values(["inst", "date"])
    g = d.groupby(["inst", "day"], sort=False)["y"]
    nx = [g.shift(-k) for k in range(1, TARGET_AUX_N + 1)]                  # 收盘前不足 N 个截面时用现有的（14:30 没有）
    d["a"] = sum(v.fillna(0.0) for v in nx).where(sum(v.notna().astype(int) for v in nx) > 0)
    s = d.groupby("date")["a"]
    z = ((d["a"] - s.transform("mean")) / (s.transform("std") + 1e-8)).clip(-5, 5).fillna(0.0)
    return z.sort_index().to_numpy(np.float32)                                # d 的行号即 keys 的原始顺序


def e04_train_and_save(backend, model_path, on_epoch_end=None, on_data_ready=None):
    device = pick_device()
    logger.info("训练设备", device=str(device), arch="e04", win_groups=WIN_GROUPS)
    assert n_win_feat() == E04_CFG["n_win_feat"], (WIN_GROUPS, E04_CFG["n_win_feat"])
    instruments = backend.pool(TRAIN_START, TRAIN_END)
    if MAX_TRAIN_INSTRUMENTS:
        instruments = instruments[:MAX_TRAIN_INSTRUMENTS]
    X, y, keys, stats = e04_build_dataset(backend, TRAIN_START, TRAIN_END, "train", instruments)
    y_raw = y
    y = make_target(y, keys)
    if TARGET_AUX_W > 0:
        y = (y + TARGET_AUX_W * aux_target(y_raw, keys)).astype(np.float32)
    del y_raw
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
    sec_id, sec_dates = pd.factorize(keys["date"])
    sec_id = sec_id.astype(np.int64)
    sec_day = pd.factorize(pd.DatetimeIndex(sec_dates).normalize())[0]
    inst = pd.factorize(keys["instrument"])[0].astype(np.int64)
    del keys
    if on_data_ready is not None:
        on_data_ready(stats)
    logger.info("训练样本", samples=len(y), models=N_MODELS, skip_0930=LOSS_SKIP_0930, day_batch=DAY_BATCH,
                turn_w=TURN_WEIGHT, icir_w=ICIR_WEIGHT, aux_w=TARGET_AUX_W, aux_n=TARGET_AUX_N,
                n_params=sum(p.numel() for p in E04Net(**dict(E04_CFG, n_feat=N_FEAT, seq_len=SEQ_LEN)).parameters()))
    res = [_e04_train_one(X, y, sec_id, SEED + k, device, on_epoch_end, k, style, inst, sec_day) for k in range(N_MODELS)]
    tolist = lambda a: np.asarray(a, np.float32).tolist()  # noqa: E731
    save_model({
        "arch": "e04",
        "state_dicts": [m.state_dict() for m, _ in res],
        "model_cfg": res[0][1],
        "feature_cols": FEATURE_COLS, "field_set": FIELD_SET, "norm": "rel2",
        "label_neutral": LABEL_NEUTRAL, "out_neutral": OUT_NEUTRAL, "target": TARGET,
        "train_neutral": TRAIN_NEUTRAL, "pind_k": PIND_K, "pind_days": PIND_DAYS, "zero_0930": ZERO_0930,
        "peer_from_daily": PEER_FROM_DAILY, "loss_skip_0930": LOSS_SKIP_0930, "day_batch": DAY_BATCH, "turn_weight": TURN_WEIGHT, "icir_weight": ICIR_WEIGHT,
        "target_aux_w": TARGET_AUX_W, "target_aux_n": TARGET_AUX_N,
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
        if out_neutral and out_neutral > 0:
            keys["factor"] = neutralize(keys["factor"].to_numpy(), keys[CHAR_COLS].to_numpy(), keys["date"].values, out_neutral)
        if zero:
            keys["factor"] = zero_0930(keys["factor"].to_numpy(), keys["date"])
        parts.append(keys if keep_chars else keys[["date", "instrument", "factor"]])
    if not parts:
        raise RuntimeError(f"e04_predict_range 无样本 ({sd}~{ed})")
    return pd.concat(parts, ignore_index=True)


def zero_0930(factor, dates):
    """09:30 截面输出常数 0：平台对常数截面不计 IC、不计换手（因子赛道已验证）。"""
    return np.where(pd.DatetimeIndex(dates).strftime("%H%M") == "0930", 0.0, np.asarray(factor, np.float64))


def predict_any(models, X, keys, device):
    """统一的数组推理入口（本地验证用）：E04 需要按截面分组。"""
    return e04_predict_array(models, X, keys, device) if ARCH == "e04" else predict_array(models, X, device)


if __name__ == "__main__":
    train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"})
