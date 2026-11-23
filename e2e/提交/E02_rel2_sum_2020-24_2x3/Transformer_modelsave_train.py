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
TRAIN_START, TRAIN_END = "2020-01-01", "2024-12-31 23:59:59"   # 训练区间写死，切勿用测试区间训练
SEQ_LEN = 64  # 1m 回看根数
COARSE_LEN = 0  # 30m 回看根数（每天 9 根含集合竞价，约 5 个交易日）；0 = 关闭
SUMMARY = True  # True = 加入截面摘要特征 token（多尺度反转、波动、盘口等，逐截面 z-score）
EPOCHS, BATCH, LR, SEED = 3, 512, 0.001, 42
WEIGHT_DECAY = 0.0  # >0 时使用 AdamW
LR_SCHEDULE = "none"  # none / cos
N_MODELS = 2  # 种子集成个数（种子 SEED, SEED+1, ...）
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
                 coarse_len=COARSE_LEN, n_summary=0)
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
                 coarse_len=0, n_summary=0, dropout=0.1):
        super(StockTransformer, self).__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.coarse_len = coarse_len
        if coarse_len:
            self.proj_c = nn.Linear(n_feat, d_model)
            self.pos_c = nn.Parameter(torch.zeros(1, coarse_len, d_model))
        self.n_summary = n_summary
        if n_summary:
            self.proj_s = nn.Linear(n_summary, d_model)
            self.pos_s = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x, xc=None, xs=None):                     # (B, L, F), (B, Lc, F), (B, K) -> (B,)
        h = self.proj(x) + self.pos
        if self.coarse_len:
            h = torch.cat([h, self.proj_c(xc) + self.pos_c], dim=1)
        if self.n_summary:
            h = torch.cat([h, self.proj_s(xs).unsqueeze(1) + self.pos_s], dim=1)
        return self.head(self.encoder(h).mean(dim=1)).squeeze(-1)


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


def summary_names():
    return SUMMARY_BASE + (SUMMARY_FULL if "bid_num_orders1" in FEATURE_COLS else [])


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


def build_windows(bars, sections, seq_len=None, coarse_len=None):
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
        yield Xf, Xc, Xs, y, keys[["date", "instrument"]]


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


def predict_range(models, table, sd, ed, instruments, stats, device, norm=None):
    """推理：逐月切窗、标准化、预测，返回 DataFrame[date, instrument, factor]。每个截面只用自身的输入窗口。"""
    backend = _backend(table)
    parts = []
    for Xf, Xc, Xs, _, keys in iter_chunks(backend, sd, ed, instruments, with_label=False, norm=norm):
        X = (standardize(Xf, stats[0], stats[1]), standardize(Xc, stats[2], stats[3]) if Xc is not None else None, Xs)
        keys = keys.copy()
        keys["factor"] = predict_array(models, X, device).astype(np.float64)
        parts.append(keys)
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
        m = StockTransformer(**ckpt["model_cfg"]).to(device)
        m.load_state_dict({k: v.to(device) for k, v in sd.items()})
        m.eval()
        models.append(m)
    return models


def configure_from_ckpt(ckpt):
    """推理前按权重文件里的配置设置字段集合与回看长度，保证与训练一致。"""
    global SEQ_LEN, COARSE_LEN, NORM, SUMMARY
    set_fields(ckpt.get("field_set", "base"))
    SUMMARY = bool(ckpt["model_cfg"].get("n_summary", 0))
    SEQ_LEN = ckpt["model_cfg"]["seq_len"]
    COARSE_LEN = ckpt["model_cfg"].get("coarse_len", 0)
    NORM = ckpt.get("norm", "rel")


def load_stats(ckpt):
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
    device = pick_device()
    logger.info("训练设备", device=str(device))
    # 结构超参中与输入相关的部分按当前配置推导，保证与写死的常量一致
    MODEL_CFG.update(n_feat=N_FEAT, seq_len=SEQ_LEN, coarse_len=COARSE_LEN,
                     n_summary=len(summary_names()) if SUMMARY else 0)

    instruments = backend.pool(TRAIN_START, TRAIN_END)
    if MAX_TRAIN_INSTRUMENTS:
        instruments = instruments[:MAX_TRAIN_INSTRUMENTS]
    X, y, keys, stats = build_dataset(backend, TRAIN_START, TRAIN_END, "train", instruments)
    y = make_target(y, keys)
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
        "norm": NORM,
        "target": TARGET,
        "loss": LOSS,
        "train_range": [TRAIN_START, TRAIN_END],
        "data_alignment": DATA_ALIGNMENT,
        "mean": tolist(mf), "std": tolist(sf), "mean_c": tolist(mc), "std_c": tolist(sc),
    }, model_path)
    logger.info("模型已保存", path=model_path, bytes=os.path.getsize(model_path))
    return model_path


if __name__ == "__main__":
    train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"})
