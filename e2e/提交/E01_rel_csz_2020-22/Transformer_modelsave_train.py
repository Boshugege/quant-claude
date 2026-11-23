# -*- coding: utf-8 -*-
"""Transformer 端到端模型 —— 训练侧脚本（共享定义 + 从零训练并持久化）。

基于官方 demo 改写，接口保持一致（推理 notebook 从本文件导入）：
    MODEL_PATH, BATCH, StockTransformer, build_dataset, pool, train_and_save, load_model
以及推理用的分片预测函数 predict_range。

相对 demo 的改动（均属于类型转换 / 标准化 / 切窗，不构造跨字段的衍生特征）：
  1. 逐样本数值变换 NORM="rel"：量、额字段取 log1p；价格字段除以窗口内最后一个有效 close
     后取 log（消除价格水平差异）；随后按训练集统计做全局标准化。
  2. 训练目标 TARGET="cs_z"：vwap_return 在每个截面内做 z-score（截断 ±5）后用 MSE 拟合，
     与评分使用的截面秩相关一致，避免少数极端收益主导损失。
  3. 按月分片读取与切窗，样本以 float16 存储，控制内存；全局分钟网格切窗，停牌缺行记为缺失。
  4. 设备优先 cuda，其次 mps，最后 cpu；固定随机种子。

数据层：平台上默认使用 DaiBackend（datasources["bar1m"] + cpt_jyc_2026_instruments +
cpt_jyc_2026_vwap）。参赛者在本地训练时可传入实现了相同接口的 backend；模型、切窗、
标准化、训练逻辑完全共用本文件。

用法：
    python Transformer_modelsave_train.py           # 平台环境，训练并保存 transformer_model.json
    from Transformer_modelsave_train import train_and_save
    train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"})
"""
import json
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
SEQ_LEN = 64                   # 每个截面回看的原始 1m bar 数
EPOCHS, BATCH, LR, SEED = 5, 512, 1e-3, 42
MAX_TRAIN_INSTRUMENTS = 0      # 0 = 不限制
NORM = "rel"                   # raw / log / rel，见 transform
TARGET = "cs_z"                # raw / cs_z
BUFFER_DAYS = 10               # 每个分片向前多读的自然日，保证 64 根回看
STATS_SAMPLE = 200000          # 计算标准化统计量所用的训练样本数上限

FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "amount",
    "bid_price1", "ask_price1", "bid_volume1", "ask_volume1",
]
PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1"]
SIZE_COLS = ["volume", "amount", "bid_volume1", "ask_volume1"]
N_FEAT = len(FEATURE_COLS)
LABEL_TABLE = "cpt_jyc_2026_vwap"
SECTION_TABLE = "cpt_jyc_2026_instruments"

MODEL_CFG = dict(n_feat=N_FEAT, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN)
DATA_ALIGNMENT = "section_uses_grid_bars_le_section_time; label=vwap_return"


# ---------- 模型：逐 bar 投影 -> Transformer 编码 -> 均值池化 -> 回归头（同 demo） ----------
class StockTransformer(nn.Module):
    def __init__(self, n_feat, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN, dropout=0.1):
        super(StockTransformer, self).__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):                                       # (B, L, N_FEAT) -> (B,)
        h = self.encoder(self.proj(x) + self.pos).mean(dim=1)
        return self.head(h).squeeze(-1)


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

    def bars(self, sd, ed, instruments):
        sql = f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {self.table}"
        df = dai.query(sql, filters={"date": [sd, ed], "instrument": instruments}).df()
        df["date"] = pd.to_datetime(df["date"])
        return df


def _backend(table_or_backend):
    return DaiBackend(table_or_backend) if isinstance(table_or_backend, str) else table_or_backend


def pool(sd, ed, backend=None):
    """区间内中证 1000 成分股代码（同 demo 接口）。"""
    return (backend or DaiBackend(None)).pool(sd, ed)


# ---------- 切窗与标准化 ----------
def build_windows(bars, sections, seq_len=SEQ_LEN):
    """为每个 (date, instrument) 截面取全局分钟网格上 date <= 截面时间的最近 seq_len 根 bar。

    网格 = bars 中出现过的全部分钟时间戳；某股票某分钟缺行（停牌等）记为 NaN。
    返回 X (N, seq_len, N_FEAT) float32（含 NaN）与对应的 keys。"""
    grid = np.sort(bars["date"].unique())
    inst = np.sort(sections["instrument"].unique())
    b = bars[bars["instrument"].isin(inst)]
    ti = np.searchsorted(grid, b["date"].values)
    ii = np.searchsorted(inst, b["instrument"].values)
    cube = np.full((len(inst), len(grid), N_FEAT), np.nan, np.float32)
    cube[ii, ti] = b[FEATURE_COLS].to_numpy(np.float32)

    s = sections.drop_duplicates(["date", "instrument"])
    end = np.searchsorted(grid, s["date"].values, side="right")
    ok = end >= seq_len
    s, end = s[ok], end[ok]
    si = np.searchsorted(inst, s["instrument"].values)
    idx = end[:, None] - seq_len + np.arange(seq_len)[None, :]
    X = cube[si[:, None], idx]
    if len(s) and not (grid[idx[:, -1]] <= s["date"].values).all():
        raise RuntimeError("特征时点泄漏")
    return X, s.reset_index(drop=True)


def transform(X, norm=None):
    """逐样本数值变换，只做单字段的单调变换与价格归一，不组合不同字段。"""
    norm = norm or NORM
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    if norm == "raw":
        return X
    si = [FEATURE_COLS.index(c) for c in SIZE_COLS]
    X[..., si] = np.log1p(np.clip(X[..., si], 0, None))
    if norm == "log":
        return X
    pi = [FEATURE_COLS.index(c) for c in PRICE_COLS]
    close = X[..., FEATURE_COLS.index("close")]
    ref = pd.DataFrame(close.T).ffill().iloc[-1].to_numpy(np.float32)     # 每个样本最后一个有效 close
    p = X[..., pi]
    p[p <= 0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        X[..., pi] = np.log(p / ref[:, None, None])
    return X


def standardize(X, stats, chunk=200000):
    """原地分块标准化，NaN/inf 置 0，输出 float16。"""
    m, s = stats
    out = np.empty(X.shape, np.float16)
    for i in range(0, len(X), chunk):
        z = (X[i:i + chunk].astype(np.float32) - m) / s
        out[i:i + chunk] = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def month_chunks(sd, ed):
    sd, ed = pd.Timestamp(sd), pd.Timestamp(ed)
    starts = pd.date_range(sd.normalize().replace(day=1), ed, freq="MS")
    for st in starts:
        lo = max(st, sd)
        hi = min(st + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59, seconds=59), ed)
        if lo <= hi:
            yield lo, hi


def _fmt(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def iter_chunks(backend, sd, ed, instruments, with_label, norm=None):
    """逐月产出 (X_transformed float16, y, keys)。"""
    for lo, hi in month_chunks(sd, ed):
        t0 = time.time()
        sec = backend.sections(_fmt(lo), _fmt(hi), instruments)
        if with_label:
            lab = backend.labels(_fmt(lo), _fmt(hi), instruments)
            lab = lab.replace([np.inf, -np.inf], np.nan).dropna(subset=["y"])
            sec = sec.merge(lab[["date", "instrument", "y"]], on=["date", "instrument"], how="inner")
        if sec.empty:
            continue
        bars = backend.bars(_fmt(lo - pd.Timedelta(days=BUFFER_DAYS)), _fmt(hi),
                            sorted(sec["instrument"].unique().tolist()))
        X, keys = build_windows(bars, sec)
        del bars
        if not len(keys):
            continue
        X = transform(X, norm).astype(np.float16)
        y = keys["y"].to_numpy(np.float32) if with_label else None
        logger.info("分片完成", start=_fmt(lo)[:10], samples=len(keys), elapsed=round(time.time() - t0, 1))
        yield X, y, keys[["date", "instrument"]]


def build_dataset(table, sd, ed, mode, instruments, stats=None):
    """构建 30 分钟截面样本（接口同 demo）。

    table 可以是 bar1m 表名（平台）或实现了 DaiBackend 接口的 backend 对象（本地）。
    返回 (X float16 已标准化, y 或 None, keys 或 None, stats)；mode="train" 时 keys 也一并返回。"""
    backend = _backend(table)
    with_label = mode == "train"
    Xs, ys, ks = [], [], []
    for X, y, k in iter_chunks(backend, sd, ed, instruments, with_label):
        Xs.append(X)
        ks.append(k)
        if with_label:
            ys.append(y)
    if not Xs:
        raise RuntimeError(f"build_dataset 无样本 (mode={mode}, {sd}~{ed})")
    X = np.concatenate(Xs)
    del Xs
    keys = pd.concat(ks, ignore_index=True)
    if stats is None:                                           # 只在训练集上计算
        rs = np.random.RandomState(SEED)
        sub = X[rs.choice(len(X), min(len(X), STATS_SAMPLE), replace=False)].astype(np.float32)
        flat = sub.reshape(-1, N_FEAT)
        stats = (np.nanmean(flat, 0).astype(np.float32), (np.nanstd(flat, 0) + 1e-6).astype(np.float32))
    X = standardize(X, stats)
    logger.info(f"{mode} 集构建完成", samples=len(keys))
    if with_label:
        return X, np.concatenate(ys), keys, stats
    return X, None, keys, stats


def make_target(y, keys, target=None):
    target = target or TARGET
    if target == "raw":
        return np.array(y, np.float32)
    s = pd.Series(y)
    g = s.groupby(keys["date"].values)
    z = (s - g.transform("mean")) / (g.transform("std") + 1e-8)
    return np.array(z.clip(-5, 5).fillna(0.0), np.float32)       # 复制一份：pandas 3 返回只读数组


@torch.no_grad()
def predict_array(model, X, device, batch=4096):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(device).float()
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def predict_range(model, table, sd, ed, instruments, stats, device, norm=None):
    """推理：逐月切窗、标准化、预测，避免整段测试区间常驻内存。返回 DataFrame[date, instrument, factor]。"""
    backend = _backend(table)
    parts = []
    for X, _, keys in iter_chunks(backend, sd, ed, instruments, with_label=False, norm=norm):
        keys = keys.copy()
        keys["factor"] = predict_array(model, standardize(X, stats), device).astype(np.float64)
        parts.append(keys)
    if not parts:
        raise RuntimeError(f"predict_range 无样本 ({sd}~{ed})")
    return pd.concat(parts, ignore_index=True)


# ==== 模型存/读：文本 JSON（格式同 demo） ====
def save_model(ckpt, model_path=MODEL_PATH):
    tensors = {}
    for k, v in ckpt["state_dict"].items():
        t = v.detach().cpu()
        tensors[k] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape),
                      "data": t.reshape(-1).tolist()}
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = tensors
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


# ==== 训练并持久化 ====
def train_and_save(datasources, model_path=MODEL_PATH, backend=None, on_epoch_end=None):
    """在写死的训练区间上从零训练，把 权重 + 标准化统计 + 结构超参 一并存盘。

    backend：本地训练时传入数据后端；平台上留空，使用 dai。
    on_epoch_end(model, device, stats, epoch)：可选回调（本地用于验证集评估），平台上留空。"""
    backend = backend or DaiBackend(datasources["bar1m"])
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = pick_device()
    logger.info("训练设备", device=str(device))

    instruments = backend.pool(TRAIN_START, TRAIN_END)
    if MAX_TRAIN_INSTRUMENTS:
        instruments = instruments[:MAX_TRAIN_INSTRUMENTS]
    Xtr, ytr, ktr, stats = build_dataset(backend, TRAIN_START, TRAIN_END, "train", instruments)
    ytr = make_target(ytr, ktr)

    model = StockTransformer(**MODEL_CFG).to(device)
    logger.info("可训练参数量", n_params=sum(p.numel() for p in model.parameters()), samples=len(ytr))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    gen = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        model.train()
        t, tot, nb = time.time(), 0.0, 0
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(perm), BATCH):
            j = perm[i:i + BATCH]
            xb = Xt[j].to(device).float()
            yb = yt[j].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        logger.info("epoch 完成", epoch=ep + 1, loss=round(tot / max(nb, 1), 6), elapsed=round(time.time() - t, 1))
        if on_epoch_end is not None:
            on_epoch_end(model, device, stats, ep + 1)

    mean, std = stats
    save_model({
        "state_dict": model.state_dict(),
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "norm": NORM,
        "target": TARGET,
        "train_range": [TRAIN_START, TRAIN_END],
        "data_alignment": DATA_ALIGNMENT,
        "mean": np.asarray(mean, np.float32).tolist(),
        "std": np.asarray(std, np.float32).tolist(),
    }, model_path)
    logger.info("模型已保存", path=model_path, bytes=os.path.getsize(model_path))
    return model_path


if __name__ == "__main__":
    train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"})
