# -*- coding: utf-8 -*-
"""远端本地训练：直接调用提交脚本 Transformer_modelsave_train.train_and_save，数据换成 FeatherBackend。

训练逻辑与平台私榜重训完全同一份代码；本脚本只负责覆盖配置、逐 epoch 验证、JSON 重载回归、
以及训练结束后用本地评分器五项指标评估验证区间。

用法（远端 quant 目录）：
  python e2e/scripts/train_local.py --run smoke --train 2020-02-01 2020-02-29 --val 2020-03-01 2020-03-31 \
      --max-inst 120 --max-days 5 --epochs 1
产物：e2e/runs/<run>/{config.json, log.txt, metrics.json, val_pred.parquet, transformer_model.json}
"""
import argparse
import hashlib
import json
import os
try:
    import resource                                                         # Windows 没有（10-25：4090 上跑）
except ImportError:
    resource = None
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend, set_read_cols, IND_COLS  # noqa: E402
from metrics import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--train", nargs=2, default=["2020-01-01", "2022-12-31"])
ap.add_argument("--val", nargs=2, default=["2023-01-01", "2023-12-31"])
ap.add_argument("--max-inst", type=int, default=0)
ap.add_argument("--max-days", type=int, default=0)
ap.add_argument("--norm", default=T.NORM, choices=["raw", "log", "rel", "rel2", "rel3"])
ap.add_argument("--target", default=T.TARGET, choices=["raw", "cs_z", "cs_rank"])
ap.add_argument("--loss", default=T.LOSS, choices=["mse", "ic", "mse+ic"])
ap.add_argument("--secs-per-batch", type=int, default=T.SECS_PER_BATCH)
ap.add_argument("--fields", default=T.FIELD_SET, choices=list(T.FIELDS))
ap.add_argument("--coarse", type=int, default=T.COARSE_LEN, help="30m 回看根数，0 = 关闭")
ap.add_argument("--seq-len", type=int, default=T.SEQ_LEN)
ap.add_argument("--summary", action="store_true", help="加入截面摘要特征")
ap.add_argument("--summary-set", default=T.SUMMARY_SET, choices=["all", "rev"])
ap.add_argument("--summary-pool", action="store_true", help="摘要特征加到池化向量（序列保持 64 个 token）")
ap.add_argument("--label-neutral", type=float, default=T.LABEL_NEUTRAL)
ap.add_argument("--out-neutral", type=float, default=T.OUT_NEUTRAL)
ap.add_argument("--device", default="", help="强制设备（cpu/mps），测试用")
ap.add_argument("--epochs", type=int, default=T.EPOCHS)
ap.add_argument("--batch", type=int, default=T.BATCH)
ap.add_argument("--lr", type=float, default=T.LR)
ap.add_argument("--wd", type=float, default=T.WEIGHT_DECAY)
ap.add_argument("--sched", default=T.LR_SCHEDULE, choices=["none", "cos"])
ap.add_argument("--n-models", type=int, default=T.N_MODELS)
ap.add_argument("--seed", type=int, default=T.SEED)
ap.add_argument("--d-model", type=int, default=T.MODEL_CFG["d_model"])
ap.add_argument("--nlayers", type=int, default=T.MODEL_CFG["nlayers"])
ap.add_argument("--dim-ff", type=int, default=T.MODEL_CFG["dim_ff"])
ap.add_argument("--nhead", type=int, default=T.MODEL_CFG["nhead"])
ap.add_argument("--no-analyze", action="store_true", help="跳过训练后的本地评分器评估")
ap.add_argument("--arch", default=T.ARCH, choices=["v1", "e04"])
ap.add_argument("--train-neutral", default=T.TRAIN_NEUTRAL, choices=["none", "proxy", "proxy_ind", "official"],
                help="训练标签与损失的剔除：不剔除 / 行情代理特征 / 代理特征 + 伪行业（E07 起不再读官方风格暴露表）")
ap.add_argument("--pind-k", type=int, default=T.PIND_K)
ap.add_argument("--pind-days", type=int, default=T.PIND_DAYS)
ap.add_argument("--aux-w", type=float, default=T.TARGET_AUX_W, help="多周期目标权重（同日之后 --aux-n 个截面标签之和）")
ap.add_argument("--aux-n", type=int, default=T.TARGET_AUX_N)
ap.add_argument("--aux-decay", type=float, default=T.TARGET_AUX_DECAY, help="第 k 个之后截面的权重 = decay^(k-1)")
ap.add_argument("--zero-0930", action="store_true", help="推理时 09:30 截面输出常数 0（[val] 同口径）")
ap.add_argument("--skip-0930-loss", action="store_true", help="09:30 截面不参与训练")
ap.add_argument("--peer-daily", action="store_true", help="同伴日收益取自一次性构建的日线面板（可配合 --peer-len 240）")
ap.add_argument("--peer-len", type=int, default=T.E04_CFG["peer_len"])
ap.add_argument("--daily-len", type=int, default=T.E04_CFG["daily_len"], help="日线专家的交易日数（reports/15 T3）")
ap.add_argument("--pool", default="mean", choices=["mean", "attn", "mix"], help="各专家 token 的池化方式（10-20 结构探索）")
ap.add_argument("--peer-mode", default="attn", choices=["attn", "mean", "none"], help="同伴对比方式（10-21 简化消融）")
ap.add_argument("--drop-experts", default="", help="去掉的尺度专家：s／i／d 的组合（10-21 简化消融）")
ap.add_argument("--mtl-set", default="rv,lv", help="多任务目标：rv,lv,r60,r10,hi,lo,sv 的组合（10-21 R26）")
ap.add_argument("--mtl-w", type=float, default=0.0, help="多任务辅助损失权重（未来 30 分钟波动、量能；10-20 R22）")
ap.add_argument("--mgate", action="store_true", help="市场门控（10-20 结构探索）")
ap.add_argument("--short-layers", type=int, default=2, help="1 分钟专家的 Transformer 层数（10-20 结构探索）")
ap.add_argument("--sam-rho", type=float, default=0.0, help="SAM 扰动半径（10-20 结构探索）")
ap.add_argument("--xscale", action="store_true", help="三个专家池化表示做跨尺度注意力（10-20 结构探索）")
ap.add_argument("--xs-layers", type=int, default=0, help=">0：可学习的截面 Transformer 层数（S12 A1）")
ap.add_argument("--fuse", action="store_true", help="跨尺度融合头（S12 A2）")
ap.add_argument("--fuse-hidden", type=int, default=0, help="融合头加宽加深的隐层宽度（S17）")
ap.add_argument("--fuse-peer", action="store_true", help="融合头输入再拼同伴画像编码（S17）")
ap.add_argument("--weight-ema", type=float, default=0.0, help="权重滑动平均的衰减系数（如 0.995；S20）")
ap.add_argument("--extra-times", default="", help="只在训练时加入的中间时刻，如 09:45,10:15,10:45,11:15,13:15,13:45,14:15（S19）")
ap.add_argument("--boost-pred", default="", help="R5d 两阶段提升：第一阶段在训练区间上的预测（parquet 通配）")
ap.add_argument("--boost-beta", type=float, default=0.0, help="目标 = 原目标 − β × 第一阶段预测")
ap.add_argument("--slot-w", default="", help="按截面时刻的损失权重，如 14:30:0.5（R3）")
ap.add_argument("--year-w", default="", help="按年份的样本权重，如 2020:0.5,2021:0.83,2022:1.17,2023:1.5（S18）")
ap.add_argument("--head-w", type=float, default=1.0, help="融合模型里三个专家头的损失权重（S17）")
ap.add_argument("--tail-w", type=float, default=0.0, help="尾部加权相关损失权重（S12 A4）")
ap.add_argument("--listnet-w", type=float, default=0.0, help="列表式排序损失权重（ListNet，头尾对称；R1a）")
ap.add_argument("--dropout", type=float, default=0.1, help="E04 各层与融合头的 dropout（R1e）")
ap.add_argument("--e04-d", type=int, default=T.E04_CFG["d_model"], help="E04 的 d_model（S12 A5）")
ap.add_argument("--patch-len", type=int, default=T.E04_CFG.get("patch_len", 0), help=">0（取 42）：日内专家接当日 5 分钟 patch token（reports/15 T2）")
ap.add_argument("--anchor-train", type=float, default=0.0, help=">0：训练时合成分混入日线专家头（S24），推理同时用同样的 --daily-mix")
ap.add_argument("--daily-mix", type=float, default=0.0, help=">0：推理最终分混入日线专家头（日内不变锚，S23）")
ap.add_argument("--tod-route", default="token", choices=["token", "fuse", "film"], help="时段 one-hot 的走法（需 --fuse 且 win-groups 以 tod 结尾）：token / fuse / film；reports/15 S22")
ap.add_argument("--slot-len", type=int, default=T.E04_CFG.get("slot_len", 0), help=">0：日内专家接 2×N 个同时段 token（reports/15 T1）")
ap.add_argument("--win-feat", type=int, default=T.E04_CFG["n_win_feat"], choices=[4, 7],
                help="日内/日线专家的通道数：4 = 原始；7 = 加上影线占比、实体占比、平均单笔成交额")
ap.add_argument("--win-groups", default=None,
                help="日内/日线专家的通道组（逗号分隔，取自 shape,book,book2,book3,state,mkt,pos,slot,lead,xrank,tod）；给出时覆盖 --win-feat")
ap.add_argument("--disp-w", type=float, default=0.0, help="按截面标签离散度加权的幂次（10-20 压力对齐）")
ap.add_argument("--seq-enc", default="tf", choices=["tf", "gru"], help="尺度专家的序列编码器（10-27：gru = 不同模型族的集成成员）")
ap.add_argument("--pair-units", action="store_true", help="训练单元 = 同日相邻两截面（10-25，腿换手损失用）")
ap.add_argument("--leg-turn", type=float, default=0.0, help="腿换手损失权重（10-25）")
ap.add_argument("--leg-tau", type=float, default=0.15, help="软腿成员温度（10-25）")
ap.add_argument("--day-bag", type=float, default=0.0, help="每个成员随机去掉这一比例的交易日（R16 抽样集成）")
ap.add_argument("--day-batch", type=int, default=T.DAY_BATCH, help=">0：每批取若干完整交易日（换手惩罚需要）")
ap.add_argument("--turn-w", type=float, default=T.TURN_WEIGHT, help="换手惩罚权重")
ap.add_argument("--icir-w", type=float, default=T.ICIR_WEIGHT, help="IC 稳定性（批内截面 IC 标准差）权重")
ap.add_argument("--resume", action="store_true", help="从 runs/<run>/ckpt 的最新存档续训")
ap.add_argument("--no-cache", action="store_true", help="不读写数据集缓存（--measure 时自动关闭）")
ap.add_argument("--measure", action="store_true",
                help="合规测量：只执行平台上的 train_and_save（不建验证集、不评估、不用缓存），记录内存峰值与耗时")
args = ap.parse_args()
# 结束日期只给到日时补齐到当日收盘之后（ssh 传参不便携带空格）
args.train[1], args.val[1] = [d + " 23:59:59" if len(d) == 10 else d for d in (args.train[1], args.val[1])]

OUT = os.path.join("e2e", "runs", args.run)
CKPT = os.path.join(OUT, "ckpt")
if os.path.isdir(CKPT) and os.listdir(CKPT) and not args.resume:
    sys.exit(f"{CKPT} 已有存档：续训请加 --resume，重新训练请换一个 --run 名")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "log.txt"), "a" if args.resume else "w")


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


def rss_gb():
    if resource is None:                                                       # Windows：用 psutil 的峰值工作集
        try:
            import psutil
            mi = psutil.Process().memory_info()
            return getattr(mi, "peak_wset", mi.rss) / 1e9
        except Exception:
            return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # macOS: bytes


def cur_rss_gb():
    if resource is None:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e9
        except Exception:
            return float("nan")
    try:
        import subprocess
        return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) / 1e6
    except Exception:
        return float("nan")


class _Log(object):
    def info(self, msg, **kw):
        log(msg, " ".join(f"{k}={v}" for k, v in kw.items()), f"rss={cur_rss_gb():.1f}GB peak={rss_gb():.1f}GB")


# 覆盖提交脚本的配置（只改模块全局量）
T.logger = _Log()
T.set_fields(args.fields)
T.TRAIN_START, T.TRAIN_END = args.train
T.MAX_TRAIN_INSTRUMENTS = args.max_inst
T.NORM, T.TARGET = args.norm, args.target
T.E04_NORM = args.norm if args.norm in ("rel2", "rel3") else "rel2"                     # 10-24：e04 的 1 分钟输入变换（rel3 买卖同类字段共用参照）
T.LOSS, T.SECS_PER_BATCH = args.loss, args.secs_per_batch
T.SEQ_LEN, T.COARSE_LEN, T.SUMMARY = args.seq_len, args.coarse, args.summary
T.SUMMARY_SET, T.SUMMARY_POOL = args.summary_set, args.summary_pool
T.ARCH = args.arch
T.TRAIN_NEUTRAL, T.PIND_K, T.ZERO_0930 = args.train_neutral, args.pind_k, args.zero_0930
T.PIND_DAYS, T.TARGET_AUX_W, T.TARGET_AUX_N = args.pind_days, args.aux_w, args.aux_n
T.TARGET_AUX_DECAY = args.aux_decay
T.LOSS_SKIP_0930, T.DAY_BATCH, T.TURN_WEIGHT, T.ICIR_WEIGHT = args.skip_0930_loss, args.day_batch, args.turn_w, args.icir_w
T.PEER_FROM_DAILY = args.peer_daily
T.WIN_GROUPS = args.win_groups if args.win_groups is not None else ("shape" if args.win_feat == 7 else "")
assert all(g in T.WIN_GROUP_DIM for g in T.win_groups()), T.WIN_GROUPS
T.E04_CFG.update(seq_enc=args.seq_enc)
T.E04_CFG.update(peer_len=args.peer_len, daily_len=args.daily_len, slot_len=args.slot_len, patch_len=args.patch_len, xs_layers=args.xs_layers, pool=args.pool, xscale=args.xscale, mgate=args.mgate, mtl=args.mtl_w > 0, n_layers_short=args.short_layers, fuse=args.fuse,
                 peer_mode=args.peer_mode, drop=args.drop_experts, mtl_n=len(args.mtl_set.split(",")),
                 fuse_hidden=args.fuse_hidden, fuse_peer=args.fuse_peer, tod_route=args.tod_route,
                 d_model=args.e04_d, dim_ff=2 * args.e04_d, n_win_feat=T.n_win_feat(), dropout=args.dropout)
assert args.tod_route == "token" or (args.fuse and T.win_groups()[-1] == "tod"), "tod_route 需要 --fuse 且 win-groups 以 tod 结尾"
T.E04_TAIL_WEIGHT = args.tail_w
T.E04_LISTNET_WEIGHT = args.listnet_w
T.E04_HEAD_W = args.head_w
T.E04_WEIGHT_EMA = args.weight_ema
T.E04_SAM_RHO = args.sam_rho
T.E04_MTL_W = args.mtl_w
T.E04_MTL_SET = args.mtl_set
T.AUX_COLS = ["aux_" + c_ for c_ in args.mtl_set.split(",")]
assert all(c_ in ("rv", "lv", "r60", "r10", "hi", "lo", "sv", "au", "dr") for c_ in args.mtl_set.split(",")), args.mtl_set
T.E04_DAILY_MIX = args.daily_mix
T.E04_ANCHOR_TRAIN = args.anchor_train
if args.anchor_train > 0 and args.daily_mix == 0:
    T.E04_DAILY_MIX = args.anchor_train                                   # 学习式锚：推理用同样的混合权重
T.EXTRA_TIMES = [t for t in args.extra_times.split(",") if t]
T.DAY_BAG = args.day_bag
T.PAIR_UNITS, T.E04_LEG_TURN, T.E04_LEG_TAU = args.pair_units, args.leg_turn, args.leg_tau
T.DISP_W = args.disp_w
T.YEAR_W = {int(a.split(":")[0]): float(a.split(":")[1]) for a in args.year_w.split(",") if a}
T.SLOT_W = {a.rsplit(":", 1)[0]: float(a.rsplit(":", 1)[1]) for a in args.slot_w.split(",") if a}
T.BOOST_PRED, T.BOOST_BETA = args.boost_pred, args.boost_beta
if args.peer_len > 60 and not args.peer_daily:
    sys.exit("--peer-len > 60 需要 --peer-daily（逐月窗口表只回看 130 个自然日）")
if T.TURN_WEIGHT > 0 and T.DAY_BATCH <= 0:
    sys.exit("--turn-w 需要 --day-batch > 0（相邻截面要在同一批里）")
T.LABEL_NEUTRAL, T.OUT_NEUTRAL = args.label_neutral, args.out_neutral
if args.device:
    T.pick_device = lambda: torch.device(args.device)
T.EPOCHS, T.BATCH, T.LR, T.SEED = args.epochs, args.batch, args.lr, args.seed
T.WEIGHT_DECAY, T.LR_SCHEDULE, T.N_MODELS = args.wd, args.sched, args.n_models
T.MODEL_CFG = dict(n_feat=T.N_FEAT, d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers,
                   dim_ff=args.dim_ff, seq_len=T.SEQ_LEN, coarse_len=T.COARSE_LEN,
                   n_summary=len(T.summary_names()) if T.SUMMARY else 0, summary_pool=T.SUMMARY_POOL)
T.CKPT_DIR = None if args.measure else CKPT
set_read_cols(T.FEATURE_COLS)
backend = FeatherBackend(max_days=args.max_days)

# 数据集缓存：同一配置的训练/验证集只构建一次（崩溃后续训直接读盘，内存映射方式加载）
CACHE_VERSION = "v5"   # v5：E07 起数据集不再含官方风格列（训练剔除特征在 train_and_save 里按日级代理表另行计算）
_orig_build = T.build_dataset


def cached_build(table, sd, ed, mode, instruments, stats=None):
    if args.no_cache or args.measure:
        return _orig_build(table, sd, ed, mode, instruments, stats)
    spec = {"v": CACHE_VERSION, "fields": T.FIELD_SET, "L": T.SEQ_LEN, "Lc": T.COARSE_LEN, "norm": T.NORM,
            "summary": T.summary_names() if T.SUMMARY else [], "chars": T.need_chars(), "arch": T.ARCH,
            "e04": ({k: v for k, v in T.E04_CFG.items() if k not in ("xs_layers", "fuse", "fuse_hidden", "fuse_peer", "tod_route", "d_model", "dim_ff", "nhead", "n_layers_short", "n_layers_long")}
                    if T.ARCH == "e04" else None),   # 只与网络有关的参数不影响数据集，不进缓存键 "peer_daily": T.PEER_FROM_DAILY, "win_groups": T.WIN_GROUPS,
            "sd": sd, "ed": ed, "mode": mode, "max_days": args.max_days,
            "inst": hashlib.md5(",".join(instruments).encode()).hexdigest(),
            "stats": None if stats is None else hashlib.md5(np.concatenate(
                [a for a in stats if a is not None]).tobytes()).hexdigest()}
    if T.EXTRA_TIMES and mode == "train":                                  # S19：只有训练集带中间时刻（不设时缓存键与以前相同）
        spec["extra"] = list(T.EXTRA_TIMES)
    key = hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    d = os.path.join("e2e", "cache", key)
    if os.path.exists(os.path.join(d, "DONE")):
        log(f"读取数据集缓存 {d} ({mode} {sd[:10]}~{ed[:10]})")
        meta = json.load(open(os.path.join(d, "spec.json")))
        X = tuple(np.load(os.path.join(d, f"X{j}.npy"), mmap_mode="r") if os.path.exists(os.path.join(d, f"X{j}.npy"))
                  else None for j in range(meta["n_x"]))
        y = np.load(os.path.join(d, "y.npy")) if os.path.exists(os.path.join(d, "y.npy")) else None
        keys = pd.read_parquet(os.path.join(d, "keys.parquet"))
        z = np.load(os.path.join(d, "stats.npz"))
        st = tuple(z[f"s{j}"] if z[f"s{j}"].size else None for j in range(meta["n_stats"]))
        return X, y, keys, st
    X, y, keys, st = _orig_build(table, sd, ed, mode, instruments, stats)
    os.makedirs(d, exist_ok=True)
    for j, a in enumerate(X):
        if a is not None:
            np.save(os.path.join(d, f"X{j}.npy"), a)
    if y is not None:
        np.save(os.path.join(d, "y.npy"), y)
    keys.to_parquet(os.path.join(d, "keys.parquet"))
    e = np.zeros(0, np.float32)
    np.savez(os.path.join(d, "stats.npz"), **{f"s{j}": (e if a is None else a) for j, a in enumerate(st)})
    spec = dict(spec, n_x=len(X), n_stats=len(st))
    json.dump(spec, open(os.path.join(d, "spec.json"), "w"), indent=1)
    open(os.path.join(d, "DONE"), "w").write("ok")
    log(f"已写入数据集缓存 {d}")
    return X, y, keys, st


T.build_dataset = cached_build

_orig_e04_build = T.e04_build_dataset


def cached_e04_build(table, sd, ed, mode, instruments, stats=None):
    """10-25：e04 数据集缓存（同一数据配置的实验不再重建，4090 上每月构建约 20 秒）。键只含影响数据的设置。"""
    if args.no_cache or args.measure or mode != "train":                      # 验证／测试集已由 cached_build 缓存
        return _orig_e04_build(table, sd, ed, mode, instruments, stats)
    dk = ("intra_len", "daily_len", "peer_len", "slot_len", "patch_len", "n_win_feat")
    spec = {"v": "e04v1", "fields": T.FIELD_SET, "norm": T.E04_NORM, "L": T.SEQ_LEN, "cfg": {k: T.E04_CFG.get(k) for k in dk},
            "win": T.WIN_GROUPS, "wlb": T.WIN_LOOKBACK_DAYS, "peer_daily": T.PEER_FROM_DAILY,
            "mtl": list(T.AUX_COLS) if (mode == "train" and T.E04_MTL_W > 0) else [],
            "extra": list(T.EXTRA_TIMES) if mode == "train" else [], "sd": sd, "ed": ed, "mode": mode, "max_days": args.max_days,
            "inst": hashlib.md5(",".join(instruments).encode()).hexdigest(),
            "stats": None if stats is None else hashlib.md5(np.concatenate([np.asarray(a).ravel() for a in stats if a is not None]).tobytes()).hexdigest()}
    key = "e04_" + hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    d = os.path.join("e2e", "cache", key)
    if os.path.exists(os.path.join(d, "DONE")):
        log(f"读取 e04 数据集缓存 {d} ({mode} {sd[:10]}~{ed[:10]})")
        meta = json.load(open(os.path.join(d, "spec.json")))
        X = tuple(np.load(os.path.join(d, f"X{j}.npy")) for j in range(meta["n_x"]))
        y = np.load(os.path.join(d, "y.npy")) if os.path.exists(os.path.join(d, "y.npy")) else None
        keys = pd.read_parquet(os.path.join(d, "keys.parquet"))
        z = np.load(os.path.join(d, "stats.npz"))
        st = tuple(z[f"s{j}"] for j in range(meta["n_stats"]))
        return X, y, keys, st
    X, y, keys, st = _orig_e04_build(table, sd, ed, mode, instruments, stats)
    os.makedirs(d, exist_ok=True)
    for j, a in enumerate(X):
        np.save(os.path.join(d, f"X{j}.npy"), a)
    if y is not None:
        np.save(os.path.join(d, "y.npy"), y)
    keys.to_parquet(os.path.join(d, "keys.parquet"))
    np.savez(os.path.join(d, "stats.npz"), **{f"s{j}": np.asarray(a) for j, a in enumerate(st)})
    json.dump(dict(spec, n_x=len(X), n_stats=len(st)), open(os.path.join(d, "spec.json"), "w"), indent=1)
    open(os.path.join(d, "DONE"), "w").write("ok")
    log(f"已写入 e04 数据集缓存 {d}")
    return X, y, keys, st


T.e04_build_dataset = cached_e04_build
json.dump({**vars(args), "model_cfg": T.MODEL_CFG, "torch": torch.__version__, "device": str(T.pick_device()),
           "label": "local y = close/VWAP of window (t,t+30] - 1, rebuilt from 1m (platform: vwap_return)",
           "features": T.FEATURE_COLS}, open(os.path.join(OUT, "config.json"), "w"), indent=2, ensure_ascii=False)

# 只用于本地 [val] 的平台口径评估（官方 10 风格 + 行业剔除）；提交代码不读这张表
STYLE_COLS = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
VAL = {}
history = []


def val_inst():
    inst = backend.pool(args.val[0], args.val[1])
    return inst[:args.max_inst] if args.max_inst else inst


def on_data_ready(stats):
    _ex, T.EXTRA_TIMES = T.EXTRA_TIMES, []                                 # S19：验证集只用官方截面（中间时刻只加在训练集）
    X, y, k, _ = T.build_dataset(backend, args.val[0], args.val[1], "train", val_inst(), stats)
    T.EXTRA_TIMES = _ex
    VAL.update(X=X, y=y, k=k, stats=stats, style=None)
    try:  # 平台口径的 [val]：当日 10 个官方风格暴露（只用于评估）
        ex = backend.exposure(args.val[0], args.val[1], sorted(k["instrument"].unique().tolist()))
        kd = k[["date", "instrument"]].assign(_day=k["date"].dt.normalize()).merge(
            ex.rename(columns={"date": "_day"}), on=["_day", "instrument"], how="left")
        VAL["style"] = kd[STYLE_COLS + IND_COLS].to_numpy(np.float64)     # 平台评测：10 个风格 + 行业
        log(f"[val] 平台口径：官方风格暴露覆盖率 {np.isfinite(VAL['style']).all(1).mean():.3f}")
    except Exception as e:  # 暴露数据缺失时只报原始口径
        log(f"[val] 官方风格暴露不可用，只报原始口径：{e!r}")


def on_epoch_end(model, device, member, ep):
    t0 = time.time()
    pv = T.predict_any(model, VAL["X"], VAL["k"], device)
    on = T.OUT_NEUTRAL if T.need_chars() else 0.0          # 与交付推理一致：先做输出中性化，再按平台口径剔除风格
    pf = T.neutralize(pv, VAL["k"][T.CHAR_COLS].to_numpy(), VAL["k"]["date"].values, on) if on > 0 else pv
    if T.ZERO_0930:
        pf = T.zero_0930(pf, VAL["k"]["date"])
    raw = evaluate(VAL["k"], pv, VAL["y"])
    m = evaluate(VAL["k"], pf, VAL["y"], style=VAL.get("style")) if VAL.get("style") is not None else raw
    history.append({"member": member, "epoch": ep, **m, **{f"raw_{a}": b for a, b in raw.items()}})
    tag = f"平台口径(输出中性化{on:g}{'+09:30置零' if T.ZERO_0930 else ''}+风格剔除)" if VAL.get("style") is not None else "原始口径"
    log(f"[val] member {member} epoch {ep} {tag}: ic={m['ic']:.4f} icir={m['icir']:.3f} ls={m['ls_sharpe']:.2f} "
        f"stress={m['stress']:.3f} turn={m['turnover']:.3f} | 原始: ic={raw['ic']:.4f} icir={raw['icir']:.3f} "
        f"| pred_std={pv.std():.3g} nan={int(np.isnan(pv).sum())} eval={time.time() - t0:.0f}s rss={rss_gb():.1f}GB")


t_all = time.time()
mpath = os.path.join(OUT, "transformer_model.json")
if args.measure:
    T.train_and_save({}, model_path=mpath, backend=backend)
    size = os.path.getsize(mpath)
    res = {"total_sec": time.time() - t_all, "peak_rss_gb": rss_gb(), "json_bytes": size,
           "epochs": args.epochs, "train": args.train}
    json.dump(res, open(os.path.join(OUT, "measure.json"), "w"), indent=2)
    log(f"[measure] 平台流程 train_and_save 完成：总耗时 {res['total_sec'] / 60:.1f} 分钟，"
        f"进程内存峰值 {res['peak_rss_gb']:.1f} GB，JSON {size / 2**20:.2f} MiB")
    log("done")
    sys.exit(0)
T.train_and_save({}, model_path=mpath, backend=backend, on_epoch_end=on_epoch_end, on_data_ready=on_data_ready)

# JSON 重载回归：CPU 与训练设备分别重建，预测对照
size = os.path.getsize(mpath)
ck = T.load_model(mpath)
T.configure_from_ckpt(ck)
stats = T.load_stats(ck)
assert np.allclose(stats[0], VAL["stats"][0]) and len(stats) == len(VAL["stats"])
dev = T.pick_device()
m_dev, m_cpu = T.build_models(ck, dev), T.build_models(ck, torch.device("cpu"))
n = min(len(VAL["y"]), 20000)
Xn = tuple(a[:n] if a is not None else None for a in VAL["X"])
kn = VAL["k"].iloc[:n]
diff = float(np.abs(T.predict_any(m_dev, Xn, kn, dev) - T.predict_any(m_cpu, Xn, kn, torch.device("cpu"))).max())
log(f"json={mpath} bytes={size:,} ({size / 2**20:.2f} MiB) <50MB={size < 50 * 2**20} "
    f"models={len(ck['state_dicts'])} reload_maxdiff={diff:.2e}")

# 推理路径：验证区间全部截面的原始预测（不平滑），并核对与训练路径一致
pred = T.predict_range(m_dev, backend, args.val[0], args.val[1], val_inst(), stats, dev, out_neutral=0,
                       keep_chars=T.need_chars(), zero=False)
ref = VAL["k"][["date", "instrument"]].assign(ref=T.predict_any(m_dev, VAL["X"], VAL["k"], dev))
j = pred.merge(ref, on=["date", "instrument"], how="inner")
log(f"predict_range rows={len(pred):,} matched={len(j):,} maxdiff={np.abs(j['factor'] - j['ref']).max():.2e}")
pred.to_parquet(os.path.join(OUT, "val_pred.parquet"))                    # 原始预测（未做输出中性化），含风格特征列

result = {"history": history, "json_bytes": size, "reload_maxdiff": diff, "max_rss_gb": rss_gb()}
if not args.no_analyze and not args.max_inst:
    from analyze_preds import analyze
    vs, ve = args.val[0][:7], args.val[1][:7]
    base = pred[["date", "instrument", "factor"]]
    t = analyze(base, vs, ve, args.run)
    log("analysis（原始输出）\n" + t.round(4).to_string())
    if T.need_chars():
        ch = pred[T.CHAR_COLS].to_numpy()
        expo = {}
        for name, f in [("raw", pred["factor"].to_numpy())] + [
                (f"neut{fr}", T.neutralize(pred["factor"].to_numpy(), ch, pred["date"].values, fr)) for fr in (1.0, 1.5)]:
            df = pd.DataFrame({"d": pred["date"].values, "f": f}).join(pd.DataFrame(ch, columns=T.CHAR_COLS))
            expo[name] = {c: round(df.groupby("d").apply(lambda g: g["f"].rank().corr(g[c].rank())).mean(), 3)
                          for c in T.CHAR_COLS}
            if name != "raw":
                tn = analyze(base.assign(factor=f), vs, ve, f"{args.run}_{name}", refs=())
                log(f"analysis（输出中性化 {name}）\n" + tn.round(4).to_string())
                result[f"analysis_{name}"] = tn.reset_index().to_dict(orient="records")
        log("风格暴露（与四个特征的平均截面秩相关）: " + json.dumps(expo, ensure_ascii=False))
        result["style_exposure"] = expo
    result["analysis"] = t.reset_index().to_dict(orient="records")
result["total_sec"] = time.time() - t_all
json.dump(result, open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=float)
log("done", f"{time.time() - t_all:.0f}s")
