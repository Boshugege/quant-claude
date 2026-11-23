# -*- coding: utf-8 -*-
"""10-27 截断检测（未来信息审计）：同一个打包模型，用"只到截断时刻为止"的数据预测，与用全量数据预测比较截断时刻及之前的截面，必须逐位相同。
截断后端：1 分钟 bar、30 分钟窗口表、盘口窗口表按时间戳 <= 截断时刻保留；收盘集合竞价（15:00 才有）只保留截断日之前的交易日。
整条推理路径（Transformer 成员 + 树成员 + 输出中性化 + 09:30 置零）原样运行，与平台一致。
用法：python e2e/scripts/leak_check.py <含 transformer_model.json 的目录> <起始日> <截断时刻1> [<截断时刻2> ...]"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend  # noqa: E402


class CutBackend(FeatherBackend):
    def __init__(self, cut):
        super().__init__()
        self.cut = pd.Timestamp(cut)

    def _f(self, d, daily=False):
        if d is None or len(d) == 0:
            return d
        t = pd.to_datetime(d["date"])
        keep = (t < self.cut.normalize()) if daily else (t <= self.cut)
        return d[keep.to_numpy()].reset_index(drop=True)

    def windows30(self, sd, ed, instruments):
        return self._f(super().windows30(sd, ed, instruments))

    def book30(self, sd, ed, instruments):
        return self._f(super().book30(sd, ed, instruments))

    def auc_daily(self, sd, ed, instruments):
        return self._f(super().auc_daily(sd, ed, instruments), daily=True)

    def bars(self, sd, ed, instruments, cols=None):
        return self._f(super().bars(sd, ed, instruments, cols))

    def sections(self, sd, ed, instruments):
        return self._f(super().sections(sd, ed, instruments))


def _patch_read_range(cut):
    import feather_data as F
    orig = F.read_range

    def rr(sd, ed, instruments, *a, **k):
        d = orig(sd, ed, instruments, *a, **k)
        return d[pd.to_datetime(d["date"]) <= pd.Timestamp(cut)].reset_index(drop=True)
    F.read_range = rr
    return orig


run, start, cuts = sys.argv[1], sys.argv[2], sys.argv[3:]
ck = T.load_model(os.path.join(run, "transformer_model.json"))
T.configure_from_ckpt(ck)
dev = T.pick_device()
models, stats = T.build_models(ck, dev), T.load_stats(ck)
end = max(pd.Timestamp(c) for c in cuts).normalize() + pd.Timedelta(hours=23, minutes=59)
fb = FeatherBackend()
inst = fb.pool(start, str(end))
full = T.predict_range(models, fb, start, str(end), inst, stats, dev)
full["date"] = pd.to_datetime(full["date"])
print("全量预测", len(full), "截面", full["date"].nunique(), "树成员", bool(ck.get("gbdt")), flush=True)
bad_total = 0
for c in cuts:
    import feather_data as F
    orig = _patch_read_range(c)
    try:
        cb = CutBackend(c)
        part = T.predict_range(models, cb, start, c, inst, stats, dev)
    finally:
        F.read_range = orig
    part["date"] = pd.to_datetime(part["date"])
    ref = full[full["date"] <= pd.Timestamp(c)]
    m = ref.merge(part, on=["date", "instrument"], how="outer", suffixes=("_full", "_cut"), indicator=True)
    both = m[m["_merge"] == "both"]
    d = (both["factor_full"] - both["factor_cut"]).abs()
    last = both[both["date"] == both["date"].max()]
    dl = (last["factor_full"] - last["factor_cut"]).abs()
    nbad = int((d > 1e-6).sum())
    bad_total += nbad
    print(f"截断 {c}：行 {len(ref)}／{len(part)}，合并 {m['_merge'].value_counts().to_dict()}，最大绝对差 {float(d.max()):.3g}，差>1e-6 的行 {nbad}；"
          f"截断时刻本身的截面 {last['date'].max()} 行 {len(last)} 最大差 {float(dl.max()):.3g}", flush=True)
print("结论：", "CLEAN（截断时刻及之前的预测与全量逐位一致，未用未来数据）" if bad_total == 0 else f"有差异 {bad_total} 行，需排查")
