# 10-24：rel3 自检（CPU）：ob1 字段能从本地 1 分钟数据读到；rel3 下买一量／卖一量共用参照（两者变换后的均值差反映不平衡），rel2 不变。
import sys
import numpy as np
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission_next")
import Transformer_modelsave_train as T
from feather_data import FeatherBackend
T.set_fields("ob1")
fb = FeatherBackend()
lo, hi = "2023-03-01 00:00:00", "2023-03-10 23:59:59"
inst = fb.pool(lo, hi)[:200]
sec = fb.sections(lo, hi, inst)
bars = fb.bars("2023-02-20 00:00:00", hi, sorted(sec["instrument"].unique().tolist()), T.FEATURE_COLS + ["adjust_factor"])
print("ob1 字段缺失率", {c: round(float(bars[c].isna().mean()), 3) for c in ("bid_num_orders1", "ask_num_orders1")})
Xf, Xc, Xs, keys = T.build_windows(bars, sec, coarse_len=0, compute_chars=False)
a2, _ = T.transform(Xf, None, "rel2"); a3, _ = T.transform(Xf, None, "rel3")
F = T.FEATURE_COLS; ib, ia = F.index("bid_volume1"), F.index("ask_volume1"); nb, na = F.index("bid_num_orders1"), F.index("ask_num_orders1")
raw_imb = np.log(np.nanmean(Xf[..., ib], 1) + 1) - np.log(np.nanmean(Xf[..., ia], 1) + 1)
for nm, a in (("rel2", a2), ("rel3", a3)):
    d = np.nanmean(a[..., ib].astype(np.float32), 1) - np.nanmean(a[..., ia].astype(np.float32), 1)
    dn = np.nanmean(a[..., nb].astype(np.float32), 1) - np.nanmean(a[..., na].astype(np.float32), 1)
    ok = np.isfinite(d) & np.isfinite(raw_imb)
    print(nm, "变换后 买一量−卖一量 均值差 与 原始 log 不平衡 的相关 %.3f；笔数差标准差 %.3f" % (np.corrcoef(d[ok], raw_imb[ok])[0, 1], np.nanstd(dn)))
same = [i for i, c in enumerate(F) if c not in sum(T.NORM_SHARE, [])]
print("非共享字段 rel2 与 rel3 逐位相同：", bool(np.allclose(np.nan_to_num(a2[..., same].astype(np.float32)), np.nan_to_num(a3[..., same].astype(np.float32)))))
