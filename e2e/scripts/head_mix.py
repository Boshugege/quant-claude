# 融合模型的推理期头部混合（只推理、不训练）：最终分 = (1 − w)·rank(融合头) + w·mean(rank(三个专家头))。
# 用法：python e2e/scripts/head_mix.py <run_dir（fold-A 融合模型）> <w1,w2,...> [daily]
#   第三个参数 daily：只混日线专家头（只用之前交易日的数据，同一交易日内几乎不变，作"日内不变锚"压换手；reports/15 S23）
#   -> <run_dir>/heads_<year>.parquet（四个头各自的截面居中排名，2023 与 2024）
#   -> <run_dir>/mix_w<w>/val_pred.parquet、pred_2024.parquet（与 cmp_runs.py 的两折口径一致）
import os, sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
from feather_data import FeatherBackend

run, ws = sys.argv[1], [float(x) for x in sys.argv[2].split(",")]
DAILY = len(sys.argv) > 3 and sys.argv[3] == "daily"
ck = T.load_model(os.path.join(run, "transformer_model.json"))
T.configure_from_ckpt(ck)
T.LABEL_NEUTRAL = max(T.LABEL_NEUTRAL, 1e-9)  # 让 need_chars() 为真，以便输出风格特征列
dev = T.pick_device()
models, stats, fb = T.build_models(ck, dev), T.load_stats(ck), FeatherBackend()
assert len(models) == 1 and getattr(models[0], "fuse", False), "只用于单个融合模型"
HEADS = []


@torch.inference_mode()
def capture(models_, X, keys, device):
    m = models_[0]
    order = np.argsort(keys["date"].values, kind="stable")
    dates = keys["date"].values[order]
    H = np.zeros((len(keys), 4))
    for a, b in T._segments(dates):
        idx = order[a:b]
        S = m(*T._e04_tensors(X, idx, device), [b - a]).float().cpu().numpy()
        H[idx] = pd.DataFrame(S).rank(pct=True).to_numpy() - 0.5
    HEADS.append(H)
    return H[:, 3]


T.e04_predict_array = capture
for year, fn in (("2023", "val_pred.parquet"), ("2024", "pred_2024.parquet")):
    HEADS.clear()
    parts = [T.predict_range(models, fb, str(lo), str(hi), fb.pool(str(lo), str(hi)), stats, dev, out_neutral=0, keep_chars=True, zero=False)
             for lo, hi in T.month_chunks(f"{year}-01-01 00:00:00", f"{year}-12-31 23:59:59")]
    d = pd.concat(parts, ignore_index=True)
    H = np.concatenate(HEADS)
    assert len(H) == len(d)
    for k in range(4):
        d[f"h{k}"] = H[:, k]
    d.to_parquet(os.path.join(run, f"heads_{year}.parquet"))
    ref = pd.read_parquet(os.path.join(run, fn), columns=["date", "instrument", "factor"])
    chk = d.merge(ref, on=["date", "instrument"], suffixes=("", "_ref"))
    print(year, "融合头与原预测的相关", round(float(np.corrcoef(chk["h3"], chk["factor_ref"])[0, 1]), 6), "行数", len(d), len(ref))
    base = d.drop(columns=[f"h{k}" for k in range(4)])
    for w in ws:
        o = os.path.join(run, f"{'dmix' if DAILY else 'mix'}_w{w:g}")
        os.makedirs(o, exist_ok=True)
        other = d["h2"] if DAILY else d[["h0", "h1", "h2"]].mean(1)
        base.assign(factor=(1 - w) * d["h3"] + w * other).to_parquet(os.path.join(o, fn))
print("done")
