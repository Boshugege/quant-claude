# 用某次训练的 transformer_model.json 预测一整年（原始输出，不做输出中性化，保留四个风格特征列），供第二验证折使用。
# 用法：python e2e/scripts/pred_year.py <run_dir> <year>   ->  <run_dir>/pred_<year>.parquet
import os, sys
import pandas as pd
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
from feather_data import FeatherBackend
run, year = sys.argv[1], sys.argv[2]
ck = T.load_model(os.path.join(run, "transformer_model.json"))
T.configure_from_ckpt(ck)
T.LABEL_NEUTRAL = max(T.LABEL_NEUTRAL, 1e-9)  # 让 need_chars() 为真，以便输出风格特征列
dev = T.pick_device()
models, stats, fb = T.build_models(ck, dev), T.load_stats(ck), FeatherBackend()
parts = []
for lo, hi in T.month_chunks(f"{year}-01-01 00:00:00", f"{year}-12-31 23:59:59"):
    parts.append(T.predict_range(models, fb, str(lo), str(hi), fb.pool(str(lo), str(hi)), stats, dev, out_neutral=0, keep_chars=True, zero=False))
out = pd.concat(parts, ignore_index=True)
out.to_parquet(os.path.join(run, f"pred_{year}.parquet"))
print("saved", len(out), list(out.columns))
