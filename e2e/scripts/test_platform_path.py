# -*- coding: utf-8 -*-
"""用模拟 dai 跑通提交包：DaiBackend 训练（极小规模）-> JSON -> 执行 predict notebook 的 main -> 检查输出合规。

用法：python e2e/scripts/test_platform_path.py <run_dir_with_json 或 空> [start] [end]
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fake_dai"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
work = os.path.join("e2e", "runs", "platform_test")
os.makedirs(work, exist_ok=True)
for f in ["Transformer_modelsave_train.py", "Transformer_modelsave_predict.ipynb"]:
    shutil.copy(os.path.join(HERE, "..", "submission", f), work)
sys.path.insert(0, work)
import Transformer_modelsave_train as T  # noqa: E402
import dai  # noqa: E402

src_json = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
start, end = (sys.argv[2:4] + ["2024-01-01 00:00:00", "2024-01-31 23:59:59"][len(sys.argv[2:4]):])[:2]
if src_json:
    shutil.copy(src_json, T.MODEL_PATH)
else:   # 极小规模走一遍 DaiBackend 训练路径
    T.TRAIN_START, T.TRAIN_END, T.MAX_TRAIN_INSTRUMENTS, T.EPOCHS = "2020-02-01", "2020-02-29 23:59:59", 150, 1
    T.N_MODELS = 2
    T.set_fields(os.environ.get("FIELDS", "base"))
    if os.environ.get("ARCH") == "e04":         # E04 结构：按环境变量设置窗口通道组等（与 train_local 同口径）
        T.ARCH, T.NORM, T.TRAIN_NEUTRAL, T.ZERO_0930, T.LOSS_SKIP_0930 = "e04", "rel2", "proxy_ind", True, True
        T.SEQ_LEN, T.COARSE_LEN, T.SUMMARY, T.OUT_NEUTRAL = 64, 0, False, 1.0
        T.WIN_GROUPS = os.environ.get("WIN_GROUPS", "shape")
        T.E04_CFG.update(daily_len=int(os.environ.get("DAILY_LEN", T.E04_CFG["daily_len"])),
                         slot_len=int(os.environ.get("SLOT_LEN", T.E04_CFG.get("slot_len", 0))),
                         patch_len=int(os.environ.get("PATCH_LEN", T.E04_CFG.get("patch_len", 0))),
                         xs_layers=int(os.environ.get("XS_LAYERS", 0)), fuse=os.environ.get("FUSE", "0") == "1", n_win_feat=T.n_win_feat())
        T.E04_TAIL_WEIGHT = float(os.environ.get("TAIL_W", "0"))
        T.TARGET_AUX_W, T.PIND_DAYS = float(os.environ.get("AUX_W", "0")), int(os.environ.get("PIND_DAYS", T.PIND_DAYS))
    T.train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"})
print("train/vwap queries:", [q for q in dai.LOG if q[0] != "cpt_jyc_2026_stock_bar1m"][:4])

nb = json.load(open(os.path.join(work, "Transformer_modelsave_predict.ipynb")))
ns = {"__name__": "nb"}
exec(compile("".join(nb["cells"][1]["source"]), "predict.ipynb", "exec"), ns)
t0 = time.time()
out = ns["main"]({"bar1m": "cpt_jyc_2026_stock_bar1m"}, start, end)
el = time.time() - t0
print(out.head())
print(f"rows={len(out):,} elapsed={el:.0f}s columns={list(out.columns)}")
hm = sorted(out["date"].dt.strftime("%H:%M").unique())
print("times:", hm)
cnt = out.groupby("date").size()
expected = dai.query("SELECT date, instrument FROM cpt_jyc_2026_instruments", filters={"date": [start, end]}).df().groupby("date").size()
miss = 1 - (cnt.reindex(expected.index).fillna(0) / expected)
print(f"dup keys={out.duplicated(['date', 'instrument']).sum()} inf={np.isinf(out['factor']).sum()} "
      f"nan={out['factor'].isna().sum()} max_missing_rate={miss.max():.4f} sections={len(expected)}")
print("factor std by slot:", out.groupby(out['date'].dt.strftime('%H:%M'))['factor'].std().round(4).to_dict())
