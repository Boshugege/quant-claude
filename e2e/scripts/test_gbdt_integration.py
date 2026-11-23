# 10-25：树模型成员接入的平台路径自检（模拟 dai，CPU）：极小规模 train_and_save（含树）→ 读 JSON → notebook 同款推理；树开／关输出要不同。
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fake_dai")); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.environ.get("SUBDIR", os.path.join(HERE, "..", "submission")))
import dai  # noqa
import torch
import Transformer_modelsave_train as T
T.pick_device = lambda: torch.device("cpu")
T.ARCH, T.NORM, T.E04_NORM, T.TARGET = "e04", "rel2", "rel2", "cs_rank"
T.WIN_GROUPS = "shape,book,tod"; T.E04_CFG.update(n_win_feat=T.n_win_feat(), fuse=True, mtl=True, mtl_n=2)
T.E04_MTL_W, T.TARGET_AUX_W, T.TARGET_AUX_N, T.TRAIN_NEUTRAL, T.ZERO_0930, T.LOSS_SKIP_0930, T.OUT_NEUTRAL = 0.5, 0.5, 2, "proxy_ind", True, True, 1.0
T.TRAIN_START, T.TRAIN_END, T.MAX_TRAIN_INSTRUMENTS, T.EPOCHS, T.N_MODELS = "2020-02-01", "2020-02-29 23:59:59", 150, 1, 1
T.E04_GBDT = {"bags": 1, "trees": 20, "depth": 4, "min_leaf": 50, "win": "shape,book,book2,book3,tod,pos,state"}
os.makedirs("e2e/runs/test_gbdt", exist_ok=True)
p = T.train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"}, model_path="e2e/runs/test_gbdt/tiny.json")
ck = json.load(open(p))
print("JSON", os.path.getsize(p), "bytes；gbdt:", None if not ck.get("gbdt") else (len(ck["gbdt"]["models"]), len(ck["gbdt"]["models"][0]), len(ck["gbdt"]["edges"])))
ck = T.load_model(p, map_location="cpu")
T.configure_from_ckpt(ck)
models = T.build_models(ck, torch.device("cpu"))
stats = T.load_stats(ck)
inst = T.pool("2024-03-01 00:00:00", "2024-03-08 23:59:59")[:300]
a = T.e04_predict_range(models, "cpt_jyc_2026_stock_bar1m", "2024-03-01 00:00:00", "2024-03-08 23:59:59", inst, stats, torch.device("cpu"))
gk = T.GBDT_CK; T.GBDT_CK = None
b = T.e04_predict_range(models, "cpt_jyc_2026_stock_bar1m", "2024-03-01 00:00:00", "2024-03-08 23:59:59", inst, stats, torch.device("cpu"))
T.GBDT_CK = gk
m = a.merge(b, on=["date", "instrument"])
m = m[m["date"].dt.strftime("%H%M") != "0930"]
c = m.groupby("date")[["factor_x", "factor_y"]].corr(method="spearman").xs("factor_x", level=1)["factor_y"]
print("行", len(a), "列", list(a.columns), "nan", int(a["factor"].isna().sum()), "；树开／关截面秩相关均值 %.3f（应 < 1）" % c.mean())
print("FIELD_SET 恢复为", T.FIELD_SET, "E04_NORM", T.E04_NORM, "WIN_GROUPS", T.WIN_GROUPS)
