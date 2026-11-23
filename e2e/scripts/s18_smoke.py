# S18 冒烟测试：按年份加权的训练路径能跑通；权重全为 1 时与不加权的损失逐位相同。
import sys
import numpy as np
import torch
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T

ck = T.load_model("e2e/runs/S12_A2_fuse/transformer_model.json")
T.configure_from_ckpt(ck)
mc = ck["model_cfg"]
T.EPOCHS, T.CKPT_DIR, T.SECS_PER_BATCH = 1, None, 2
rng = np.random.default_rng(0)
ns, per = 6, 40
n = ns * per
X = (rng.standard_normal((n, mc["seq_len"], mc["n_feat"])).astype(np.float32),
     rng.standard_normal((n, mc["intra_len"], mc["n_win_feat"])).astype(np.float32),
     rng.standard_normal((n, mc["daily_len"], mc["n_win_feat"])).astype(np.float32),
     rng.standard_normal((n, mc["peer_len"])).astype(np.float32))
y = rng.standard_normal(n).astype(np.float32)
sec_id = np.repeat(np.arange(ns), per).astype(np.int64)
sec_day = np.array([0, 0, 1, 1, 2, 2])
inst = np.tile(np.arange(per), ns).astype(np.int64)
dev = torch.device("cpu")
outs = {}
for nm, w in (("none", None), ("ones", np.ones(ns)), ("ramp", np.array([0.5, 0.5, 1.0, 1.0, 1.5, 1.5]))):
    r = T._e04_train_one(X, y, sec_id, 42, dev, member=0, style=None, inst=inst, sec_day=sec_day, sec_w=w)
    m = r[0] if isinstance(r, tuple) else r
    sd = m.state_dict() if hasattr(m, "state_dict") else r[0]
    outs[nm] = torch.cat([v.flatten().float() for v in sd.values()])
    print(nm, "ok")
print("none vs ones 最大差", float((outs["none"] - outs["ones"]).abs().max()))
print("none vs ramp 最大差", float((outs["none"] - outs["ramp"]).abs().max()))
