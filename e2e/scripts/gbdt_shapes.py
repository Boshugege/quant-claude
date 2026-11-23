# 10-24：树模型成员的准备——看 e04 数据集各张量的形状与通道（一个月，CPU）
import sys
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
T.WIN_GROUPS = "shape,book,tod"; T.E04_CFG["n_win_feat"] = T.n_win_feat(); T.E04_CFG["fuse"] = True
from feather_data import FeatherBackend
fb = FeatherBackend()
inst = fb.pool("2022-03-01 00:00:00", "2022-03-31 23:59:59")[:300]
X, y, keys, stats = T.e04_build_dataset(fb, "2022-03-01 00:00:00", "2022-03-31 23:59:59", "train", inst)
for nm, a in zip(("Xf", "Xi", "Xd", "Xp"), X):
    print(nm, a.shape, a.dtype)
print("keys", keys.columns.tolist(), len(keys), "y", None if y is None else y.shape)
print("win groups", T.win_groups(), "n_win_feat", T.n_win_feat(), "fields", T.FEATURE_COLS)
