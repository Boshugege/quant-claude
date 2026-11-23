# S17 冒烟测试：旧融合模型权重仍能严格载入（默认结构不变）；fuse_hidden / fuse_peer 两个新结构能前向，输出 4 列。
import sys
import torch
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T

ck = T.load_model("e2e/runs/S12_A2_fuse/transformer_model.json")
T.configure_from_ckpt(ck)
m0 = T.build_models(ck, torch.device("cpu"))[0]                       # strict load_state_dict
mc = ck["model_cfg"]
B = 50
xs = [torch.randn(B, mc["seq_len"], mc["n_feat"]), torch.randn(B, mc["intra_len"] + 2 * mc.get("slot_len", 0) + mc.get("patch_len", 0), mc["n_win_feat"]),
      torch.randn(B, mc["daily_len"], mc["n_win_feat"]), torch.randn(B, mc["peer_len"])]
print("旧结构", tuple(m0.eval()(*xs, [B]).shape))
for kw in (dict(fuse_hidden=256), dict(fuse_peer=True), dict(fuse_hidden=256, fuse_peer=True)):
    m = T.E04Net(**dict(mc, **kw)).eval()
    n = sum(p.numel() for p in m.fuse_head.parameters())
    print(kw, tuple(m(*xs, [B]).shape), "融合头参数", n)
print("ok")
