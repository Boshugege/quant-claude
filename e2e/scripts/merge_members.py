# 把两个（或多个）同配置、同训练区间、不同种子的全量训练合并成一个种子集成（JSON 的 state_dicts 拼接）。
# 等价于 --n-models N --seed <第一个的种子> 的训练：第 k 个成员的种子必须是 SEED + k（打包后私榜重训按 SEED, SEED+1, … 训练）。
# 用法：python3 e2e/scripts/merge_members.py <输出 run 目录> <run_0> <run_1> [...]
import json, os, shutil, sys
out, runs = sys.argv[1], sys.argv[2:]
DEFAULTS = {"head_w": 1.0, "extra_times": [], "extra_members": [], "year_w": {}, "weight_ema": 0.0, "daily_mix": 0.0, "anchor_train": 0.0}                                     # 旧版本 JSON 里没有的字段按默认值比较
cks = [json.load(open(os.path.join(r, "transformer_model.json"), encoding="utf-8")) for r in runs]
CFG_DEFAULTS = {"fuse_hidden": 0, "fuse_peer": False, "tod_route": "token", "dropout": 0.1, "seq_enc": "tf"}       # 旧版本 model_cfg 里没有的结构字段（默认 = 旧结构）
for c in cks:
    c["model_cfg"] = dict(CFG_DEFAULTS, **c["model_cfg"])
cfgs = [json.load(open(os.path.join(r, "config.json"), encoding="utf-8")) for r in runs]
keys = set().union(*[set(c) for c in cks]) - {"state_dicts"}
import numpy as np
MIXED = os.environ.get("MIXED_EXTRA") == "1"                     # 10-23 混合集成：允许部分成员用中间时刻截面（其余配置必须一致）
STATS_FROM = int(os.environ.get("STATS_FROM", "0"))               # 10-24：合并后统一用哪个成员的统计量（混合集成用中间时刻成员的，与私榜一次性重训同口径）
for i, c in enumerate(cks):                                       # 标准化统计量来自按种子抽样，允许极小差异
    ref = cks[STATS_FROM]
    if MIXED and (c.get("extra_times") or []) != (ref.get("extra_times") or []):
        d = max(float(np.nanmax(np.abs(np.asarray(x, float) - np.asarray(y, float)))) for x, y in zip(ref["stats"], c["stats"]))
        print(f"成员 {i} 与统计量来源成员 {STATS_FROM} 的截面集合不同（时刻 one-hot 统计量本来就不同），最大差 {d:.4f}；改用来源成员的统计量（stats_check2 实测预测秩相关 ≥ 0.9994）")
        continue
    for x, y in zip(ref["stats"], c["stats"]):
        assert np.allclose(np.asarray(x, float), np.asarray(y, float), rtol=0, atol=float(os.environ.get("STATS_ATOL", "2e-3")), equal_nan=True), "stats 差异过大"
# E14（含时段 one-hot）的时段通道统计量按种子抽样差约 0.015；scripts/stats_check2.py 实测成员改用第一个成员的统计量后预测截面秩相关 ≥ 0.99986，可用 STATS_ATOL=0.02
keys = keys - {"stats"}
if MIXED:
    ets = [c.get("extra_times", []) for c in cks]
    used = [e for e in ets if e]
    assert used and all(e == used[0] for e in used), ("用中间时刻的成员必须用同一组时刻", ets)
    mixed_times, mixed_members = list(used[0]), [i for i, e in enumerate(ets) if e]
    keys = keys - {"extra_times", "extra_members"}
GRU = [i for i, c in enumerate(cks) if c["model_cfg"].get("seq_enc", "tf") == "gru"]   # 10-27：GRU 成员，结构只允许 seq_enc 不同
if GRU:
    base_cfg = {k_: v_ for k_, v_ in cks[0]["model_cfg"].items() if k_ != "seq_enc"}
    for c in cks:
        assert {k_: v_ for k_, v_ in c["model_cfg"].items() if k_ != "seq_enc"} == base_cfg, "GRU 成员除 seq_enc 外结构必须一致"
    keys = keys - {"model_cfg"}
keys = keys - {"member_cfgs", "gru_members"}                        # 每个 run 是单成员，按成员重新拼
member_cfgs = [dict(c["model_cfg"]) for c in cks]
for k in sorted(keys):
    vals = [c.get(k, DEFAULTS.get(k)) for c in cks]
    assert all(v == vals[0] for v in vals), f"字段 {k} 不一致"
seed0 = cfgs[0]["seed"]
for i, c in enumerate(cfgs):
    assert c["seed"] == seed0 + i, ("种子必须依次为 SEED + k", [c["seed"] for c in cfgs])
    assert c["n_models"] == 1
    assert c["train"] == cfgs[0]["train"], "训练区间不一致"
merged = dict(cks[0])
merged["stats"] = cks[STATS_FROM]["stats"]
merged.update({k: DEFAULTS[k] for k in DEFAULTS if k not in merged})
merged["state_dicts"] = [sd for c in cks for sd in c["state_dicts"]]
merged["member_cfgs"], merged["gru_members"] = member_cfgs, GRU
if GRU:
    print("GRU 成员", GRU)
if MIXED:
    merged["extra_times"], merged["extra_members"] = mixed_times, mixed_members
    print("混合集成：中间时刻成员", mixed_members, mixed_times)
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "transformer_model.json"), "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False)
cfg = dict(cfgs[0], run=os.path.basename(out.rstrip("/")), n_models=len(runs), merged_from=[os.path.basename(r.rstrip("/")) for r in runs])
json.dump(cfg, open(os.path.join(out, "config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
for fn in ("metrics.json",):
    if os.path.exists(os.path.join(runs[0], fn)):
        shutil.copy(os.path.join(runs[0], fn), os.path.join(out, fn))
print("merged", len(merged["state_dicts"]), "members ->", out, os.path.getsize(os.path.join(out, "transformer_model.json")), "bytes")
