# -*- coding: utf-8 -*-
"""把几个 h2_robust 导出的树模型合并成一个"平均模型"：树拼接、每棵树的贡献 ÷ 模型数（学习率 × 1/k）、base 取平均。
要求各模型的 feat_order 与 edges 完全相同（同样的特征、同样的拟合窗分箱）。用法：merge_gbdt.py 输出.json 输入1.json 输入2.json …"""
import json, sys
out, ins = sys.argv[1], sys.argv[2:]
J = [json.load(open(p, encoding="utf-8")) for p in ins]
for j in J[1:]:
    assert j["feat_order"] == J[0]["feat_order"], "feat_order 不同"
    assert j["edges"] == J[0]["edges"], "edges 不同"
k = len(J)
lr = [float(j["model"]["params"]["lr"]) for j in J]
assert len(set(lr)) == 1, lr
trees = []
for j in J:
    trees.extend(j["model"]["trees"])
P = dict(J[0]["model"]["params"]); P["lr"] = lr[0] / k; P["n_trees"] = len(trees)
P["depth"] = max(int(j["model"]["params"]["depth"]) for j in J)
P["merged_from"] = ins
M = dict(J[0]); M["model"] = {"params": P, "base": sum(float(j["model"]["base"]) for j in J) / k, "trees": trees}
json.dump(M, open(out, "w", encoding="utf-8"))
print("合并 %d 个模型 → %d 棵树，lr %.4f，depth %d，base %.5f" % (k, len(trees), P["lr"], P["depth"], M["model"]["base"]))
