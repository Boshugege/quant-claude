# -*- coding: utf-8 -*-
"""公榜估分重做（10-22，用户："你是否打算再次调整一下本地的公榜估分策略？"）。
旧估分（评分机 refs.py）：候选在 52 个旧因子（参照池）里的五项百分位等权 → 线性拟合公榜分。参照池太弱，候选四项都封顶，只剩换手在动，
于是一味奖励压换手（本地 1647 > 1645 > 1643，公榜正好相反）。
新做法第二步（本文件）：用我们自己 38 个有公榜五项的提交，拟合"公榜五项 → 公榜分"：分数 = 五项各自的单调百分位曲线之和 / 5
（加性保序回归，反复回代），相当于反推榜上其他队伍在每一项上的分布。留一法检验。
用法：python3 c110_public_model.py <public_rows.json>"""
import json
import sys
import numpy as np


class IsotonicRegression:
    """加权保序回归（PAV），预测时按分段线性插值、两端截平。"""
    def __init__(self, increasing=True, out_of_bounds="clip"):
        self.inc = increasing

    def fit(self, x, y, sample_weight):
        o = np.argsort(x, kind="mergesort")
        xs, ys, ws = np.asarray(x, float)[o], np.asarray(y, float)[o], np.asarray(sample_weight, float)[o]
        if not self.inc:
            ys = -ys
        blocks = []                                                          # [加权均值, 权重, 起, 止]
        for i in range(len(xs)):
            blocks.append([ys[i], ws[i], i, i])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                a, b = blocks[-2], blocks.pop()
                tw = a[1] + b[1]
                a[0], a[1], a[3] = (a[0] * a[1] + b[0] * b[1]) / tw, tw, b[3]
        fit = np.empty(len(xs))
        for v, _, i0, i1 in blocks:
            fit[i0:i1 + 1] = v
        if not self.inc:
            fit = -fit
        ux, idx = np.unique(xs, return_index=True)
        self.x_, self.y_ = ux, np.array([fit[xs == u].mean() for u in ux])
        return self

    def predict(self, x):
        return np.interp(np.asarray(x, float), self.x_, self.y_)

R = json.load(open(sys.argv[1], encoding="utf-8"))
ids = sorted(R, key=int)
K = ["ic", "ir", "to", "ls", "st"]
X = np.array([[R[i][k] for k in K] for i in ids], float)
y = np.array([R[i]["score"] for i in ids], float)
w = np.array([3.0 if int(i) >= 1600 else 1.0 for i in ids])                     # 近期的提交权重大（榜在变）


def fit(X, y, w, it=60):
    fs = [IsotonicRegression(increasing=(k != "to"), out_of_bounds="clip") for k in K]
    parts = np.zeros_like(X)
    b = float(np.average(y, weights=w))
    for _ in range(it):
        for j in range(len(K)):
            r = y - b - parts.sum(1) + parts[:, j]
            fs[j].fit(X[:, j], r, sample_weight=w)
            parts[:, j] = fs[j].predict(X[:, j])
            parts[:, j] -= np.average(parts[:, j], weights=w)
    return b, fs


def predict(model, Xn):
    b, fs = model
    return b + sum(fs[j].predict(Xn[:, j]) for j in range(len(K)))


m = fit(X, y, w)
p = predict(m, X)
print("样本内：误差绝对值均值 %.4f，相关 %.3f" % (np.abs(p - y).mean(), np.corrcoef(p, y)[0, 1]))
loo = np.array([predict(fit(np.delete(X, i, 0), np.delete(y, i), np.delete(w, i)), X[i:i + 1])[0] for i in range(len(y))])
print("留一法：误差绝对值均值 %.4f（1600 之后 %.4f），相关 %.3f" % (np.abs(loo - y).mean(), np.abs(loo - y)[[int(i) >= 1600 for i in ids]].mean(), np.corrcoef(loo, y)[0, 1]))
for i, idn in enumerate(ids):
    if int(idn) >= 1600:
        print("%5s 公榜 %.4f 留一 %.4f 差 %+.4f | IC %.4f IR %.3f 换手 %.3f 多空 %.1f 压力 %.3f" % (idn, y[i], loo[i], loo[i] - y[i], *X[i]))
# 各项曲线在关键区间的斜率（每单位变化带来的分数）
b, fs = m
for j, k in enumerate(K):
    grid = {"ic": [0.085, 0.09, 0.095, 0.1], "ir": [1.6, 1.7, 1.8, 1.9, 2.0], "to": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5], "ls": [43, 48, 53, 58], "st": [1.5, 1.6, 1.7, 1.8, 1.9]}[k]
    print(k, " ".join("%.3g→%+.4f" % (g, fs[j].predict([g])[0]) for g in grid))


# ===== 第一步：本地（评分机 2023–24，同一份代码输出）→ 公榜五项。用 1617 一族有本地复现的提交拟合 =====
# 公榜/本地 的比值：IC 稳定约 1.11；IC_IR、压力、多空的比值随换手降低而变小（重缓冲在公榜区间损失更大），按本地换手做线性拟合
PAIRS = {  # 编号: 本地（IC, IC_IR, 换手, 多空, 压力）
    "1617": (0.0888, 1.658, 0.518, 51.09, 1.533), "1631": (0.0847, 1.609, 0.396, 48.34, 1.504),
    "1642": (0.0894, 1.731, 0.496, 51.59, 1.603), "1643": (0.0857, 1.690, 0.390, 48.46, 1.582),
    "1644": (0.0893, 1.687, 0.474, 51.23, 1.563), "1645": (0.0837, 1.666, 0.362, 47.40, 1.574),
    "1647": (0.0766, 1.555, 0.257, 40.18, 1.506), "1646": (0.0801, 1.613, 0.305, 43.94, 1.541),
    "1648": (0.0734, 1.494, 0.214, 36.35, 1.459), "1658": (0.0884, 1.787, 0.365, 48.10, 1.652),
    "1659": (0.0881, 1.795, 0.374, 47.97, 1.658), "1660": (0.0893, 1.792, 0.378, 48.32, 1.655),
    "1661": (0.0891, 1.771, 0.356, 47.80, 1.638), "1664": (0.0921, 1.799, 0.357, 50.44, 1.650),
    "1666": (0.0914, 1.791, 0.291, 48.25, 1.646), "1667": (0.0905, 1.784, 0.237, 46.37, 1.640)}   # 1662 是 1626 底座（树拟合到 2024），2023–24 本地在样本内，不入配对
L = np.array([PAIRS[i] for i in PAIRS]); P_ = np.array([[R[i][k] for k in K] for i in PAIRS])
ratio = P_ / L
coef = [np.polyfit(L[:, 2], ratio[:, j], 1) for j in range(len(K))]               # 比值 = a·本地换手 + b
print("\n本地 → 公榜的比值（按本地换手线性）：")
for j, k in enumerate(K):
    print("  %s 比值 %s，拟合斜率 %+.3f 截距 %.3f" % (k, " ".join("%.3f" % r for r in ratio[:, j]), coef[j][0], coef[j][1]))


def to_public(loc):
    loc = np.asarray(loc, float)
    return np.array([loc[j] * np.polyval(coef[j], loc[2]) for j in range(len(K))])


def estimate(loc, model=m):
    return float(predict(model, to_public(loc)[None, :])[0])


print("\n验证（两步都留一：去掉该提交后重拟比值与百分位曲线）：")
for i in PAIRS:
    keep = [x for x in PAIRS if x != i]
    Lk = np.array([PAIRS[x] for x in keep]); Pk = np.array([[R[x][k] for k in K] for x in keep])
    ck = [np.polyfit(Lk[:, 2], (Pk / Lk)[:, j], 1) for j in range(len(K))]
    pub = np.array([PAIRS[i][j] * np.polyval(ck[j], PAIRS[i][2]) for j in range(len(K))])
    idx = [n for n, x in enumerate(ids) if x != i]
    mk = fit(X[idx], y[idx], w[idx])
    print("  %s 公榜 %.4f 两步留一估计 %.4f 差 %+.4f" % (i, R[i]["score"], predict(mk, pub[None, :])[0], predict(mk, pub[None, :])[0] - R[i]["score"]))

CANDS = {  # 待交的候选（本地 2023–24）
    "1659 第二十一批（待出分）": (0.0881, 1.795, 0.374, 47.97, 1.658),
    "1660 第二十二批 14:30 反向": (0.0893, 1.792, 0.378, 48.32, 1.655),
    "1661 第二十二批 14:30 反向 规模比 0.2": (0.0891, 1.771, 0.356, 47.80, 1.638),
    "1656 订单流等 + 抬升 0.3 β1.0": (0.0885, 1.787, 0.407, 49.40, 1.660),
    "1657 订单流等 + 抬升 0.6 β0.85": (0.0879, 1.787, 0.392, 47.73, 1.658),
    "1652 抬升 0.6 β1.0": (0.0862, 1.707, 0.357, 46.52, 1.593),
    "1649 抬升 1.0 β1.5": (0.0820, 1.649, 0.257, 41.01, 1.557),
    "订单流等 + 抬升 0.3 β0.7": (0.0893, 1.799, 0.451, 50.76, 1.664),
    "订单流等 + 抬升 0.6 β0.7": (0.0883, 1.790, 0.418, 48.64, 1.659),
    "订单流等 + 抬升 0.6 β1.0": (0.0873, 1.776, 0.368, 47.27, 1.654)}
print("\n候选的公榜估计：")
for nm, loc in CANDS.items():
    pub = to_public(loc)
    print("  %-28s 估计 %.4f | 估公榜 IC %.4f IR %.3f 换手 %.3f 多空 %.1f 压力 %.3f" % (nm, estimate(loc), *pub))


# ===== 第二步的平滑版：每项一条 logistic 百分位曲线（换手反向），分数 = c0 + c1·五项均值；12 个参数，最小二乘 =====
from scipy.optimize import least_squares
SC = X.std(0)


def logi(theta, Xn):
    c0, c1 = theta[:2]
    mu, ls_ = theta[2:7], theta[7:12]
    z = (Xn - mu) / (np.exp(ls_) * SC)
    z[:, 2] = -z[:, 2]                                                            # 换手越低越好
    return c0 + c1 * (1 / (1 + np.exp(-z))).mean(1)


def fit_logi(Xf, yf, wf):
    th0 = np.r_[0.7, 0.3, np.median(Xf, 0), np.zeros(5)]
    r = least_squares(lambda t: np.sqrt(wf) * (logi(t, Xf) - yf), th0, max_nfev=20000)
    return r.x


TH = fit_logi(X, y, w)
pl = logi(TH, X)
lo2 = np.array([logi(fit_logi(np.delete(X, i, 0), np.delete(y, i), np.delete(w, i)), X[i:i + 1])[0] for i in range(len(y))])
recent = np.array([int(i) >= 1600 for i in ids])
print("\n平滑版：样本内误差 %.4f；留一 %.4f（1600 之后 %.4f），相关 %.3f" % (np.abs(pl - y).mean(), np.abs(lo2 - y).mean(), np.abs(lo2 - y)[recent].mean(), np.corrcoef(lo2, y)[0, 1]))
print("  中心（百分位 50%% 处）：" + " ".join("%s %.4g" % (k, TH[2 + j]) for j, k in enumerate(K)) + "；尺度：" + " ".join("%s %.3g" % (k, np.exp(TH[7 + j]) * SC[j]) for j, k in enumerate(K)))
for i in PAIRS:
    keep = [x for x in PAIRS if x != i]
    Lk = np.array([PAIRS[x] for x in keep]); Pk = np.array([[R[x][k] for k in K] for x in keep])
    ck = [np.polyfit(Lk[:, 2], (Pk / Lk)[:, j], 1) for j in range(len(K))]
    pub = np.array([PAIRS[i][j] * np.polyval(ck[j], PAIRS[i][2]) for j in range(len(K))])
    idx = [n for n, x in enumerate(ids) if x != i]
    tk = fit_logi(X[idx], y[idx], w[idx])
    print("  两步留一 %s 公榜 %.4f 估计 %.4f 差 %+.4f" % (i, R[i]["score"], logi(tk, pub[None, :])[0], logi(tk, pub[None, :])[0] - R[i]["score"]))
print("候选（平滑版）：")
for nm, loc in CANDS.items():
    print("  %-28s 估计 %.4f" % (nm, logi(TH, to_public(loc)[None, :])[0]))
