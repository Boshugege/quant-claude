# H2 非线性堆叠：模型 JSON 的复现说明（第十四轮，2026-10-17）

> 历史文档（写于 2026-10-17），只说明最早的两个模型文件，文末“评测机仍在跑”是当时的状态，评分机已停用。最终因子用的树模型是本目录的 `h2_avg16.json`（1617 底座，1669 用）和 `h2_avg16_f24.json`（1626 底座，1670 用），拟合与合并方法见 `文档/复现与消融指南.md` 2.3 节。

两个文件（都是 100 棵树、深度 3、lr 0.06、L2 正则、行抽样 8%、列抽样 80%）：

| 文件 | 输入特征 | 本地五项（2023／2024 为**样本外**） |
|---|---|---|
| `h2_gbdt_100.json` | 24 列（D2 的 15 分量 + 1363 + s152 + 7 个截面时刻 one-hot） | 2023 IC 0.09136／IR 1.9986；2024 0.07412／1.3022 |
| `h2_seas_gbdt_100.json` | 28 列 = 上者 + 4 个季节性族特征（`L_5`／`VSH20`／`PACE20`／`DOW10`） | 2023 0.09492／2.0266；2024 0.07717／1.3539 |

## 怎么算分（纯 numpy）

```python
import json, numpy as np
J = json.load(open("h2_seas_gbdt_100.json"))
E = np.asarray(J["edges"], np.float32)        # (28, 31) 分箱边界
X = ...                                        # (n_rows, 28) float32，列序见 J["feat_order"]
B = np.searchsorted(E[j], X[:, j], side="left").astype(np.uint8)   # 逐列，得到 bin 0..31
score = np.full(len(X), J["model"]["base"])
for tr in J["model"]["trees"]:
    node = np.zeros(len(X), np.int64)
    for item in tr:
        if item[0] == "split":
            _, nid, f, k, _d = item
            sel = node == nid
            node[sel] = np.where(B[sel, int(f)] <= int(k), 2 * nid + 1, 2 * nid + 2)
        else:
            _, nid, v = item
            score[node == nid] += J["model"]["params"]["lr"] * v
```

- 叶子 id 是堆式编号（左 2n+1、右 2n+2），`split` 的最后一维是深度，可直接用。
- 09:30 截面合成前置 0（与评测机 values 口径一致）。
- `feat_order` 里 `slot1000…slot1430` 是截面时刻的 one-hot（`slot_of_date == 1000` 记 1）。
- 最后 4 列（仅在 `h2_seas_gbdt_100.json` 里）是季节性族特征：
  `L_5`＝下一个半小时段 lab 的过去 5 日均值；`VSH20`＝下一个半小时段在过去 20 日的成交额占比均值；
  `PACE20`＝今日累计成交额 ÷ 过去 20 日同一时刻均值；`DOW10`＝过去 10 个同星期几的下一段 lab 均值。
  构造见 `因子实验/h1_season.py`（数据源 `data/bar30m`，等价于 1232 的 1 分钟聚段）。
- 拟合窗固定为 2020-01-01～2022-12-31（`fit_span`）：2020 用 `o20_` 轴、2021–22 用 `o21_` 轴；
  `edges` 也在该窗上分位数算出，**不要在别的窗上重算**，否则与提交的 values 任务对不上。

## 已提交的 values 任务（本地口径，评测机仍在跑）

- `v_H2GBDT`：main `1017-135451-47a4`／o21 `1017-135457-a715`／o20 `1017-135501-3f8d`
- `v_H2SEAS`：main `1017-142042-3fc6`／o21 `1017-142048-93d6`／o20 `1017-142053-c551`
