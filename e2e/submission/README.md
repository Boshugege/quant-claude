# 端到端提交三件套（开发版）

> 历史文档（写于 2026-10-11）。本目录是开发用的工作副本，训练脚本在 E21 交付之后还有改动；最终提交的 E21 以 `e2e/提交/E21_E18g加两个GRU成员_2020-24/` 为准。文中“本地验证（远端 macstudio）”一节说的远端机器已停用。

最后更新：2026-10-11。交付的版本在 `e2e/提交/E<NN>_*/`，由 `scripts/package.py` 从本目录加上某次训练的配置和权重生成。

| 文件 | 作用 |
|---|---|
| `Transformer_modelsave_train.py` | 训练和全部共享定义（数据层、切窗、数值变换、模型、JSON 存取、分片推理）。私榜阶段平台调用 `train_and_save` 从零重训 |
| `Transformer_modelsave_predict.ipynb` | 推理：只加载 JSON 打分，不训练。平台调用 `main(datasources, start_date, end_date)` |
| `transformer_model.json` | 权重和全部推理配置（文本 JSON，上限 50 MB）。本目录原有一份开发用的旧权重，未随仓库公开；交付时用训练取回的 |

三个文件平铺在同一目录上传。notebook 用 `from Transformer_modelsave_train import ...` 导入，所以**训练脚本文件名不能改**。

## 两种结构

- `ARCH="v1"`（E01–E03）：64 根 1 分钟 bar 加 45 根 30 分钟 bar 的 Transformer，可选截面摘要特征、四特征标签中性化和输出中性化。
- `ARCH="e04"`（E04 起）：三个尺度专家（64 根 1 分钟的 10 个原始字段、18 个 30 分钟窗口、20 个交易日）加同伴编码器（60 日收益）和同伴注意力；专家截面排名等权合成，再做四特征输出中性化（`OUT_NEUTRAL`）。30 分钟窗口和日线由 `_WIN30_SQL` 从 1 分钟数据聚合，09:30 窗口取 09:25 竞价 bar 的价格（修复平台 SQL 的 09:30 缺陷，见 reports/12）。
- `USE_STYLE=True`：训练时用官方风格暴露 `cpt_jyc_2026_exposure` 做标签中性化和损失中的风格剔除（只用于训练，不作为模型输入，推理不需要）。目前只剔除 10 个风格，平台评测还剔除 32 个申万一级行业，计划从 E07 起训练也加入行业。

标签用平台表 `cpt_jyc_2026_vwap` 的 `vwap_return`（截面之后 30 分钟窗口的收盘价 ÷ 同窗口 VWAP − 1）。设备优先 cuda，其次 mps，最后 cpu。

## 本地验证（远端 macstudio）

- `scripts/train_local.py`：通过 FeatherBackend 调用同一个 `train_and_save`，每个 epoch 在验证集上按平台口径评估；结束后重新加载 JSON，检查 `predict_range` 与训练路径的预测一致。
- `scripts/verify_package.py` + `scripts/fake_dai/`：用模拟的 `dai`（30 分钟窗口 SQL 用 DuckDB 0.9.2 真实执行）跑通交付包的训练和 notebook 的 `main`，检查三列输出、8 个截面、重复键、inf/NaN 和每个截面的缺失率。

## 交付前检查

- [ ] JSON 由当前训练脚本生成，`train_range`、结构和配置与脚本一致；小于 50 MB。
- [ ] `verify_package.py` 通过。
- [ ] 在平台上用 notebook 的自测单元跑一小段区间，`M.jyc_eval` 能出分。
