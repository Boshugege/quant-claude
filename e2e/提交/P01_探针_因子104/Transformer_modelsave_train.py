# -*- coding: utf-8 -*-
"""诊断探针 P01 的训练脚本（占位）：推理 notebook 原样复用因子 104 号的公式，没有可训练参数。
train_and_save 只写出说明用的 transformer_model.json，便于按端到端格式提交。"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")


def train_and_save(datasources=None, model_path=MODEL_PATH):
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump({"type": "probe_factor104", "trainable_params": 0,
                   "note": "推理直接计算同伴残差阶梯 + 四特征 1.5 倍中性化（因子赛道 104 号），无需权重"}, f, ensure_ascii=False)
    return model_path


if __name__ == "__main__":
    train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"})
