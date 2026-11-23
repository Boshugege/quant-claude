#!/bin/bash
# E17 打包：E16 的四个 Transformer 成员 + 2020–24 树模型成员（e17_add_gbdt.py 的输出）→ 打包 → 同步 → verify_package。本机项目根目录运行。
set -e
cd "$(dirname "$0")/../.."
M=e2e/runs/F30-33G_2020-24_E17_mixed_x7_4seeds_gbdt
mkdir -p $M
rsync -a macstudio:Desktop/work/quant/$M/ $M/
N="${1:-E17_E16四成员加树模型成员_2020-24}"
python3 e2e/scripts/package.py $M "$N" 2>&1 | tail -1
grep -n "^N_MODELS\|^EPOCHS, BATCH, LR, SEED\|^WIN_GROUPS\|^E04_EXTRA\|^E04_GBDT\|^E04_NORM\|^FIELD_SET" "e2e/提交/$N/Transformer_modelsave_train.py" | cut -c1-200
ls -la "e2e/提交/$N/"
rsync -a "e2e/提交/$N" "e2e/提交/$N.zip" macstudio:Desktop/work/quant/e2e/提交/
bash e2e/scripts/remote.sh e2e/scripts/verify_package.py "e2e/提交/$N" 2>&1 | grep -E "校验|通过|失败|Error|assert|截面|members|models|推理|树" | tail -8
