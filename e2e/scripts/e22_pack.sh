#!/bin/bash
# E22 打包：E18 的四个 Transformer（F40–F43，种子 42–45）+ 四个 GRU 成员（F70–F73，种子 46–49，4090 全量，同配方 --seq-enc gru）+ E17b 树（权重 TREE_W，默认 1）
# → 合并 → 打包 → verify_package（含 8 成员极小规模重训）→ 截断检测。用法：TREE_W=1 bash e2e/scripts/e22_pack.sh [包名]
set -e
cd "$(dirname "$0")/../.."
TREE_W=${TREE_W:-1}
for r in F40_2020-24_E18_leg01_s42 F41_2020-24_E18_leg01_s43 F42_2020-24_E18_leg01_s44 F43_2020-24_E18_leg01_s45; do
    mkdir -p e2e/runs/$r
    rsync -a macstudio:Desktop/work/quant/e2e/runs/$r/{transformer_model.json,config.json,metrics.json,log.txt} e2e/runs/$r/
done
for r in F70_2020-24_E21_gru_leg01_s46 F71_2020-24_E21_gru_leg01_s47 F72_2020-24_E22_gru_leg01_s48 F73_2020-24_E22_gru_leg01_s49; do
    mkdir -p e2e/runs/$r
    for f in transformer_model.json config.json metrics.json log.txt; do scp -q "4090:D:/quant-claude/e2e/runs/$r/$f" e2e/runs/$r/; done
done
RUNS="F40_2020-24_E18_leg01_s42 F41_2020-24_E18_leg01_s43 F42_2020-24_E18_leg01_s44 F43_2020-24_E18_leg01_s45 F70_2020-24_E21_gru_leg01_s46 F71_2020-24_E21_gru_leg01_s47 F72_2020-24_E22_gru_leg01_s48 F73_2020-24_E22_gru_leg01_s49"
M=e2e/runs/F40-43_70-73_2020-24_E22_tf4_gru4_tw$TREE_W
STATS_ATOL=${STATS_ATOL:-0.02} python3 e2e/scripts/merge_members.py $M $(for r in $RUNS; do echo e2e/runs/$r; done)
python3 - "$M" e2e/runs/F30-33G8_2020-24_E17b_gbdt_win12 "$TREE_W" <<'PY'
import json, os, sys
m, g, tw = sys.argv[1], sys.argv[2], float(sys.argv[3])
ck = json.load(open(os.path.join(m, "transformer_model.json")))
gb = json.load(open(os.path.join(g, "transformer_model.json")))["gbdt"]
gb["cfg"] = dict(gb["cfg"], weight=tw)
ck["gbdt"] = gb
json.dump(ck, open(os.path.join(m, "transformer_model.json"), "w"))
c = json.load(open(os.path.join(m, "config.json"))); c["gbdt"] = gb["cfg"]
json.dump(c, open(os.path.join(m, "config.json"), "w"), ensure_ascii=False, indent=1)
print("接上树成员（权重 %g）；成员数" % tw, len(ck["state_dicts"]), "gru_members", ck.get("gru_members"))
PY
N="${1:-E22_E18g加四个GRU成员_2020-24}"
python3 e2e/scripts/package.py $M "$N" 2>&1 | tail -1
grep -n "^N_MODELS\|^E04_LEG_TURN\|^E04_GBDT\|^E04_GRU_MEMBERS" "e2e/提交/$N/Transformer_modelsave_train.py" | cut -c1-160
rsync -a "e2e/提交/$N" "e2e/提交/$N.zip" macstudio:Desktop/work/quant/e2e/提交/
bash e2e/scripts/remote.sh e2e/scripts/verify_package.py "e2e/提交/$N" 2>&1 | grep -E "校验|通过|失败|Error|assert" | tail -4
ssh macstudio "cd ~/Desktop/work/quant; /opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python e2e/scripts/leak_check.py 'e2e/提交/$N' 2024-03-04 '2024-03-06 10:30:00' '2024-03-06 14:30:00' 2>&1 | grep -E '截断|结论'" < /dev/null
