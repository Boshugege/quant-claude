# C11：1609（= 1604 + CLS20 0.05／14:30 +0.10 + CP 再 0.05）上根剂量 ×1.25／×1.5（Δ 相对 1609）
import numpy as np
import root_eval as R
from c1_upgrades import G, CP, CLS, ax, W1363, SMALL, DEAL, MET
from g_go import WINS, IXW
c = G.c; z930 = G.z930; slot = c.slot_of_date[c.date_id]
SAFE = [r for r in W1363 if r not in DEAL]
NAMES = ("fm_s152", "o21_s152", "o20_s152")
K = 0.2908 / 0.289
HEAD = {}
for ix in (0, 1, 2):
    x = np.asarray(R.load(NAMES[ix])[0], float).copy(); x[z930] = 0.0; HEAD[ix] = x
def b(ix, dose):
    B = HEAD[ix].copy()
    for r in SAFE:
        B += dose * W1363[r] * ax(r, ix)
    for k, w in SMALL.items():
        B += w * G.axis(k, ix)
    B += 0.10 * K * CP + 0.05 * K * CLS + 0.10 * K * CLS * (slot == 1430)
    B[z930] = 0.0
    return B
def ev5(Bs):
    return {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn, _, _ in WINS}
ref = ev5({ix: b(ix, 1.0) for ix in (0, 1, 2)})
print("1609 参考 " + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[wn][m] for m in MET]) for wn in ("2023", "2024", "2021", "2022", "2020")), flush=True)
for dose in (0.75, 1.25, 1.5):
    r = ev5({ix: b(ix, dose) for ix in (0, 1, 2)})
    line, star = "剂量 ×%.2f" % dose, 0
    for wn in ("2023", "2024", "2021", "2022", "2020"):
        d = {k: r[wn][k] - ref[wn][k] for k in MET}
        ok = d["ic_mean"] >= 0 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
        star += ok
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
    print(line + "  %d/5" % star, flush=True)
