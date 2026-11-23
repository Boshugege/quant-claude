# C10：1605／1606（剂量 ×1.5 + CLS20 + CP0.05）再加 CP 是否五项全好（Δ 相对 cp=0.05）
import sys
from c1_upgrades import build, evalB, MET
for kind in ("1603", "1601"):
    ref = evalB({ix: build(kind, ix, dose=1.5, cp=0.05) for ix in (0, 1, 2)})
    for cp in (0.075, 0.10):
        r = evalB({ix: build(kind, ix, dose=1.5, cp=cp) for ix in (0, 1, 2)})
        line, star = "%s cp%.3f" % (kind, cp), 0
        for wn in ("2023", "2024", "2021", "2022", "2020"):
            d = {k: r[wn][k] - ref[wn][k] for k in MET}
            ok = d["ic_mean"] >= 0 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
            star += ok
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
        print(line + "  %d/5" % star, flush=True)
