# -*- coding: utf-8 -*-
"""第十八轮 N1（更正版）：把去成交笔数合成的三区间结果汇总成一张表（对 1753 对照）。

1753 = 只留六根（L_lab10/M_q_lin/W_vdir/W_zero/Z_mfr60/Z_mfr90）= 公榜现在实际跑的 1363。
1752／1758 = 1753 再加两根去 deal_number 的近 5 日根；1759 = 1602 的修复版。
"""
import json
import urllib.request

JOBS = {
    "1753": {"main": "1017-222400-bf4f", "o21": "1017-222400-c246", "o20": "1017-222400-d5b7"},
    "1752": {"main": "1017-222400-d6f1", "o21": "1017-222400-8d55", "o20": "1017-222400-441c"},
    "1758": {"main": "1017-223235-d1a0", "o21": "1017-223235-6437", "o20": "1017-223235-ecf6"},
    "1759": {"main": "1017-223416-46db", "o21": "1017-223416-31ec", "o20": "1017-223416-e27e"},
    "1760": {"main": "1017-230126-e3b0", "o21": "1017-230126-c948"},
    "1761": {"main": "1017-233301-10e9", "o21": "1017-233301-b4e0", "o20": "1017-233301-20d3"},
    "1759v2": {"main": "1018-000209-765b", "o21": "1018-000209-a46a", "o20": "1018-000209-a29d"},
    "1762": {"main": "1017-233354-e00d", "o21": "1017-233354-c50f", "o20": "1017-233354-576f"},
    "1360code": {"main": "1017-233354-fa8b"},
}
M = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")


def get(j):
    return json.load(urllib.request.urlopen("http://127.0.0.1:46220/api/jobs/%s/files/result.json" % j, timeout=30))


def main():
    res = {}
    for nm, wins in JOBS.items():
        for w, j in wins.items():
            try:
                res[(nm, w)] = get(j)
            except Exception:
                pass
    for w in ("main", "o21", "o20"):
        print("== %s ==" % w)
        ref = res.get(("1753", w))
        for nm in ("1753", "1752", "1758", "1761", "1760", "1759v2", "1762", "1360code"):
            d = res.get((nm, w))
            if not d:
                print("  %-5s 未出分" % nm)
                continue
            m = d["metrics"]
            base = "  %-5s IC %+.5f IR %+.4f 多空 %+6.2f 压力 %+.4f 换手 %.4f" % (
                nm, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"])
            if nm == "1762" and res.get(("1360code", w)):
                r = res[("1360code", w)]["metrics"]
                base += " | Δ对1360code IC %+.5f IR %+.4f 多空 %+.2f 压力 %+.4f 换手 %+.4f" % (
                    m["ic_mean"] - r["ic_mean"], m["ic_ir"] - r["ic_ir"], m["ls_sharpe"] - r["ls_sharpe"],
                    m["stress"] - r["stress"], m["turnover"] - r["turnover"])
            elif ref and nm not in ("1753", "1360code"):
                r = ref["metrics"]
                base += " | Δ IC %+.5f IR %+.4f 多空 %+.2f 压力 %+.4f 换手 %+.4f" % (
                    m["ic_mean"] - r["ic_mean"], m["ic_ir"] - r["ic_ir"], m["ls_sharpe"] - r["ls_sharpe"],
                    m["stress"] - r["stress"], m["turnover"] - r["turnover"])
            print(base)


if __name__ == "__main__":
    main()
