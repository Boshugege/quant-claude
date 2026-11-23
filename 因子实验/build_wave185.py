# -*- coding: utf-8 -*-
"""第一百八十五批（精选优化第三批）：线 A「更强同族底座 × 中性化力度阶梯」。

任务 §2 点名的三个可选同族底座（337 几何混合聚合、326 官方风格中性化、445 真换手率中性化）
在暴露表禁用期被作废，但它们的**前日错位合法版**其实已经在库里：
  342 = 337 几何聚合的前日市值版（合法头部，换手 0.5256）
  444 = 326「官方风格/行业中性化」的前日行业版（IC 均值 z +8.63，本族最高）
  445 = 445 真换手率中性化（IC 均值 z +7.97）
本批把「中性化力度」这条**公榜已证明有效**的轴铺到这几个更强的底座上（它们目前只有 1.5 倍一个点），
用来回答：IC 均值维度领先的底座能不能同时吃到中性化力度杠杆。

- 1080 / 1081 / 1086：445 真换手率中性化底座 × 2.0 / 2.5 / 2.75
- 1082 / 1083 / 1085：444 前日行业中性化底座 × 2.0 / 2.5 / 2.75
- 1084：450 前日市值加权同伴底座 × 2.5
- 1087：342 几何聚合前日市值底座 × 2.25（补 476=2.0 与 500=2.5 之间的点）
"""
import io, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
OLD = 'result["factor"] = neutralize("signal", CHARS4, 1.5)'


def load(num):
    for f in sorted(os.listdir(D)):
        if f.startswith(str(num) + "_") and f.endswith(".py"):
            return io.open(os.path.join(D, f), encoding="utf-8").read()
    raise SystemExit("missing base %d" % num)


def save(num, name, text):
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write(text)
    print("wrote %d_%s.py" % (num, name))


def mult(t, m, note):
    assert OLD in t, "no 1.5x neutralize"
    t = t.replace(OLD, 'result["factor"] = neutralize("signal", CHARS4, %s)' % m, 1)
    lines = t.split("\n")
    assert lines[1].lstrip().startswith('"""'), lines[1]
    lines.insert(2, "    %s" % note)
    return "\n".join(lines)


N_445 = "【精选优化第三批 1080 族】445 真换手率中性化底座，中性化力度 %s 倍（原 445 为 1.5 倍）。"
N_444 = "【精选优化第三批】444 前日行业中性化底座，中性化力度 %s 倍（原 444 为 1.5 倍）。"
N_450 = "【精选优化第三批】450 前日市值加权同伴底座，中性化力度 %s 倍（原 450 为 1.5 倍）。"
N_342 = "【精选优化第三批】342 几何聚合前日市值底座，中性化力度 %s 倍（原 342 为 1.5 倍）。"

save(1080, "真换手率中性化二点零", mult(load(445), "2.0", N_445 % "2.0"))
save(1081, "真换手率中性化二点五", mult(load(445), "2.5", N_445 % "2.5"))
save(1082, "行业中性化前日行业二点零", mult(load(444), "2.0", N_444 % "2.0"))
save(1083, "行业中性化前日行业二点五", mult(load(444), "2.5", N_444 % "2.5"))
save(1084, "前日市值加权同伴二点五", mult(load(450), "2.5", N_450 % "2.5"))
save(1085, "行业中性化前日行业二点七五", mult(load(444), "2.75", N_444 % "2.75"))
save(1086, "真换手率中性化二点七五", mult(load(445), "2.75", N_445 % "2.75"))
save(1087, "几何混合聚合前日市值二点二五", mult(load(342), "2.25", N_342 % "2.25"))
