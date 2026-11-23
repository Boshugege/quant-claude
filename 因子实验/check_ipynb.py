# -*- coding: utf-8 -*-
"""因子 notebook 的静态检查（10-19，1620／1622 平台报错后加）：notebook 代码单元里被调用的顶层函数名必须都有定义。
用法：python3 因子实验/check_ipynb.py <.ipynb> [...]；有缺失则退出码 1。"""
import ast, builtins, json, sys
bad = 0
for p in sys.argv[1:]:
    nb = json.load(open(p, encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    t = ast.parse(src)
    top = {n.name for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    defined = {n.name for n in ast.walk(t) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    imported = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            imported |= {a.asname or a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported |= {a.asname or a.name for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.Lambda)):
            for a in ast.walk(n.args):
                if isinstance(a, ast.arg):
                    imported.add(a.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            imported.add(n.id)
        elif isinstance(n, (ast.For, ast.comprehension)):
            for x in ast.walk(n.target):
                if isinstance(x, ast.Name):
                    imported.add(x.id)
    called = {n.func.id for n in ast.walk(t) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    called |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and (n.id.startswith("main_b") or n.id.startswith("_"))}
    missing = sorted(x for x in called if x not in defined and x not in imported and not hasattr(builtins, x))
    print("%s：顶层函数 %d 个；缺失 %s" % (p.split("/")[-1][:40], len(top), missing or "无"))
    bad += bool(missing)
sys.exit(1 if bad else 0)
