# -*- coding: utf-8 -*-
"""命令行评分（不经过 HTTP 服务，流程相同：runner 产出因子值 → 评分核心评分 → result.json / report.md）。

  .venv/bin/python score.py 104_同伴残差阶梯过度中性.ipynb --start 2024-01 --end 2024-12
  .venv/bin/python score.py E05.zip --start 2024-01 --end 2024-03
  .venv/bin/python score.py factor.parquet
  .venv/bin/python score.py --fid 104 --start 2023-01 --end 2024-12
提交类型自动识别，也可以用 --kind 指定。结果写在 jobs/<任务号>/，HTTP 页面的“任务”里同样能看到。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402


def detect(names):
    low = [n.lower() for n in names]
    if any(n.endswith(".zip") for n in low) or "transformer_model.json" in low:
        return "e2e"
    if any(n.endswith((".parquet", ".pq", ".csv")) for n in low):
        return "values"
    if any(n.endswith(".sql") for n in low):
        return "sql"
    if any(n.endswith(".ipynb") for n in low):
        return "platform"
    src = Path(names[0]).read_text(encoding="utf-8", errors="ignore")
    return "platform" if "def main(" in src else "factor_py"


def main():
    ap = argparse.ArgumentParser(description="JYC 评分机命令行")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--kind", default="auto")
    ap.add_argument("--fid")
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--label", default="y_off", choices=["y_off", "official"])
    ap.add_argument("--chunk", default=None, choices=["quarter", "month", "none"])  # 默认：platform 按季度，e2e 按月
    ap.add_argument("--name")
    a = ap.parse_args()
    kind = a.kind if a.kind != "auto" else ("fid" if a.fid and not a.files else detect(a.files))
    jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = C.JOBS / jid
    (d / "input").mkdir(parents=True)
    for f in a.files:
        shutil.copy(f, d / "input" / Path(f).name)
    names = [Path(f).name for f in a.files]
    spec = {"kind": kind, "inputs": names, "fid": a.fid, "start": a.start, "end": a.end, "label": a.label,
            "chunk": a.chunk, "name": a.name or (f"factor_lib {a.fid}" if kind == "fid" else names[0])}
    (d / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    state = {"id": jid, "name": spec["name"], "kind": kind, "start": a.start, "end": a.end, "label": a.label,
             "status": "producing", "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    print(f"任务 {jid}（{kind}），目录 {d}", flush=True)
    rc = subprocess.call([sys.executable, str(HERE / "runner.py"), "--job", str(d)])
    prod = json.loads((d / "produce.json").read_text(encoding="utf-8"))
    if rc or not prod.get("ok"):
        state.update(status="failed", error=prod.get("error"))
        (d / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        sys.exit(f"失败：{prod.get('error')}")
    import engine
    import scoring
    core = engine.Core(C.CORE_DIR, workers=C.ENGINE_THREADS)
    refpool = None
    if (C.REFS_DIR / "refs.json").exists():
        import refs
        refpool = refs.RefPool()
    res = scoring.score(core, refpool, d, spec, prod)
    state.update(status="done", metrics=res["metrics"], warnings=len(res.get("warnings", [])),
                 estimated_public_score=(res.get("reference_pool") or {}).get("estimated_public_score"),
                 finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (d / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    print((d / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
