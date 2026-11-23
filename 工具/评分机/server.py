# -*- coding: utf-8 -*-
"""评分机 HTTP 服务（只监听 127.0.0.1，经 ssh -L 隧道访问）。

- 启动时把评分核心（内存映射）和参照池载入内存，之后每个任务只需算因子值本身。
- 提交是异步任务：POST /api/jobs 立即返回任务号；因子值在子进程里计算（内存上限、超时与平台一致，可取消），
  评分在服务进程内用常驻的评分核心完成。加 wait=true 可同步等待结果（脚本调用方便）。
- 接口文档：页面“API”标签，或 /docs（自动生成的 OpenAPI）。
启动：.venv/bin/uvicorn server:app --host 127.0.0.1 --port 46220
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import psutil
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import engine  # noqa: E402
import scoring  # noqa: E402

app = FastAPI(title="JYC 本地评分机", version=scoring.VERSION,
              description="按平台 jyc_eval 口径（官方标签 + 10 风格与 32 个行业剔除）评分。说明见首页“说明”和“API”标签。")

KINDS = {
    "values": "因子值文件（parquet / csv，三列 date, instrument, factor）",
    "platform": "传统因子提交（.py 或 .ipynb，定义 main(datasources, start_date, end_date)），在模拟平台上运行",
    "e2e": "端到端提交（三个固定文件名的文件，或它们的 zip），运行推理 notebook 的 main",
    "factor_py": "本地研究脚本（.py，定义 factor(panel)）",
    "fid": "factor_lib 已登记的因子编号（不需要上传文件）",
    "sql": "一条 DuckDB SQL（.sql，可引用平台表名），返回 date, instrument, factor",
}
STATE = {"core": None, "refs": None, "jobs": {}, "procs": {}, "started": time.time()}
LOCK = threading.Lock()
DRAIN = {"on": False, "since": None}   # 排空：不再放行新任务（升级前用），排队的任务保留，重启后按原顺序继续
SCORE_LOCK = threading.BoundedSemaphore(C.SCORE_SLOTS)   # 服务进程内同时评分的任务数（秩缓存有锁，其余为只读共享）
QUEUE = deque()                    # 排队中的任务号（先进先出）
RUNNING = {}                       # 正在产出因子值的任务：{任务号: {"slots", "reserve"}}
LAST_LAUNCH = {"t": 0.0, "known": True}   # 上一次放行的时间，以及那个任务的内存是否有本家族历史可依
COND = threading.Condition()


# ---------------------------------------------------------------- 启动
@app.on_event("startup")
def startup():
    # 同步接口跑在 anyio 线程池里（默认 40 个线程）。wait=true 的请求会一直占着线程，
    # 太多脚本同时等待时会把 /api/health 也堵住，导致健康检查失败，所以放大线程池。
    import anyio.to_thread
    anyio.to_thread.current_default_thread_limiter().total_tokens = 256
    C.JOBS.mkdir(parents=True, exist_ok=True)
    if (C.CORE_DIR / "meta.json").exists():
        STATE["core"] = engine.Core(C.CORE_DIR, workers=C.ENGINE_THREADS)
    if (C.REFS_DIR / "refs.json").exists():
        import refs
        STATE["refs"] = refs.RefPool()
    requeue = []
    for d in sorted(C.JOBS.iterdir()):
        st = d / "state.json"
        if st.exists():
            s = json.loads(st.read_text(encoding="utf-8"))
            changed = _slim(s)
            if "chunk" not in s and (d / "spec.json").exists():      # 2.4.1 前的任务：补上分块方式和 SQL 家族，供内存预测用
                sp = json.loads((d / "spec.json").read_text(encoding="utf-8"))
                s["chunk"] = sp.get("chunk") or ("month" if s.get("kind") == "e2e" else "quarter")
                s["family"] = _family(d, sp.get("inputs") or []) if s.get("kind") in ("platform", "e2e", "sql") else None
                changed = True
            if changed:
                st.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
            STATE["jobs"][s["id"]] = s
            if s["status"] in ("queued", "producing", "scoring") and (d / "spec.json").exists():
                if s["status"] != "queued":
                    with open(d / "log.txt", "a", encoding="utf-8") as f:
                        f.write(f"\n{time.strftime('%H:%M:%S')} 服务重启，任务重新排队，从头运行\n")
                s.update(status="queued", started=None, peak_rss_gb=None)
                if s.get("oom_retry") and s.get("exclusive") and not s.get("oom_tries"):
                    # 2.5.2 及以前：第一次 OOM 就独占重跑，会让整个队列停摆。改成按默认预留与其它任务并发（2.5.3）
                    s.update(exclusive=None, oom_tries=1, reserve_gb=float(C.KIND_MEM_GB.get(s["kind"], 6)))
                st.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
                requeue.append(s)
    for s in sorted(requeue, key=lambda x: (x.get("priority") != "high", x["created"])):   # 高优先级在前，其余按提交顺序
        QUEUE.append(s["id"])
    threading.Thread(target=dispatcher, daemon=True, name="dispatcher").start()


def ensure_loaded():
    """评分核心 / 参照池在服务启动后才建好时，第一次用到时载入。"""
    with LOCK:
        if STATE["core"] is None and (C.CORE_DIR / "meta.json").exists():
            STATE["core"] = engine.Core(C.CORE_DIR, workers=C.ENGINE_THREADS)
        if STATE["refs"] is None and (C.REFS_DIR / "refs.json").exists():
            import refs
            STATE["refs"] = refs.RefPool()


ERROR_MAX = 2000


def _slim(s):
    """报错信息截断到 2000 字（完整内容在任务日志里）。曾有 KeyError 把整列因子值打进报错，单个任务状态近 10 MB，任务列表一次 70 MB。"""
    e = s.get("error")
    if isinstance(e, str) and len(e) > ERROR_MAX:
        s["error"] = e[:1500] + f" …（共 {len(e):,} 字，已截断，完整内容见任务日志）… " + e[-300:]
        return True
    return False


def save_state(s):
    _slim(s)
    with LOCK:
        STATE["jobs"][s["id"]] = s
        (C.JOBS / s["id"] / "state.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- 类型识别
def detect_kind(names: List[str], contents: dict) -> str:
    low = [n.lower() for n in names]
    if any(n.endswith(".zip") for n in low) or "transformer_model.json" in low or "transformer_modelsave_predict.ipynb" in low:
        return "e2e"
    if any(n.endswith((".parquet", ".pq", ".csv")) for n in low):
        return "values"
    if any(n.endswith(".sql") for n in low):
        return "sql"
    for n in names:
        if n.lower().endswith(".ipynb"):
            return "platform"
        if n.lower().endswith(".py"):
            src = contents.get(n, "")
            if re.search(r"^def\s+main\s*\(", src, re.M):
                return "platform"
            if re.search(r"^def\s+factor\s*\(", src, re.M):
                return "factor_py"
    raise HTTPException(400, "无法识别提交类型，请在 kind 中指定")


# ---------------------------------------------------------------- 执行
def watch(proc, s, logf):
    """子进程看门狗：超时或内存（含子进程）超过上限时终止。"""
    t0 = time.time()
    peak = 0.0
    while proc.poll() is None:
        try:
            p = psutil.Process(proc.pid)
            rss = sum(x.memory_info().rss for x in [p] + p.children(recursive=True)) / 2**30
            peak = max(peak, rss)
            s["rss_gb"] = round(rss, 2)
            if rss > C.JOB_MEMORY_GB:
                proc.kill()
                return f"内存超过 {C.JOB_MEMORY_GB} GB（平台上限 30 GB），任务被终止"
        except psutil.NoSuchProcess:
            break
        if time.time() - t0 > C.JOB_TIMEOUT_S:
            proc.kill()
            return f"运行超过 {C.JOB_TIMEOUT_S // 3600} 小时（平台上限 3 小时），任务被终止"
        if s.get("cancel"):
            proc.kill()
            return "已取消"
        s["peak_rss_gb"] = round(peak, 2)
        time.sleep(1)
    return None


def _job_slots(s):
    return C.KIND_SLOTS.get(s["kind"], 1)


def _months(s):
    try:
        return (int(s["end"][:4]) - int(s["start"][:4])) * 12 + int(s["end"][5:7]) - int(s["start"][5:7]) + 1
    except Exception:
        return None


def _job_reserve(s):
    """预测任务的内存峰值（GB）。优先用同一 SQL 家族、同区间长度、同分块方式的已结束任务的峰值（取 75% 分位；
    峰值是短暂的，各任务的峰值一般不重叠，超出的风险由容器实际余量和内存压力两项实时检查兜底），
    其次用同类型、同区间长度、同分块方式的；样本不足 3 个时用 KIND_MEM_GB 的默认值。"""
    if s.get("exclusive"):
        return C.MEM_BUDGET_GB
    if s.get("reserve_gb"):
        return float(s["reserve_gb"])
    default = C.KIND_MEM_GB.get(s["kind"], 6)
    key = (s["kind"], _months(s), s.get("chunk"))
    hist = [(x.get("family"), x["peak_rss_gb"]) for x in list(STATE["jobs"].values())
            if x is not s and x.get("peak_rss_gb") and x.get("status") in ("done", "failed")
            and (x["kind"], _months(x), x.get("chunk")) == key]
    for pool in ([v for f, v in hist if s.get("family") and f == s.get("family")], [v for f, v in hist]):
        pool = pool[-30:]
        if len(pool) >= 3:
            return float(min(max(np.quantile(pool, 0.75), 1.0), C.JOB_MEMORY_GB))
    return default


def _memory_known(s):
    """这个任务的内存峰值是否有依据：OOM 重跑时记下的预留，或同 SQL 家族已有至少 3 个结束的任务。"""
    if s.get("reserve_gb") or s.get("exclusive"):
        return True
    fam = s.get("family")
    if not fam:
        return False
    n = sum(1 for x in list(STATE["jobs"].values()) if x is not s and x.get("family") == fam
            and x.get("peak_rss_gb") and x.get("status") in ("done", "failed"))
    return n >= 3


def _family(d, names):
    """SQL 家族：输入文件里所有含 SELECT 的三引号字符串的摘要。同一批文本补丁变体通常 SQL 相同，内存峰值也接近。"""
    try:
        txt = "".join((d / "input" / n).read_text(encoding="utf-8", errors="ignore") for n in names)
        sqls = [x for x in re.findall(r'"""(.*?)"""', txt, re.S) if "select" in x.lower()]
        return hashlib.md5("".join(sqls).encode("utf-8")).hexdigest()[:12] if sqls else None
    except Exception:
        return None


def _cgroup_avail_gb():
    """容器（cgroup v2）剩余可用内存：上限 − (当前用量 − 可回收的非活跃页缓存)。不在容器里时返回 None。
    psutil 在容器里看到的是宿主机内存（43 GB），看不到容器 36 GB 的上限，所以两者取小。"""
    try:
        cg = Path("/sys/fs/cgroup")
        mx = (cg / "memory.max").read_text().strip()
        if mx == "max":
            return None
        cur = int((cg / "memory.current").read_text())
        inactive = next((int(l.split()[1]) for l in (cg / "memory.stat").read_text().splitlines() if l.startswith("inactive_file ")), 0)
        return (int(mx) - (cur - inactive)) / 2**30
    except Exception:
        return None


def _avail_gb():
    host = psutil.virtual_memory().available / 2**30
    cg = _cgroup_avail_gb()
    return host if cg is None else min(host, cg)


def _trim():
    """评分、重新评分、赋分表计算后把 glibc 堆里空闲的内存还给系统（否则主进程 RSS 停在峰值，挤占任务的内存）。"""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _cgroup_pressure():
    """容器内存压力（PSI some avg10，%）：进程因等内存而停顿的时间占比。"""
    try:
        line = (Path("/sys/fs/cgroup") / "memory.pressure").read_text().splitlines()[0]
        return float(line.split("avg10=")[1].split()[0])
    except Exception:
        return 0.0


def _mem_budget():
    """任务内存预算：容器上限 − 服务进程现有常驻内存 − 评分预留（每个评分槽位 JYC_SCORE_HEADROOM_GB），且不超过 JYC_MEM_BUDGET_GB。"""
    try:
        mx = (Path("/sys/fs/cgroup") / "memory.max").read_text().strip()
        limit = int(mx) / 2**30 if mx != "max" else None
    except Exception:
        limit = None
    if limit is None:
        return C.MEM_BUDGET_GB
    srv = psutil.Process().memory_info().rss / 2**30
    return min(C.MEM_BUDGET_GB, limit - srv - C.SCORE_HEADROOM_GB * C.SCORE_SLOTS)


def _fits(s):
    """队首任务现在能否放行（动态并发）：
    - 运行中任务按 max(预测峰值, 当前常驻) 计，加上新任务的预测峰值，不超过动态预算；
    - 容器实际剩余内存够新任务的预测峰值；
    - 容器内存压力低（PSI avg10 < JYC_MAX_MEM_PRESSURE，默认 5%），正在换页时不再放行；
    - 槽位上限 JYC_MAX_CODE_JOBS 只作兜底。没有任务在跑时总是放行。"""
    if not RUNNING:
        return True
    if s.get("exclusive"):
        return False
    slots = sum(r["slots"] for r in RUNNING.values())
    if slots + _job_slots(s) > C.MAX_CODE_JOBS:
        return False
    if not LAST_LAUNCH["known"] and time.time() - LAST_LAUNCH["t"] < C.LAUNCH_STAGGER_S:
        return False
    committed = sum(max(r["reserve"], STATE["jobs"][j].get("rss_gb") or 0) for j, r in RUNNING.items())
    need = _job_reserve(s)
    # 运行中任务尚未涨到峰值的部分也要从实际剩余里扣掉
    growth = sum(max(0.0, r["reserve"] - (STATE["jobs"][j].get("rss_gb") or 0)) for j, r in RUNNING.items())
    return (committed + need <= _mem_budget() and _avail_gb() - growth >= need + 1
            and _cgroup_pressure() < C.MAX_MEM_PRESSURE)


def dispatcher():
    """调度线程：values 任务随到随跑；其余按先进先出，在槽位和内存允许时放行。"""
    while True:
        with COND:
            try:
                for jid in [j for j in QUEUE if j not in STATE["jobs"]]:      # 防御：状态丢失的任务号直接丢弃
                    QUEUE.remove(jid)
                for jid in [j for j in QUEUE if STATE["jobs"][j]["kind"] == "values" or STATE["jobs"][j].get("cancel")]:
                    QUEUE.remove(jid)
                    _launch(jid)
                while QUEUE and not DRAIN["on"] and _fits(STATE["jobs"][QUEUE[0]]):
                    _launch(QUEUE.popleft())
            except Exception:                        # 调度线程一旦退出，队列会永久停住，所以任何异常都只记录、继续
                import traceback
                print("[dispatcher] " + traceback.format_exc(), file=sys.stderr, flush=True)
            COND.wait(timeout=2)                     # 运行中任务的内存会变化，定期重新判断


def _launch(jid):
    s = STATE["jobs"][jid]
    RUNNING[jid] = {"slots": _job_slots(s), "reserve": _job_reserve(s)}
    LAST_LAUNCH.update(t=time.time(), known=_memory_known(s))
    threading.Thread(target=run_job, args=(jid,), daemon=True, name=f"job-{jid}").start()


def enqueue(jid, front=False):
    """front：排到最前（OOM 重跑）。priority=high 的任务排在所有普通任务之前、已有高优先级任务之后。"""
    with COND:
        if front:
            QUEUE.appendleft(jid)
        elif STATE["jobs"][jid].get("priority") == "high":
            i = next((k for k, j in enumerate(QUEUE) if STATE["jobs"].get(j, {}).get("priority") != "high"), len(QUEUE))
            QUEUE.insert(i, jid)
        else:
            QUEUE.append(jid)
        COND.notify_all()


def _child_setup():
    os.nice(5)
    try:                                             # 内存吃紧时让内核优先杀任务子进程，而不是服务进程
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def run_job(jid):
    s = STATE["jobs"][jid]
    d = C.JOBS / jid
    err, proc = None, None
    try:
        if s.get("cancel"):
            s.update(status="failed", error="已取消", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            save_state(s)
            return
        s.update(status="producing", started=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_state(s)
        (d / "produce.json").unlink(missing_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   JYC_TMP=str(d / "tmp"))
        (d / "tmp").mkdir(exist_ok=True)
        with open(d / "log.txt", "ab") as logf:
            proc = subprocess.Popen([sys.executable, str(HERE / "runner.py"), "--job", str(d)], stdout=logf,
                                    stderr=subprocess.STDOUT, env=env, cwd=str(HERE), preexec_fn=_child_setup)
            STATE["procs"][jid] = proc
            err = watch(proc, s, logf)
            proc.wait()
        STATE["procs"].pop(jid, None)
        shutil.rmtree(d / "tmp", ignore_errors=True)
    finally:
        with COND:
            RUNNING.pop(jid, None)
            COND.notify_all()
    s.pop("rss_gb", None)
    tries = s.get("oom_tries") or (1 if s.get("oom_retry") else 0)
    if err is None and proc is not None and proc.returncode == -9 and tries < 2:
        # 不是看门狗杀的却收到 SIGKILL：多任务并发时容器内存不足被内核 OOM 终止，结果不受影响，重跑即可。
        # 第一次按被杀时的峰值 ×1.3 预留、排到队首、照常与其它任务并发（2.5.2 是直接独占，整个队列会停摆）；第二次才独占。
        if tries == 0:
            s.update(status="queued", oom_retry=True, oom_tries=1, started=None,
                     reserve_gb=round(max((s.get("peak_rss_gb") or 0.0) * 1.3, C.KIND_MEM_GB.get(s["kind"], 6)), 1))
            msg = f"子进程被系统 OOM 终止（并发任务内存不足），按 {s['reserve_gb']} GB 预留排到队首重跑"
        else:
            s.update(status="queued", oom_tries=2, exclusive=True, started=None)
            msg = "子进程再次被系统 OOM 终止，排到队首单独重跑"
        with open(d / "log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{time.strftime('%H:%M:%S')} {msg}\n")
        save_state(s)
        enqueue(jid, front=True)
        return
    if err is None and proc is not None and proc.returncode == -9:
        err = "子进程被系统 OOM 终止：单独运行时仍超出容器内存上限，请减少内存占用或改用 chunk=quarter / month 分块调用"
    prod = json.loads((d / "produce.json").read_text(encoding="utf-8")) if (d / "produce.json").exists() else {}
    if err or not prod.get("ok"):
        s.update(status="failed", error=err or prod.get("error") or f"子进程退出码 {proc.returncode}",
                 finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_state(s)
        return
    s.update(status="scoring")
    save_state(s)
    try:
        ensure_loaded()
        if STATE["core"] is None:
            raise RuntimeError("评分核心还在构建（prepare_data.py core），稍后再提交")
        with SCORE_LOCK:
            res = scoring.score(STATE["core"], STATE["refs"], d, json.loads((d / "spec.json").read_text(encoding="utf-8")), prod)
            _trim()
        _apply_result(s, res)
    except Exception as e:
        import traceback
        with open(d / "log.txt", "a", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        s.update(status="failed", error=f"评分失败：{type(e).__name__}: {e}")
    s["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(s)


def _apply_result(s, res):
    """把评分结果的摘要写进任务状态（任务列表、赋分表用）。"""
    pr = res.get("paired") or {}
    s.update(status="done", metrics=res["metrics"], warnings=len(res.get("warnings", [])),
             platform_fail=bool(res.get("platform_error")), data_violation=bool(res.get("data_violation")),
             estimated_public_score=(res.get("reference_pool") or {}).get("estimated_public_score"),
             paired={"baseline": pr.get("baseline_name"), "composite_diff": (pr.get("composite") or {}).get("diff"),
                     "composite_z": (pr.get("composite") or {}).get("z"),
                     "z": {k: v.get("z") for k, v in (pr.get("diff") or {}).items()}, "verdict": pr.get("verdict")} if pr else None)


# ---------------------------------------------------------------- 接口
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health", summary="服务状态")
def health():
    ensure_loaded()
    vm = psutil.virtual_memory()
    core = STATE["core"]
    return {"status": "ok", "version": scoring.VERSION, "uptime_s": int(time.time() - STATE["started"]),
            "core_loaded": core is not None, "core": core.meta if core else None,
            "ref_pool_size": len(STATE["refs"].series) if STATE["refs"] else 0,
            "running": [j for j, s in STATE["jobs"].items() if s["status"] in ("producing", "scoring")],
            "draining": DRAIN["on"], "draining_since": DRAIN["since"],
            "queued": list(QUEUE),
            "scheduler": {"slots_used": sum(r["slots"] for r in RUNNING.values()), "slots_total": C.MAX_CODE_JOBS,
                          "mem_committed_gb": round(sum(max(r["reserve"], STATE["jobs"][j].get("rss_gb") or 0)
                                                        for j, r in list(RUNNING.items())), 1),
                          "mem_budget_gb": round(_mem_budget(), 1), "mem_pressure_avg10": _cgroup_pressure(),
                          "reserve_gb": {j: round(r["reserve"], 1) for j, r in list(RUNNING.items())},
                          "next_reserve_gb": round(_job_reserve(STATE["jobs"][QUEUE[0]]), 1) if QUEUE else None,
                          "score_slots": C.SCORE_SLOTS},
            "memory_gb": {"total": round(vm.total / 2**30, 1), "available": round(vm.available / 2**30, 1),
                          "container_available": None if _cgroup_avail_gb() is None else round(_cgroup_avail_gb(), 1),
                          "server_rss": round(psutil.Process().memory_info().rss / 2**30, 2)},
            "cpu_percent": psutil.cpu_percent(interval=0.2), "cpu_count": C.CPU,
            "limits": {"max_code_jobs": C.MAX_CODE_JOBS, "mem_budget_gb": C.MEM_BUDGET_GB, "job_memory_gb": C.JOB_MEMORY_GB, "job_timeout_s": C.JOB_TIMEOUT_S}}


@app.get("/api/meta", summary="提交类型、数据区间、参照池")
def meta():
    ensure_loaded()
    refs = STATE["refs"]
    return {"kinds": KINDS, "data_range": [C.FULL_START, C.FULL_END], "default_range": ["2024-01", "2024-12"],
            "labels": {"y_off": "本地由 1 分钟重建的官方标签（全区间可用，与官方截面秩相关约 0.97）",
                       "official": "平台导出的官方标签片段（只有 2021-06 与 2024-01-02~03-01）"},
            "chunk": {"quarter": "按季度调用 main（platform 默认：每个任务约 4 GB，能同时跑 6 个左右，吞吐最高）", "none": "整个区间一次调用（与平台相同；单个任务快约一倍，但内存约 8 GB，同时能跑的少一半，适合最终确认）", "month": "按月调用（e2e 默认）"},
            "ref_pool": [{"fid": f, "name": refs.info[f]["name"], "public_score": refs.info[f]["public_score"],
                          "snapshot": refs.info[f].get("snapshot"), "calib": refs.info[f].get("calib", False),
                          "ic_ir": (refs.info[f].get("full_metrics") or {}).get("ic_ir"), "job": refs.info[f].get("job")}
                         for f in refs.fids if f in refs.series] if refs else [],
            "ref_pool_source": refs.source if refs else None}


@app.post("/api/jobs", summary="提交评分任务")
def create_job(files: List[UploadFile] = File(default=[]), kind: str = Form("auto"), fid: Optional[str] = Form(None),
                     start: str = Form("2024-01"), end: str = Form("2024-12"), label: str = Form("y_off"),
                     name: Optional[str] = Form(None), chunk: Optional[str] = Form(None),
                     baseline: Optional[str] = Form(None, description="配对比较的基线：已完成任务号，或 ref:<编号>（参照池平台代码因子，默认 ref:104），none 表示不比较"),
                     priority: str = Form("normal", description="normal 或 high（排到所有普通任务之前）"),
                     wait: bool = Query(False, description="true 时等待任务完成并直接返回结果（最长 3 小时）")):
    for v, ok in ((label, ("y_off", "official")), (chunk or "none", ("quarter", "month", "none")), (priority, ("normal", "high"))):
        if v not in ok:
            raise HTTPException(400, f"参数取值 {v} 不在 {ok} 中")
    try:
        s0, e0 = pd_period(start), pd_period(end)
    except Exception:
        raise HTTPException(400, "start / end 格式为 YYYY-MM")
    if not (C.FULL_START <= start <= end <= C.FULL_END):
        raise HTTPException(400, f"区间必须在 {C.FULL_START} ~ {C.FULL_END} 之内，且 start ≤ end")
    jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = C.JOBS / jid
    (d / "input").mkdir(parents=True)
    names, contents = [], {}
    for f in files:
        fn = os.path.basename(f.filename or "input")
        with open(d / "input" / fn, "wb") as out:
            shutil.copyfileobj(f.file, out)
        names.append(fn)
        if fn.endswith(".py"):
            contents[fn] = (d / "input" / fn).read_text(encoding="utf-8", errors="ignore")
    if kind == "auto":
        kind = "fid" if (fid and not names) else detect_kind(names, contents)
    if kind not in KINDS:
        raise HTTPException(400, f"kind 必须是 auto 或 {list(KINDS)}")
    if kind == "fid" and not fid:
        raise HTTPException(400, "kind=fid 需要 fid 参数")
    if kind not in ("fid",) and not names:
        raise HTTPException(400, "需要上传文件")
    if baseline and baseline not in ("none",) and not baseline.startswith("ref:"):
        b = STATE["jobs"].get(baseline)
        if b is None or b.get("status") != "done" or not (C.JOBS / baseline / "factor_keys.parquet").exists():
            raise HTTPException(400, f"基线任务 {baseline} 不存在或未完成")
    spec = {"kind": kind, "inputs": names, "fid": fid, "start": start, "end": end, "label": label,
            "chunk": chunk or ("month" if kind == "e2e" else "quarter"),   # 默认按季度：内存约为整段的一半，同时能跑的任务多一倍，吞吐更高
            "name": name or (f"factor_lib {fid}" if kind == "fid" else names[0]), "baseline": baseline or "ref:104",
            "priority": priority}
    (d / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    s = {"id": jid, "name": spec["name"], "kind": kind, "start": start, "end": end, "label": label,
         "status": "queued", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "priority": priority,
         "baseline": spec["baseline"], "chunk": spec["chunk"], "family": _family(d, names) if kind in ("platform", "e2e", "sql") else None}
    save_state(s)
    enqueue(jid)
    if wait:
        while s["status"] not in ("done", "failed"):
            time.sleep(1)
        return job(jid)
    return {"job_id": jid, "kind": kind, "status": "queued", "detail_url": f"/api/jobs/{jid}"}


def pd_period(x):
    import pandas as pd
    return pd.Period(x, "M")


JOB_FIELDS = ("id", "name", "kind", "start", "end", "label", "status", "created", "started", "finished", "error", "metrics", "warnings",
              "estimated_public_score", "peak_rss_gb", "priority", "baseline", "paired", "platform_fail", "data_violation")


@app.get("/api/jobs", summary="任务列表（分页）")
def jobs(response: Response, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
         status: Optional[str] = None, q: Optional[str] = None, kind: Optional[str] = None):
    """按创建时间倒序。status 可用逗号分隔多个（如 queued,producing,scoring）；q 匹配任务号或名称（不分大小写）。
    返回列表；总数在响应头 X-Total-Count。"""
    for d in C.JOBS.iterdir():                      # 命令行 score.py 建的任务
        st = d / "state.json"
        if d.name not in STATE["jobs"] and st.exists():
            STATE["jobs"][d.name] = json.loads(st.read_text(encoding="utf-8"))
            _slim(STATE["jobs"][d.name])
        elif st.exists() and STATE["jobs"][d.name].get("status") == "producing" and d.name not in STATE["procs"]:
            STATE["jobs"][d.name] = json.loads(st.read_text(encoding="utf-8"))
            _slim(STATE["jobs"][d.name])
    xs = list(STATE["jobs"].values())
    if status:
        want = set(status.split(","))
        xs = [x for x in xs if x.get("status") in want]
    if kind:
        xs = [x for x in xs if x.get("kind") == kind]
    if q:
        ql = q.lower()
        xs = [x for x in xs if ql in x["id"].lower() or ql in (x.get("name") or "").lower()]
    xs.sort(key=lambda s: s["created"], reverse=True)
    response.headers["X-Total-Count"] = str(len(xs))
    return [{k: s.get(k) for k in JOB_FIELDS} for s in xs[offset:offset + limit]]


@app.get("/api/jobs/{jid}", summary="任务详情与完整结果")
def job(jid: str):
    s = STATE["jobs"].get(jid)
    if s is None:
        raise HTTPException(404, "没有这个任务")
    out = dict(s)
    r = C.JOBS / jid / "result.json"
    if s["status"] == "done" and r.exists():
        out["result"] = json.loads(r.read_text(encoding="utf-8"))
    return out


@app.get("/api/jobs/{jid}/log", response_class=PlainTextResponse, summary="运行日志（offset 起的增量）")
def job_log(jid: str, offset: int = 0):
    p = C.JOBS / jid / "log.txt"
    if not p.exists():
        return PlainTextResponse("", headers={"X-Log-Size": "0"})
    with open(p, "rb") as f:
        f.seek(offset)
        b = f.read(2_000_000)
    return PlainTextResponse(b.decode("utf-8", errors="replace"), headers={"X-Log-Size": str(offset + len(b))})


ALLOWED = {"result.json", "report.md", "log.txt", "factor_raw.parquet", "factor_keys.parquet", "dai_queries.json", "spec.json"}


@app.get("/api/jobs/{jid}/files/{name}", summary="下载任务文件")
def job_file(jid: str, name: str):
    p = C.JOBS / jid / name
    if name not in ALLOWED or not p.exists():
        raise HTTPException(404, f"可下载：{sorted(ALLOWED)}")
    return FileResponse(p, filename=f"{jid}_{name}")


def _cancel(jid):
    """取消一个任务：排队中的直接判为已取消；运行中的终止子进程（看门狗随后记为已取消）。返回取消前的状态。"""
    s = STATE["jobs"].get(jid)
    if s is None:
        return None
    st = s["status"]
    if st not in ("queued", "producing", "scoring"):
        return st
    s["cancel"] = True
    with COND:
        if jid in QUEUE:
            QUEUE.remove(jid)
            s.update(status="failed", error="已取消", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            save_state(s)
        COND.notify_all()
    p = STATE["procs"].get(jid)
    if p:
        p.kill()
    return st


def _delete(jid, force=False):
    s = STATE["jobs"].get(jid)
    if s is None:
        return "没有这个任务"
    if s["status"] in ("queued", "producing", "scoring"):
        if not force:
            return "任务还在排队或运行，请先取消（或 force=true）"
        _cancel(jid)
        t0 = time.time()
        while s["status"] in ("producing", "scoring") and time.time() - t0 < 60:   # 评分阶段不能中断，等它结束
            time.sleep(0.5)
        if s["status"] in ("producing", "scoring"):
            return "任务 60 秒内没有停下，稍后再删"
    d = C.JOBS / jid
    if d.is_symlink():                               # 已归档到 HDD 的任务：删链接和 HDD 上的目录（rmtree 不跟随符号链接）
        target = d.resolve()
        d.unlink()
        shutil.rmtree(target, ignore_errors=True)
    else:
        shutil.rmtree(d, ignore_errors=True)
    with LOCK:
        STATE["jobs"].pop(jid, None)
    return None


@app.post("/api/jobs/{jid}/cancel", summary="取消任务（排队中或运行中）")
def cancel(jid: str):
    st = _cancel(jid)
    if st is None:
        raise HTTPException(404, "没有这个任务")
    return {"ok": True, "previous_status": st}


@app.delete("/api/jobs/{jid}", summary="删除任务及其文件（运行中的任务需 force=true，会先取消）")
def delete(jid: str, force: bool = False):
    err = _delete(jid, force)
    if err == "没有这个任务":
        raise HTTPException(404, err)
    if err:
        raise HTTPException(409, err)
    return {"ok": True}


class BatchReq(BaseModel):
    ids: List[str] = []
    name_prefix: Optional[str] = None
    all_queued: bool = False
    force: bool = False


def _select(req: BatchReq):
    ids = set(req.ids)
    for j, s in list(STATE["jobs"].items()):
        if req.name_prefix and (s.get("name") or "").startswith(req.name_prefix):
            ids.add(j)
        if req.all_queued and s["status"] == "queued":
            ids.add(j)
    return sorted(ids)


@app.post("/api/batch/cancel", summary="批量取消：ids、name_prefix（名称前缀）、all_queued（全部排队中）任选组合")
def batch_cancel(req: BatchReq):
    out = {j: _cancel(j) for j in _select(req)}
    return {"cancelled": [j for j, st in out.items() if st in ("queued", "producing", "scoring")],
            "skipped": {j: st for j, st in out.items() if st not in ("queued", "producing", "scoring")}}


@app.post("/api/batch/delete", summary="批量删除：选择方式同批量取消；force=true 时先取消运行中的任务")
def batch_delete(req: BatchReq):
    errs = {}
    ids = _select(req)
    for j in ids:
        e = _delete(j, req.force)
        if e:
            errs[j] = e
    return {"deleted": [j for j in ids if j not in errs], "errors": errs}


BOARD_CACHE = {}


def _board_fingerprint():
    """赋分表缓存的失效条件：已完成任务的集合与完成/重评时间、剔除清单、参照池对象。"""
    done = [(j, s.get("finished"), s.get("rescored"), s.get("data_violation"), s.get("platform_fail"))
            for j, s in list(STATE["jobs"].items()) if s.get("status") == "done"]
    ex = C.STORE / "excluded.json"
    return (len(done), hash(tuple(sorted(done, key=str))), ex.stat().st_mtime if ex.exists() else 0, id(STATE["refs"]))


@app.get("/api/leaderboard", summary="赋分表：按平台赋分规则给全部已登记因子排名")
def get_leaderboard(start: str = "2023-01", end: str = "2024-12", dedupe: bool = True, include_refs: bool = True,
                    q: Optional[str] = None, offset: int = Query(0, ge=0), limit: Optional[int] = Query(None, ge=1, le=1000),
                    sort: str = Query("score", pattern="^(score|pub_est)$")):
    """五项指标在本表全部因子中的百分位等权平均（换手越低越好）。总体 = 参照池 + 区间覆盖窗口的已完成任务（y_off 标签）。
    同名任务默认只保留最新一次（dedupe）。pub_est 是锚定同快照公榜分的公榜估计，sort=pub_est 按它排序。"""
    import leaderboard
    ensure_loaded()
    if not (C.FULL_START <= start <= end <= C.FULL_END):
        raise HTTPException(400, "窗口须在数据区间内且 start ≤ end")
    key = (start, end, dedupe, include_refs, sort)
    fp = _board_fingerprint()
    hit = BOARD_CACHE.get(key)
    if hit is None or hit[0] != fp:
        hit = (fp, leaderboard.table(STATE["refs"], start, end, dedupe, include_refs, sort))
        BOARD_CACHE.clear()                          # 只留最新一份，避免不同窗口的结果堆积
        BOARD_CACHE[key] = hit
        _trim()
    out = dict(hit[1])
    rows = out.get("rows", [])
    if q:
        rows = [r for r in rows if q.lower() in str(r.get("name", "")).lower() or q.lower() in str(r.get("key", "")).lower()]
    out["n_match"] = len(rows)
    out["rows"] = rows[offset:offset + limit] if limit else rows[offset:]
    return out


class RescoreReq(BaseModel):
    ids: List[str] = []
    name_prefix: Optional[str] = None
    baseline: Optional[str] = None


RESCORE = {"running": False, "done": 0, "total": 0, "errors": {}}


@app.post("/api/admin/rescore", summary="用当前评分代码重新评分已完成的任务（不重算因子值；可改基线），后台执行")
def rescore(req: RescoreReq):
    if RESCORE["running"]:
        raise HTTPException(409, f"已有重新评分在进行：{RESCORE['done']}/{RESCORE['total']}")
    ids = [j for j in set(req.ids) | {j for j, st in STATE["jobs"].items() if req.name_prefix and (st.get("name") or "").startswith(req.name_prefix)}
           if STATE["jobs"].get(j, {}).get("status") == "done" and (C.JOBS / j / "factor_keys.parquet").exists()]
    RESCORE.update(running=True, done=0, total=len(ids), errors={})

    def work():
        try:
            ensure_loaded()
            for j in sorted(ids):
                d = C.JOBS / j
                try:
                    spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
                    if req.baseline:
                        spec["baseline"] = req.baseline
                        (d / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
                    prod = json.loads((d / "produce.json").read_text(encoding="utf-8"))
                    with SCORE_LOCK:
                        res = scoring.score(STATE["core"], STATE["refs"], d, spec, prod)
                        _trim()
                    s = STATE["jobs"][j]
                    _apply_result(s, res)
                    if req.baseline:
                        s["baseline"] = req.baseline
                    s["rescored"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    save_state(s)
                except Exception as e:
                    RESCORE["errors"][j] = f"{type(e).__name__}: {e}"
                RESCORE["done"] += 1
        finally:
            RESCORE["running"] = False

    threading.Thread(target=work, daemon=True, name="rescore").start()
    return {"started": len(ids), "status_url": "/api/admin/rescore"}


class DrainReq(BaseModel):
    on: bool = True


@app.post("/api/admin/drain", summary="排空：停止放行新任务（运行中的照常跑完，排队的保留）；升级时先排空再重启，不丢任何进度")
def drain(req: DrainReq):
    with COND:
        DRAIN.update(on=req.on, since=time.strftime("%Y-%m-%d %H:%M:%S") if req.on else None)
        COND.notify_all()
    running = [j for j, s in STATE["jobs"].items() if s["status"] in ("producing", "scoring")]
    return {"draining": DRAIN["on"], "running": running, "queued": len(QUEUE)}


@app.get("/api/admin/rescore", summary="重新评分进度")
def rescore_status():
    return RESCORE


@app.post("/api/admin/reload_refs", summary="重新载入参照池（重建参照池后调用，不用重启服务）")
def reload_refs():
    import importlib
    import refs
    importlib.reload(refs)
    with LOCK:
        STATE["refs"] = refs.RefPool() if (C.REFS_DIR / "refs.json").exists() else None
    return {"ok": True, "ref_pool_size": len(STATE["refs"].series) if STATE["refs"] else 0}


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus 指标（observability 网络内抓取）。"""
    ensure_loaded()
    jobs_ = list(STATE["jobs"].values())
    L = ["# HELP jyc_scorer_up 服务存活", "# TYPE jyc_scorer_up gauge", "jyc_scorer_up 1",
         "# TYPE jyc_scorer_core_loaded gauge", f"jyc_scorer_core_loaded {int(STATE['core'] is not None)}",
         "# TYPE jyc_scorer_ref_pool_size gauge", f"jyc_scorer_ref_pool_size {len(STATE['refs'].series) if STATE['refs'] else 0}",
         "# HELP jyc_scorer_jobs 各状态任务数", "# TYPE jyc_scorer_jobs gauge"]
    for st in ("queued", "producing", "scoring", "done", "failed"):
        L.append(f'jyc_scorer_jobs{{status="{st}"}} {sum(1 for j in jobs_ if j.get("status") == st)}')
    L += ["# HELP jyc_scorer_job_peak_rss_gb 运行中任务的内存峰值", "# TYPE jyc_scorer_job_peak_rss_gb gauge"]
    for j in jobs_:
        if j.get("status") in ("producing", "scoring"):
            L.append(f'jyc_scorer_job_peak_rss_gb{{job="{j["id"]}",kind="{j["kind"]}"}} {j.get("peak_rss_gb") or 0}')
    L += ["# HELP jyc_scorer_last_job_seconds 最近完成任务的耗时", "# TYPE jyc_scorer_last_job_seconds gauge"]
    done = [j for j in jobs_ if j.get("status") == "done" and j.get("started") and j.get("finished")]
    if done:
        j = max(done, key=lambda x: x["finished"])
        t = time.mktime(time.strptime(j["finished"], "%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime(j["started"], "%Y-%m-%d %H:%M:%S"))
        L.append(f'jyc_scorer_last_job_seconds{{kind="{j["kind"]}"}} {t:.0f}')
    return Response("\n".join(L) + "\n", media_type="text/plain; version=0.0.4")
