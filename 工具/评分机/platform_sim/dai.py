# -*- coding: utf-8 -*-
"""本地模拟的 dai（平台 DAI 查询引擎）：提交代码里的 SQL 原样交给 DuckDB 0.9.2（与平台同版本）执行。

支持的表（名字与平台相同）：
  cpt_jyc_2026_stock_bar1m   1 分钟行情（元；OHLC 缺失为 NULL；含 pre_close、adjust_factor、instrument_id、instrument）
                             本地只有 3 档盘口，bid/ask 4–5 档为 NULL
  all_instruments            instrument_id → instrument
  cpt_jyc_2026_instruments   股票池：当日有行情的股票 × 8 个截面
  cpt_jyc_2026_vwap          标签 vwap_return（本地由 1 分钟重建，与官方截面秩相关约 0.97）
  cpt_jyc_2026_exposure      官方风格暴露（平台导出）
filters 与平台一致：{"date": [开始, 结束]} 先限定各表的日期（只给日期的结束日包含当天全天），{"instrument": [...]} 限定股票。
结果经 parquet 文件交换（DuckDB 0.9.2 直接与 numpy 2 交换数据会崩溃）。每次查询的表、区间、行数、耗时写入 stderr。

结果缓存（2026-10-12）：读 1 分钟行情的查询以 (SQL 原文, 各表的文件清单与大小/修改时间, 日期和股票过滤, DuckDB 版本) 为键，
结果 parquet 存在 store/dai_cache/。同一批因子变体大多只改 pandas 部分、SQL 原样不动，命中后直接读文件，结果与重新执行逐位相同。
同一个键用文件锁互斥：第一个任务执行，其余任务等它写完直接读。缓存总量超过 JYC_DAI_CACHE_GB（默认 20）时按最近使用时间淘汰。
SQL 并发闸门：缓存未命中真正执行 SQL 时，同时最多 JYC_DAI_SQL_SLOTS（默认 2）个查询，避免多个任务的 DuckDB 线程抢核。
线程按并发分配：拿到闸门时数一下正在执行的 SQL，线程数 = JYC_DUCKDB_MAX_THREADS（默认 24）÷ 正在执行的个数，单独执行时用满全部核。
预取（2026-10-12）：评分机按季度分块调用 main 时，runner 通过 set_chunk() 告诉本模块当前块和下一块。读 1 分钟行情的查询
拿到结果后，按"过滤区间相对当前块的偏移"推出下一块的过滤区间，在后台线程里用同一条 SQL 预先算好放进缓存；任务做完 pandas
来取下一块时直接命中。同一个键已有人在算就跳过；预取结果与按需执行完全相同（同一条 SQL、同一个缓存键）。
"""
import threading
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import uuid

import pandas as pd

_STORE = os.environ.get("JYC_PLATFORM_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "store", "platform")
_EXPO = os.environ.get("JYC_EXPOSURE_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "exposure")
_THREADS = int(os.environ.get("JYC_DUCKDB_THREADS", "8"))
_MEMORY = os.environ.get("JYC_DUCKDB_MEMORY", "12GB")
_TMP = os.environ.get("JYC_TMP") or tempfile.gettempdir()
_CACHE = os.environ.get("JYC_DAI_CACHE") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "store", "dai_cache")
_CACHE_GB = float(os.environ.get("JYC_DAI_CACHE_GB", "20"))
_SQL_SLOTS = int(os.environ.get("JYC_DAI_SQL_SLOTS", "2"))
_MAX_THREADS = int(os.environ.get("JYC_DUCKDB_MAX_THREADS", str(os.cpu_count() or 8)))
_PREFETCH = os.environ.get("JYC_DAI_PREFETCH", "1") == "1"
_CHUNK = {"cur": None, "next": None}
_PREFETCH_THREADS = []


def set_chunk(cur, nxt):
    """runner 在每次调用 main 前设置：cur / nxt 为 (开始, 结束) 字符串，nxt 为 None 表示没有下一块。"""
    _CHUNK["cur"], _CHUNK["next"] = cur, nxt


def wait_prefetch(timeout=None):
    for t in list(_PREFETCH_THREADS):
        t.join(timeout)
_NONDETERMINISTIC = re.compile(r"\b(random|uuid|gen_random_uuid|now|current_timestamp|current_date|setseed)\b", re.I)
LOG = []
WARNINGS = set()


def _months(sd, ed):
    return [p.strftime("%Y%m") for p in pd.period_range(sd[:7], ed[:7], freq="M")]


def _files(kind, sd, ed):
    if kind == "bar1m":
        fs = [os.path.join(_STORE, "bar1m", f"{m}.parquet") for m in _months(sd, ed)]
        return [f for f in fs if os.path.exists(f)]
    if kind == "exposure":
        ys = range(int(sd[:4]), int(ed[:4]) + 1)
        return [f for f in (os.path.join(_EXPO, f"exposure_{y}.parquet") for y in ys) if os.path.exists(f)]
    return [os.path.join(_STORE, kind + ".parquet")]


_TABLES = {
    "cpt_jyc_2026_stock_bar1m": "bar1m",
    "all_instruments": "all_instruments",
    "cpt_jyc_2026_instruments": "instruments",
    "cpt_jyc_2026_vwap": "vwap",
    "cpt_jyc_2026_exposure": "exposure",
}
_UNSUPPORTED = ["cpt_jyc_2026_stock_bar5m", "cpt_jyc_2026_stock_bar15m", "cpt_jyc_2026_stock_bar30m"]


class _Result(object):
    def __init__(self, df):
        self._df = df

    def df(self):
        return self._df.copy()

    def to_pandas(self):
        return self.df()


def _lit(s):
    return "'" + str(s).replace("'", "''") + "'"


def _file_sig(f):
    st = os.stat(f)
    return [f, st.st_size, int(st.st_mtime)]


def _busy_slots(mine):
    """其他正在执行的 SQL 个数（试锁其余槽位，锁不上的就是忙）。"""
    n = 0
    for j in range(_SQL_SLOTS):
        if j == mine:
            continue
        with open(os.path.join(_CACHE, "slots", f"slot{j}.lock"), "w") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh, fcntl.LOCK_UN)
            except BlockingIOError:
                n += 1
    return n


class _SqlSlot(object):
    """跨进程信号量：在 _SQL_SLOTS 个锁文件里抢一个空闲的。"""
    def __enter__(self):
        self.fh, self.waited, self.busy = None, 0.0, 1
        try:
            os.makedirs(os.path.join(_CACHE, "slots"), exist_ok=True)
        except OSError:                                    # 目录不可写（例如只读挂载）时不做并发限制
            return self
        t0 = time.time()
        while True:
            for i in range(_SQL_SLOTS):
                fh = open(os.path.join(_CACHE, "slots", f"slot{i}.lock"), "w")
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.fh, self.waited = fh, time.time() - t0
                    self.busy = 1 + _busy_slots(i)
                    return self
                except BlockingIOError:
                    fh.close()
            time.sleep(0.5)

    def __exit__(self, *a):
        if self.fh is not None:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()


def _evict(keep):
    """缓存超过上限时按最近使用时间淘汰；正被其他进程持锁的键跳过。"""
    try:
        for f in os.listdir(_CACHE):                       # 被强杀的任务留下的半成品（超过 1 小时）
            if f.endswith(".tmp") and time.time() - os.stat(os.path.join(_CACHE, f)).st_mtime > 3600:
                os.remove(os.path.join(_CACHE, f))
        fs = [os.path.join(_CACHE, f) for f in os.listdir(_CACHE) if f.endswith(".parquet")]
        fs = sorted(fs, key=lambda f: os.stat(f).st_mtime)
        total = sum(os.path.getsize(f) for f in fs)
        for f in fs:
            if total <= _CACHE_GB * 2**30:
                break
            if f == keep:
                continue
            with open(f[:-8] + ".lock", "w") as lk:
                try:
                    fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                sz = os.path.getsize(f)
                os.remove(f)
                total -= sz
    except OSError:
        pass


def _norm_dates(filters):
    d = (filters or {}).get("date")
    sd, ed = (str(d[0]), str(d[1])) if d else ("2020-01-01 00:00:00", "2024-12-31 23:59:59")
    if len(ed) == 10:
        ed += " 23:59:59"
    if len(sd) == 10:
        sd += " 00:00:00"
    return sd, ed


def _open(sql, sd, ed, inst, memory=None):
    """建连接和各表的视图；返回 (连接, 用到的表, 文件签名, SQL 正文)。"""
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory or _MEMORY}'")
    con.execute(f"SET temp_directory='{_TMP}'")
    used, sig = [], []
    if inst is not None:
        tmp_i = os.path.join(_TMP, f"dai_inst_{uuid.uuid4().hex}.parquet")
        pd.DataFrame({"instrument": [str(x) for x in inst]}).to_parquet(tmp_i, index=False)
        con.execute(f"CREATE TEMP TABLE _flt_inst AS SELECT * FROM read_parquet({_lit(tmp_i)})")
        os.remove(tmp_i)
    for name, kind in _TABLES.items():
        if not re.search(rf"\b{name}\b", sql, re.I):
            continue
        used.append(name)
        files = _files(kind, sd, ed)
        if not files:
            con.close()
            raise RuntimeError(f"{name} 在 {sd}~{ed} 没有本地数据")
        src = "read_parquet([" + ", ".join(_lit(f) for f in files) + "])"
        where = []
        if kind != "all_instruments":
            where.append(f"date >= TIMESTAMP {_lit(sd)} AND date <= TIMESTAMP {_lit(ed)}")
            if inst is not None:
                where.append("instrument IN (SELECT instrument FROM _flt_inst)")
        con.execute(f"CREATE TEMP VIEW {name} AS SELECT * FROM {src}" + (" WHERE " + " AND ".join(where) if where else ""))
        sig.append([name, [_file_sig(f) for f in files], where])
    return con, used, sig, sql.strip().rstrip(";")


def _cacheable(used, body):
    return "cpt_jyc_2026_stock_bar1m" in used and _CACHE_GB > 0 and not _NONDETERMINISTIC.search(body)


def _key(body, sig, inst):
    import duckdb
    return hashlib.sha256(json.dumps({"v": 1, "duckdb": duckdb.__version__, "sql": body, "tables": sig,
                                      "inst": None if inst is None else sorted(str(x) for x in inst)},
                                     ensure_ascii=False).encode("utf-8")).hexdigest()[:32]


def _execute(con, body, out):
    """经闸门执行 SQL 写到 out；返回在闸门前等待的秒数。线程数按正在执行的 SQL 个数分配。"""
    with _SqlSlot() as sl:
        con.execute(f"SET threads={max(2, _MAX_THREADS // max(1, sl.busy))}")
        con.execute(f"COPY ({body}) TO {_lit(out)} (FORMAT PARQUET)")
        return sl.waited


def _fill(con, body, path):
    tmp = path + f".{uuid.uuid4().hex}.tmp"
    try:
        w = _execute(con, body, tmp)
        os.replace(tmp, path)
        return w
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _prefetch(sql, sd, ed, inst):
    """后台预算下一块：同一个键已有人在算（锁不上）或已在缓存就跳过。任何异常都只记录。"""
    t0 = time.time()
    try:
        con, used, sig, body = _open(sql, sd, ed, inst, memory="6GB")
        try:
            if not _cacheable(used, body):
                return
            key = _key(body, sig, inst)
            path = os.path.join(_CACHE, key + ".parquet")
            with open(os.path.join(_CACHE, key + ".lock"), "w") as lk:
                try:
                    fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return
                if os.path.exists(path):
                    return
                _fill(con, body, path)
            _evict(path)
            print(f"[dai] 预取下一块 {sd} ~ {ed}：{time.time() - t0:.1f}s", file=sys.stderr, flush=True)
        finally:
            con.close()
    except Exception as e:
        print(f"[dai] 预取失败（不影响结果）：{type(e).__name__}: {e}", file=sys.stderr, flush=True)


def _maybe_prefetch(sql, sd, ed, inst):
    cur, nxt = _CHUNK["cur"], _CHUNK["next"]
    if not (_PREFETCH and cur and nxt):
        return
    try:
        off_s = pd.Timestamp(cur[0]) - pd.Timestamp(sd)      # 因子自己往前取的回看期
        off_e = pd.Timestamp(ed) - pd.Timestamp(cur[1])
        sd2 = (pd.Timestamp(nxt[0]) - off_s).strftime("%Y-%m-%d %H:%M:%S")
        ed2 = (pd.Timestamp(nxt[1]) + off_e).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return
    t = threading.Thread(target=_prefetch, args=(sql, sd2, ed2, inst), daemon=True, name="dai-prefetch")
    t.start()
    _PREFETCH_THREADS.append(t)


def query(sql, filters=None, **kwargs):  # compression 等参数不影响结果
    t0 = time.time()
    filters = dict(filters or {})
    sd, ed = _norm_dates(filters)
    inst = filters.get("instrument")
    for t in _UNSUPPORTED:
        if re.search(rf"\b{t}\b", sql, re.I):
            raise RuntimeError(f"本地模拟平台没有 {t}（只有 1 分钟行情 cpt_jyc_2026_stock_bar1m）")
    if re.search(r"\b(bid|ask)_(price|volume|num_orders)[45]\b", sql, re.I):
        WARNINGS.add("SQL 用到了 4–5 档盘口；本地数据只有 3 档，4–5 档为 NULL，结果与平台不同")
    con, used, sig, body = _open(sql, sd, ed, inst)
    cacheable = _cacheable(used, body)
    cache_state, slot_wait = None, 0.0
    try:
        if cacheable:
            os.makedirs(_CACHE, exist_ok=True)
            path = os.path.join(_CACHE, _key(body, sig, inst) + ".parquet")
            with open(path[:-8] + ".lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)              # 同键的其他任务 / 预取在这里等第一个算完
                if os.path.exists(path):
                    cache_state = "hit"
                    os.utime(path)
                else:
                    cache_state = "miss"
                    slot_wait = _fill(con, body, path)
                df = pd.read_parquet(path)
            if cache_state == "miss":
                _evict(path)
            _maybe_prefetch(sql, sd, ed, inst)
        else:
            out = os.path.join(_TMP, f"dai_{uuid.uuid4().hex}.parquet")
            try:
                slot_wait = _execute(con, body, out)
                df = pd.read_parquet(out)
            finally:
                if os.path.exists(out):
                    os.remove(out)
    finally:
        con.close()
    for c in df.columns:                   # 平台返回的时间列是 datetime64[ns]
        if str(df[c].dtype).startswith("datetime64"):
            df[c] = df[c].astype("datetime64[ns]")
    rec = {"tables": used, "start": sd, "end": ed, "instrument_filter": None if inst is None else len(inst),
           "rows": len(df), "seconds": round(time.time() - t0, 2), "cache": cache_state, "sql_slot_wait": round(slot_wait, 1)}
    LOG.append(rec)
    print(f"[dai] {rec}", file=sys.stderr, flush=True)
    return _Result(df)
