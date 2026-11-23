# -*- coding: utf-8 -*-
"""内存剖析：逐月构建训练样本，记录进程 RSS 与已持有样本数据量，估计五年全量的内存峰值。
用法：python e2e/scripts/mem_profile.py <fields> <coarse_len> <n_months>
"""
import gc
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend, set_read_cols, read_month  # noqa: E402


def rss():
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) / 1e6


fields, lc, nm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
T.set_fields(fields)
T.COARSE_LEN = lc
set_read_cols(T.FEATURE_COLS)
fb = FeatherBackend()
end = f"2020-{nm:02d}-28" if nm <= 12 else "2021-06-30"
inst = fb.pool("2020-01-01", end)
held, nbytes = [], 0
t0 = time.time()
print(f"start rss={rss():.2f}GB fields={fields} coarse={lc}", flush=True)
for Xf, Xc, y, k in T.iter_chunks(fb, "2020-01-01", end + " 23:59:59", inst, True):
    held.append((Xf, Xc, y, k))
    nbytes += Xf.nbytes + (Xc.nbytes if Xc is not None else 0) + y.nbytes
    gc.collect()
    print(f"{str(k['date'].iloc[0])[:7]} samples={len(k):,} held_data={nbytes / 1e9:.2f}GB rss={rss():.2f}GB "
          f"cache={read_month.cache_info().currsize} t={time.time() - t0:.0f}s", flush=True)
n = sum(len(h[3]) for h in held)
per = nbytes / n
print(f"bytes/sample={per:.0f}; 5y (~8.5M samples) data≈{8.5e6 * per / 1e9:.1f}GB")
