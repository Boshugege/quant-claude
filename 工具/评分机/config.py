# -*- coding: utf-8 -*-
"""评分机的路径与资源配置（环境变量可覆盖）。

磁盘布局（远端 ~/jyc_scorer，NVMe）：
  data/      原始输入：bar30m/*.feather、exposure/、labels_y_off.parquet、platform_probe/
  store/     预处理结果（prepare_data.py 生成，可删除重建）
    core/        评分核心：评估行 + 每个截面风格/行业回归的正交基（内存映射）
    panel/       localeval.build_panel 的结果，按月 parquet（因子脚本、已登记因子模式按需读取）
    platform/    模拟平台数据表：bar1m/YYYYMM.parquet（元、NaN、含 pre_close）、all_instruments、instruments、vwap
    refs/        参照池（factor_lib 中有公榜分数的因子）逐截面序列
  jobs/      每个任务一个目录：输入、日志、因子值、result.json、report.md
1 分钟原始 Feather 在机械盘（只在 prepare_data.py 转换时顺序读一次）。
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("JYC_DATA", ROOT / "data"))
STORE = Path(os.environ.get("JYC_STORE", ROOT / "store"))
JOBS = Path(os.environ.get("JYC_JOBS", ROOT / "jobs"))
RAW_1M = Path(os.environ.get("JYC_RAW_1M", "/srv/media/jyc_scorer/cpt_jyc_2026_e2e_bar1m"))

CORE_DIR = STORE / "core"
PANEL_DIR = STORE / "panel"
PLATFORM_DIR = STORE / "platform"
REFS_DIR = STORE / "refs"

FULL_START, FULL_END = "2020-01", "2024-12"
WARMUP_MONTHS = 6                 # 因子脚本 / 已登记因子按年分块计算时向前多取的月数（覆盖 60 个交易日回看）

CPU = os.cpu_count() or 8
ENGINE_THREADS = int(os.environ.get("JYC_ENGINE_THREADS", min(12, CPU)))
DUCKDB_THREADS = int(os.environ.get("JYC_DUCKDB_THREADS", min(6, CPU)))   # 每个任务；多个任务并行时总线程数不宜远超核数
DUCKDB_MEMORY = os.environ.get("JYC_DUCKDB_MEMORY", "12GB")
JOB_MEMORY_GB = int(os.environ.get("JYC_JOB_MEMORY_GB", 30))       # 单个代码类任务的内存上限（与平台 30 GB 一致）
JOB_TIMEOUT_S = int(os.environ.get("JYC_JOB_TIMEOUT_S", 3 * 3600)) # 与平台 3 小时一致

# 调度：按“槽位 + 内存预算”放行排队任务（先进先出；队首放不下时不让后面的插队，避免大任务饿死）。
# 平台型任务大多是单线程 pandas，实测峰值约 3–4 GB，所以槽位按核数、内存按实际占用控制，而不是固定只跑 2 个。
MAX_CODE_JOBS = int(os.environ.get("JYC_MAX_CODE_JOBS", 6))        # 槽位总数（12 个物理核）
MEM_BUDGET_GB = float(os.environ.get("JYC_MEM_BUDGET_GB", 28))     # 所有运行中任务的内存预算（容器上限 36 GB，留给服务进程和页缓存）
KIND_SLOTS = {"values": 0, "e2e": 3}                               # 未列出的类型占 1 个槽位；values 不占槽位（秒级）
# 预留内存；运行中按 max(预留, 实际 RSS) 计。平台型取 4.5 GB：截至 2026-10-11 的 41 个平台型任务峰值中位 2.8 GB、最大 4.0 GB。
# 可用环境变量 JYC_KIND_MEM_GB 覆盖（只需改 Quadlet 并重启，不用重建镜像）。Quadlet 里整条赋值要用单引号包住，
# 否则 systemd 会去掉 JSON 的双引号：Environment='JYC_KIND_MEM_GB={"platform":3.0,"e2e":12}'
# 解析失败时打印警告并沿用默认值，不会让服务起不来。
KIND_MEM_GB = {"values": 1, "sql": 4, "platform": 5, "factor_py": 8, "fid": 8, "e2e": 12}   # 没有历史样本时的默认预留；有样本时按历史峰值预测


def _kind_mem_override(raw):
    try:
        return {str(k): float(v) for k, v in json.loads(raw).items()}
    except Exception:
        try:                                     # 兼容被 systemd 去掉双引号的写法：{platform:3.0,e2e:12}
            pairs = [x.split(":") for x in raw.strip().strip("{}").split(",") if x.strip()]
            return {k.strip().strip("'\""): float(v) for k, v in pairs}
        except Exception:
            print(f"[config] 忽略无法解析的 JYC_KIND_MEM_GB={raw!r}，使用默认预留 {KIND_MEM_GB}", file=sys.stderr)
            return {}


KIND_MEM_GB.update(_kind_mem_override(os.environ.get("JYC_KIND_MEM_GB", "{}")))

SCORE_SLOTS = int(os.environ.get("JYC_SCORE_SLOTS", 2))                  # 服务进程内同时评分的任务数
SCORE_HEADROOM_GB = float(os.environ.get("JYC_SCORE_HEADROOM_GB", 3))    # 每个评分槽位给服务进程预留的内存
MAX_MEM_PRESSURE = float(os.environ.get("JYC_MAX_MEM_PRESSURE", 5))      # 容器内存压力（PSI some avg10，%）超过它时不放行新任务
LAUNCH_STAGGER_S = float(os.environ.get("JYC_LAUNCH_STAGGER_S", 90))  # 放行一个内存未知（新 SQL 家族）的任务后，等这么久、看到它的真实常驻再放下一个

# 公榜估计（refs.RefPool.pub_estimate）：锚定近邻。每个候选取标准化指标空间里最近的 PUB_KNN 个同快照锚点，
# 估计 = 锚点公榜分 + Σ PUB_BETA × (本地指标差)。PUB_BETA 是本地单位下的边际价值，由 2026-10-12 第二十七批同批 23 个公榜分拟合：
# 换手、IC 均值本地→公榜几乎 1:1（残差 0.002 / 0.0008），按拟合值全额计入；IC_IR、压力样本外残差大（0.022 / 0.039），减半；
# 多空夏普本地→公榜秩相关只有 0.59，不计入。留一法头部误差约 0.011，头部秩相关约 0.1–0.2（头部先后本地分不出来）。
PUB_BETA = {"ic_mean": 3.73, "ic_ir": 0.078, "ls_sharpe": 0.0, "stress": 0.094, "turnover": -0.514}
PUB_KNN = int(os.environ.get("JYC_PUB_KNN", 5))
PUB_HEAD = 0.75                   # “头部”锚点的公榜分门槛（报告头部误差用）
