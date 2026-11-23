# 容器健康检查（Quadlet 的 HealthCmd 不适合带引号的命令）
import sys
import urllib.request

try:
    urllib.request.urlopen("http://127.0.0.1:46220/api/health", timeout=5)
except Exception:
    sys.exit(1)
