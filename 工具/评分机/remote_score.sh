#!/usr/bin/env bash
# 本机一键评分：上传文件到评分机，等待完成，取回 report.md / result.json 到 工具/评分机/结果/<任务号>/，并打印报告。
# 用法：bash 工具/评分机/remote_score.sh <文件...> [-- start=2024-01 end=2024-12 kind=auto label=y_off chunk=none name=xxx fid=104]
# 不需要建隧道：在远端用 curl 访问 127.0.0.1 上的服务。远端别名默认 personal（环境变量 JYC_SCORER_HOST）。
# 默认别名 personal 是比赛期间的评分机主机，已停用；自己部署评分机后用 JYC_SCORER_HOST、JYC_SCORER_REMOTE_DIR 指向自己的主机和安装目录。
# 评分机没有身份认证并且会执行上传的代码，只能监听 127.0.0.1。
set -euo pipefail
HOST="${JYC_SCORER_HOST:-personal}"
BASE="${JYC_SCORER_REMOTE_DIR:-jyc_scorer}"
OUT="${JYC_LOCAL_RESULTS:-$(cd "$(dirname "$0")" && pwd)/结果}"
files=(); opts=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then shift; opts=("$@"); break; fi
  files+=("$1"); shift
done
TAG="upload_$(date +%s)_$$"
ssh -o BatchMode=yes "$HOST" "mkdir -p $BASE/incoming/$TAG"
args=""
for f in "${files[@]+"${files[@]}"}"; do
  scp -q "$f" "$HOST:$BASE/incoming/$TAG/"
  args+=" -F 'files=@$BASE/incoming/$TAG/$(basename "$f")'"
done
for o in "${opts[@]+"${opts[@]}"}"; do args+=" -F '$o'"; done
echo "提交到 $HOST ……（等待完成，期间可在网页“任务”里看实时日志）"
JSON=$(ssh "$HOST" "bash -c \"curl -s $args 'http://127.0.0.1:46220/api/jobs?wait=true'\"")
JID=$(python3 -c 'import json,sys;d=json.loads(sys.stdin.read());print(d.get("id") or d.get("job_id") or "")' <<<"$JSON")
STATUS=$(python3 -c 'import json,sys;d=json.loads(sys.stdin.read());print(d.get("status"),d.get("error") or d.get("detail") or "")' <<<"$JSON")
ssh "$HOST" "rm -rf $BASE/incoming/$TAG"
if [[ -z "$JID" ]]; then echo "提交失败：$STATUS"; exit 1; fi
mkdir -p "$OUT/$JID"
for n in report.md result.json log.txt; do scp -q "$HOST:$BASE/jobs/$JID/$n" "$OUT/$JID/" 2>/dev/null || true; done
echo "任务 $JID：$STATUS；结果在 $OUT/$JID/"
[[ -f "$OUT/$JID/report.md" ]] && cat "$OUT/$JID/report.md" || tail -30 "$OUT/$JID/log.txt"
