#!/usr/bin/env bash
# 无损升级评分机（在评分机所在主机上运行；比赛期间的主机已停用）：构建镜像 → 排空（不再放行新任务，运行中的跑完）→ 换镜像重启。
# 排队中的任务保留在 jobs/ 里，重启后按原顺序继续；没有任务会被中途打断、从头重跑。
# 用法：bash ~/jyc_scorer/build/deploy.sh <镜像标签>      例如 2.4.3-abc1234
set -euo pipefail
TAG="${1:?用法：deploy.sh <镜像标签>}"
URL=http://127.0.0.1:46220
Q=~/.config/containers/systemd/jyc-scorer.container
cd ~/jyc_scorer/build

echo "== 构建 localhost/jyc-scorer:$TAG"
podman build -q -t "localhost/jyc-scorer:$TAG" -f Containerfile . | tail -1

running() { curl -s "$URL/api/health" | python3 -c "import sys,json;print(len(json.load(sys.stdin)['running']))"; }

echo "== 排空"
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H 'Content-Type: application/json' -d '{"on":true}' "$URL/api/admin/drain")
if [ "$code" != "200" ]; then
  echo "   当前版本没有排空接口（HTTP $code），只能等到某一刻没有运行中的任务"
fi
t0=$(date +%s)
while [ "$(running)" != "0" ]; do
  sleep 5
  echo "   等待运行中的任务结束：$(running) 个，已等 $(( $(date +%s) - t0 )) s"
done

echo "== 换镜像并重启"
# 直接改写 Quadlet 文件里的 Image=，不生成 .bak；需要回退时自行用版本管理恢复
sed -i "s#^Image=.*#Image=localhost/jyc-scorer:$TAG#" "$Q"
systemctl --user daemon-reload
systemctl --user restart jyc-scorer
for i in $(seq 1 60); do curl -sf "$URL/api/health" >/dev/null && break; sleep 2; done
curl -s "$URL/api/health" | python3 -c "import sys,json;d=json.load(sys.stdin);print('   已上线', d['version'], '运行', len(d['running']), '排队', len(d['queued']), '排空', d.get('draining'))"
