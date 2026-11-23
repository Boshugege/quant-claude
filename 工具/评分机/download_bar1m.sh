#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-/srv/media/jyc_scorer/bar1m}"
URL="https://bigquant-public-download.tos-cn-shanghai.volces.com/jyc2026/cpt_jyc_2026_e2e_bar1m.tar.gz"
mkdir -p "$DEST"
cd "$DEST"
curl -L --fail --retry 8 --retry-delay 5 --continue-at - -o bar1m.tar.gz "$URL"
tar -xzf bar1m.tar.gz
rm -f bar1m.tar.gz
chmod 644 cpt_jyc_2026_e2e_bar1m/*
find cpt_jyc_2026_e2e_bar1m -type f | sort | wc -l
du -sh cpt_jyc_2026_e2e_bar1m
