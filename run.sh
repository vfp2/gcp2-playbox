#!/bin/sh
cd /home/soliax/sites/gcp2-playbox
. .venv/bin/activate
# 8051 is taken by heartmath-collab-gcp-visualizer on this host; ikijima proxies gcpeggs.fp2.dev -> :8052
GCP_EGG_PORT=8052 python experiments/4-rolling-windows/gcp_egg_web_app.py &
python experiments/6-finance-correlations/main.py serve --host 0.0.0.0 --port 8050 &
