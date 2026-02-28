#!/bin/sh
cd /home/soliax/sites/gcp2-playbox
. .venv/bin/activate
python experiments/4-rolling-windows/gcp_egg_web_app.py &
python experiments/6-finance-correlations/main.py serve --host 0.0.0.0 --port 8050 &
