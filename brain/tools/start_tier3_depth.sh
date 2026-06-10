#!/usr/bin/env bash
# Reliable launcher for the Tier-3 dense-depth sidecar. Self-logs so it works under a
# detached `screen` session (SSH-disconnect-proof):
#   screen -dmS tier3depth bash ~/duafoo/brain/tools/start_tier3_depth.sh
# Stop:  pkill -f tier3_depth_service   (or: screen -S tier3depth -X quit)
cd /home/duafoo/duafoo/brain || exit 1
exec ./.venv/bin/python -u tools/tier3_depth_service.py >> /tmp/tier3_depth.log 2>&1
