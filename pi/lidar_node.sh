#!/usr/bin/env bash
# Standalone RPLIDAR S2 node — runs SEPARATE from the senses (thin Pi → brain).
# The senses must NOT also be driving the lidar (ENABLE_LIDAR unset; default off),
# so only one process owns /dev/ttyUSB0.
#   Foreground:  ./lidar_node.sh
#   Background:  nohup ./lidar_node.sh > /tmp/jarvis-lidar.log 2>&1 &
cd "$(dirname "$0")" || exit 1
exec ./.venv/bin/python lidar_node.py
