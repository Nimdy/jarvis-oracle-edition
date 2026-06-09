"""Standalone RPLIDAR S2 node — a SEPARATE process from the senses.

Architecture (David's principle, matches the validated design): the Pi is thin
EYES — it reads the sensor and streams to the brain; the brain (desktop CPU/GPU)
does ALL the world-modeling. Running the lidar as its own process means the
1 Mbaud read isn't GIL-starved by the senses' Hailo vision + audio streaming
(which desynced the in-thread reader). It opens its own WebSocket to the brain's
perception server with a distinct sensor_id and streams scan_2d telemetry — exactly
how the senses already stream scene/vision data for world modeling.

Run:  ~/duafoo/pi/lidar_node.sh      (or: .venv/bin/python lidar_node.py)
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal

from config import SensesConfig
from transport.ws_client import TransportClient
from transport.event_schema import lidar_scan
from senses.lidar.lidar_sensor import LidarSensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("jarvis.lidar")


async def main() -> None:
    cfg = SensesConfig()
    transport = TransportClient(
        host=cfg.transport.brain_host,
        port=cfg.transport.brain_port,
        sensor_id="pi-lidar",   # distinct from the main senses node
        reconnect_interval=cfg.transport.reconnect_interval_s,
        buffer_max=cfg.transport.buffer_max_events,
    )
    await transport.start()
    logger.info("Lidar node connected to brain %s:%d as 'pi-lidar'",
                cfg.transport.brain_host, cfg.transport.brain_port)

    sensor = LidarSensor(
        emit=lambda s: transport.send_event(lidar_scan(**s)),
        port=os.environ.get("LIDAR_PORT", "/dev/ttyUSB0"),
    )
    sensor.start()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    try:
        await stop.wait()
    finally:
        logger.info("Lidar node shutting down")
        sensor.stop()
        await transport.stop()


if __name__ == "__main__":
    asyncio.run(main())
