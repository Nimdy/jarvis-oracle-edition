# RPLIDAR A1M8 → JARVIS (Pi 5)

Adds a **new sense**: the camera/Hailo tells JARVIS *what it sees*; the RPLIDAR tells it the
**2D spatial shape** around it. Telemetry-only by design — it informs nearest-obstacle /
open-space, never "I know what that object is" or canonical spatial memory.

> Path: `RPLIDAR A1M8 → bundled USB adapter → Pi 5 USB → Python reader → Pi perception
> telemetry → brain world model`. **No ROS** until you're doing SLAM/navigation.

## 1. Wiring + permissions (do this, not `chmod 777`)

```bash
sudo usermod -a -G dialout $USER
sudo reboot
# after reboot:
ls /dev/ttyUSB*        # expect /dev/ttyUSB0
dmesg | grep tty
```

## 2. Python (in a venv)

```bash
python3 -m venv ~/venvs/rplidar
source ~/venvs/rplidar/bin/activate
pip install adafruit-circuitpython-rplidar websockets
```

## 3. Prove hardware (Stage 1)

Run the bundled Adafruit smoke test (`lidar.info`, `lidar.health`, `iter_scans()`). Want stable
scans for 5–10 minutes before going further.

## 4. The JARVIS reader (Stages 2–3): sectors → brain telemetry

This reads scans, reduces them to **6 sectors** (nearest distance each), computes open sectors +
scan quality, and publishes a `lidar_scan` event to the brain's perception WebSocket. The brain
already handles `lidar_scan` (telemetry-only; `perception/server.py`) and surfaces it at
`/api/pi5` + the `/v2/pi5` dashboard.

```python
# rplidar_reader.py  — run on the Pi (telemetry-only)
import asyncio, json, time
from math import floor
from adafruit_rplidar import RPLidar
import websockets

PORT = "/dev/ttyUSB0"
BRAIN_WS = "ws://192.168.1.222:9100"      # brain perception bus
SENSOR_ID = "rplidar-a1m8"
OPEN_M = 2.0                              # sector counts as "open" if nearest > 2 m
DEGRADED_MIN_POINTS = 120

# 6 sectors of 60°, angle in degrees (0 = front of the unit; adjust to your mounting)
SECTORS = {
    "front":       lambda a: a >= 330 or a < 30,
    "front_left":  lambda a: 30 <= a < 90,
    "left":        lambda a: 90 <= a < 150,
    "back":        lambda a: 150 <= a < 210,
    "right":       lambda a: 210 <= a < 270,
    "front_right": lambda a: 270 <= a < 330,
}

def summarize(scan_data_mm):
    """scan_data_mm: list[360] of mm (0 = no reading). Return the sector summary."""
    nearest = {k: None for k in SECTORS}
    points = 0
    for deg, mm in enumerate(scan_data_mm):
        if mm <= 0:
            continue
        points += 1
        m = mm / 1000.0
        for name, in_sec in SECTORS.items():
            if in_sec(deg):
                if nearest[name] is None or m < nearest[name]:
                    nearest[name] = round(m, 2)
    open_sectors = [k for k, v in nearest.items() if v is not None and v > OPEN_M]
    quality = "healthy" if points >= DEGRADED_MIN_POINTS else "degraded"
    return nearest, open_sectors, points, quality

async def run():
    lidar = RPLidar(None, PORT, timeout=3)
    scan_data = [0] * 360
    last_pub = 0.0
    try:
        async with websockets.connect(
            BRAIN_WS, additional_headers={"x-sensor-id": SENSOR_ID}
        ) as ws:
            t0 = time.time()
            scans = 0
            for scan in lidar.iter_scans():
                for _, angle, distance in scan:
                    scan_data[min(359, max(0, floor(angle)))] = distance
                scans += 1
                now = time.time()
                if now - last_pub < 0.3:          # publish ~3 Hz (don't flood)
                    continue
                last_pub = now
                sectors, open_sectors, points, quality = summarize(scan_data)
                hz = round(scans / max(0.001, now - t0), 1)
                await ws.send(json.dumps({
                    "type": "lidar_scan",
                    "data": {
                        "sensor": "rplidar_a1m8",
                        "scan_hz": hz,
                        "points": points,
                        "range_max_m": 12,
                        "sectors": sectors,
                        "open_sectors": open_sectors,
                        "scan_quality": quality,
                        # contract — the brain enforces this anyway:
                        "authority": "spatial_telemetry_only",
                        "writes_beliefs": False,
                    },
                }))
    except KeyboardInterrupt:
        pass
    finally:
        lidar.stop(); lidar.disconnect()

if __name__ == "__main__":
    asyncio.run(run())
```

## 5. See it (Stage 4)

Open **`/v2/pi5`** on the brain dashboard — the RPLIDAR appears under *Connected Senses* with its
sector ring (front / left / right / back nearest, scan Hz, points, quality, `writes_beliefs: no`).
This is how you confirm new equipment is wired correctly.

## 6. /mind 2D ring (Stage 5, later)

Feed the sector summary into the spatial `/mind` view to draw the 2D floor-plane ring — where this
sensor really shines.

## Authority / honesty contract

The brain stores LIDAR as `spatial_telemetry_only`, `writes_beliefs: false`. The world model may
**derive** nearest-by-sector, open-space, room outline, change-over-time, and "2D spatial scan
available." It must **not** derive "I know what that object is", a full 3D mesh, or canonical
spatial memory. New sense, earned authority — same discipline as every other JARVIS capability.
