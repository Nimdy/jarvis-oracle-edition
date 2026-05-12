"""System status tool — CPU, RAM, GPU, uptime."""

from __future__ import annotations

import os
import platform
import shutil
import time


def get_system_status() -> str:
    parts: list[str] = []

    # Uptime
    try:
        with open("/proc/uptime") as f:
            uptime_s = float(f.read().split()[0])
        hours = int(uptime_s // 3600)
        minutes = int((uptime_s % 3600) // 60)
        parts.append(f"Uptime: {hours}h {minutes}m")
    except Exception:
        pass

    # CPU load
    try:
        load1, load5, load15 = os.getloadavg()
        parts.append(f"CPU load: {load1:.1f} / {load5:.1f} / {load15:.1f}")
    except Exception:
        pass

    # RAM
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem_info: dict[str, int] = {}
        for line in lines:
            key, val = line.split(":")
            mem_info[key.strip()] = int(val.strip().split()[0])
        total_gb = mem_info.get("MemTotal", 0) / 1_048_576
        avail_gb = mem_info.get("MemAvailable", 0) / 1_048_576
        used_gb = total_gb - avail_gb
        parts.append(f"RAM: {used_gb:.1f}GB / {total_gb:.1f}GB")
    except Exception:
        pass

    # Disk
    try:
        usage = shutil.disk_usage("/")
        used_gb = usage.used / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        parts.append(f"Disk: {used_gb:.0f}GB / {total_gb:.0f}GB")
    except Exception:
        pass

    # GPU (NVIDIA)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(", ")
            if len(gpu_info) >= 5:
                parts.append(f"GPU: {gpu_info[0]}, {gpu_info[1]}°C, {gpu_info[2]}% util, {gpu_info[3]}MB/{gpu_info[4]}MB")
    except Exception:
        pass

    parts.append(f"Platform: {platform.system()} {platform.release()}")
    return ". ".join(parts) + "."
