"""Jarvis Pi 5 Senses — Configuration (thin sensor node)."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel

_PI_ROOT = str(Path(__file__).resolve().parent)


def _load_dotenv(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv(os.path.join(_PI_ROOT, ".env"))

_HAILO_SYSTEM_DIR = "/usr/share/hailo-models"


class VisionConfig(BaseModel):
    camera_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 15
    detection_model: str = os.getenv("DETECTION_MODEL", "yolov8s")
    detection_threshold: float = 0.5
    face_model: str = "scrfd_2.5g"
    face_threshold: float = 0.4
    enable_tracking: bool = True
    enable_expressions: bool = False


class AudioConfig(BaseModel):
    mic_sample_rate: int = 48000
    channels: int = 1
    mic_name: str = ""
    speaker_name: str = ""


class PoseConfig(BaseModel):
    enabled: bool = True
    model_path: str = "yolov8s_pose.hef"
    threshold: float = 0.4


class TransportConfig(BaseModel):
    brain_host: str = "localhost"
    brain_port: int = 9100
    reconnect_interval_s: float = 3.0
    buffer_max_events: int = 500
    sensor_id: str = "pi5-senses"


class UIConfig(BaseModel):
    enabled: bool = True
    port: int = 8080
    host: str = "0.0.0.0"


class SensesConfig(BaseModel):
    vision: VisionConfig = VisionConfig()
    audio: AudioConfig = AudioConfig()
    pose: PoseConfig = PoseConfig()
    transport: TransportConfig = TransportConfig()
    ui: UIConfig = UIConfig()
    enable_vision: bool = True
    enable_audio: bool = True
    log_level: str = "INFO"
    project_root: str = _PI_ROOT

    def model_post_init(self, __context):
        models_dir = os.path.join(self.project_root, "models")

        if not self.pose.model_path or not os.path.isabs(self.pose.model_path):
            pose_local = os.path.join(models_dir, "yolov8s_pose.hef")
            pose_sys = os.path.join(_HAILO_SYSTEM_DIR, "yolov8s_pose.hef")
            if os.path.exists(pose_local):
                self.pose.model_path = pose_local
            elif os.path.exists(pose_sys):
                self.pose.model_path = pose_sys

        for attr in ("detection_model", "face_model"):
            name = getattr(self.vision, attr)
            if os.path.isabs(name):
                continue
            local_hef = os.path.join(models_dir, f"{name}.hef")
            sys_hef = os.path.join(_HAILO_SYSTEM_DIR, f"{name}.hef")
            if os.path.exists(local_hef):
                setattr(self.vision, attr, os.path.join(models_dir, name))
            elif os.path.exists(sys_hef):
                setattr(self.vision, attr, os.path.join(_HAILO_SYSTEM_DIR, name))

        brain_host = os.getenv("BRAIN_HOST")
        if brain_host:
            self.transport.brain_host = brain_host
        brain_port = os.getenv("BRAIN_PORT")
        if brain_port:
            self.transport.brain_port = int(brain_port)
        mic_name = os.getenv("MIC_NAME")
        if mic_name:
            self.audio.mic_name = mic_name
        speaker_name = os.getenv("SPEAKER_NAME")
        if speaker_name:
            self.audio.speaker_name = speaker_name
