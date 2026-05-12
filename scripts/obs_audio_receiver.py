#!/usr/bin/env python3
"""OBS Audio Receiver — receives Jarvis TTS audio via TCP and plays through
a virtual audio device that OBS can capture.

Run on your streaming/OBS PC (NOT on the brain or Pi).

=== Windows Setup ===
1. Install VB-Audio Virtual Cable: https://vb-audio.com/Cable/
2. Install Python deps:   py -m pip install sounddevice numpy
3. Find CABLE Input device number:
       py -c "import sounddevice as sd; print(sd.query_devices())"
4. Run:   py obs_audio_receiver.py --device DEVICE_NUMBER
5. In OBS: Sources → Audio Input Capture → "CABLE Output (VB-Audio Virtual Cable)"
6. Firewall (PowerShell admin):
       New-NetFirewallRule -DisplayName "Jarvis OBS Audio TCP 5004" `
           -Direction Inbound -Protocol TCP -LocalPort 5004 -Action Allow

=== Linux Setup ===
1. Run:   python obs_audio_receiver.py --linux
   (auto-creates a PulseAudio virtual sink called "jarvis_voice")
2. In OBS: Audio Input Capture → "Monitor of Jarvis_Voice"

=== Brain Setup (.env) ===
    OBS_AUDIO_TARGET=YOUR_OBS_PC_IP:5004

Dependencies:
    Windows: sounddevice, numpy (pip install)
    Linux:   pactl/pacat (system PulseAudio/PipeWire, no pip needed)
"""

import argparse
import io
import socket
import struct
import subprocess
import sys
import threading
import wave


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from a TCP socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf.extend(chunk)
    return bytes(buf)


def run_windows(port: int, device: int | None):
    """Windows path: TCP → sounddevice → VB-Cable Input → OBS captures Cable Output."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        print("ERROR: Install deps first:  py -m pip install sounddevice numpy", file=sys.stderr)
        sys.exit(1)

    if device is not None:
        dev_info = sd.query_devices(device)
        print(f"  Output device #{device}: {dev_info['name']}")
    else:
        devices = sd.query_devices()
        cable_idx = None
        for i, d in enumerate(devices):
            if "cable input" in d["name"].lower() and d["max_output_channels"] > 0:
                cable_idx = i
                break
        if cable_idx is not None:
            device = cable_idx
            print(f"  Auto-detected CABLE Input: device #{device}")
        else:
            print("ERROR: Could not find 'CABLE Input'. Install VB-Audio Virtual Cable")
            print("       or specify --device NUMBER manually.")
            print("\nAvailable output devices:")
            for i, d in enumerate(devices):
                if d["max_output_channels"] > 0:
                    print(f"  {i}: {d['name']} (out={d['max_output_channels']}ch)")
            sys.exit(1)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(1)
    print(f"  Listening on TCP :{port} — waiting for Jarvis brain to connect...")

    while True:
        conn, addr = server.accept()
        print(f"  Connected: {addr[0]}:{addr[1]}")
        chunks_played = 0

        try:
            while True:
                header = _recv_exact(conn, 4)
                length = struct.unpack(">I", header)[0]
                if length == 0 or length > 10_000_000:
                    print(f"  Bad frame length: {length}, skipping")
                    continue

                wav_bytes = _recv_exact(conn, length)

                try:
                    wf = wave.open(io.BytesIO(wav_bytes), "rb")
                    pcm = wf.readframes(wf.getnframes())
                    rate = wf.getframerate()
                    channels = wf.getnchannels()
                    n_frames = wf.getnframes()
                    wf.close()
                except Exception as exc:
                    print(f"  Bad WAV data ({length} bytes): {exc}")
                    continue

                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    samples = samples.reshape(-1, channels)
                else:
                    samples = samples.reshape(-1, 1)

                try:
                    sd.play(samples, samplerate=rate, device=device, blocking=True)
                    chunks_played += 1
                    dur_ms = n_frames / rate * 1000
                    if chunks_played <= 5 or chunks_played % 10 == 0:
                        print(f"  Chunk #{chunks_played}: {dur_ms:.0f}ms @ {rate}Hz ({length // 1024}KB)")
                except Exception as exc:
                    print(f"  Playback error: {exc}", file=sys.stderr)

        except ConnectionError:
            print(f"  Disconnected after {chunks_played} chunks. Waiting for reconnect...")
        except Exception as exc:
            print(f"  Error: {exc}. Waiting for reconnect...")
        finally:
            conn.close()


def run_linux(port: int, sink: str):
    """Linux path: TCP → pacat → PulseAudio virtual sink → OBS captures monitor."""
    _ensure_sink_linux(sink)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(1)
    print(f"  Listening on TCP :{port} — waiting for Jarvis brain to connect...")

    while True:
        conn, addr = server.accept()
        print(f"  Connected: {addr[0]}:{addr[1]}")

        pacat_proc = None
        pacat_rate = None
        pacat_channels = None

        try:
            while True:
                header = _recv_exact(conn, 4)
                length = struct.unpack(">I", header)[0]
                if length == 0 or length > 10_000_000:
                    continue

                wav_bytes = _recv_exact(conn, length)

                try:
                    wf = wave.open(io.BytesIO(wav_bytes), "rb")
                    pcm = wf.readframes(wf.getnframes())
                    rate = wf.getframerate()
                    channels = wf.getnchannels()
                    wf.close()
                except Exception:
                    continue

                if (pacat_proc is None or pacat_proc.poll() is not None
                        or rate != pacat_rate or channels != pacat_channels):
                    if pacat_proc and pacat_proc.poll() is None:
                        pacat_proc.stdin.close()
                        pacat_proc.wait(timeout=2)
                    pacat_proc = subprocess.Popen(
                        [
                            "pacat", "--playback",
                            f"--device={sink}",
                            f"--rate={rate}",
                            f"--channels={channels}",
                            "--format=s16le",
                            "--latency-msec=50",
                        ],
                        stdin=subprocess.PIPE,
                    )
                    pacat_rate = rate
                    pacat_channels = channels
                    print(f"  pacat: {rate}Hz {channels}ch → {sink}")

                try:
                    pacat_proc.stdin.write(pcm)
                    pacat_proc.stdin.flush()
                except BrokenPipeError:
                    pacat_proc = None

        except ConnectionError:
            print(f"  Disconnected. Waiting for reconnect...")
        except Exception as exc:
            print(f"  Error: {exc}. Waiting for reconnect...")
        finally:
            conn.close()
            if pacat_proc and pacat_proc.poll() is None:
                pacat_proc.stdin.close()
                pacat_proc.wait(timeout=2)


def _ensure_sink_linux(sink_name: str):
    """Check if the virtual sink exists, create it if not."""
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True, text=True, check=True,
        )
        if sink_name in result.stdout:
            print(f"  Virtual sink '{sink_name}' found")
            return
    except FileNotFoundError:
        print("ERROR: pactl not found. Install PulseAudio or PipeWire.", file=sys.stderr)
        sys.exit(1)

    print(f"  Creating virtual sink '{sink_name}'...")
    try:
        subprocess.run(
            [
                "pactl", "load-module", "module-null-sink",
                f"sink_name={sink_name}",
                f'sink_properties=device.description="Jarvis_Voice"',
                "rate=24000", "channels=1", "format=s16le",
            ],
            check=True,
        )
        print(f"  In OBS: Audio Input Capture → 'Monitor of Jarvis_Voice'")
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: Could not create sink: {exc}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Receive Jarvis TTS audio via TCP for OBS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=5004,
                        help="TCP port to listen on (default: 5004)")
    parser.add_argument("--device", type=int, default=None,
                        help="Windows: sounddevice output device number for CABLE Input")
    parser.add_argument("--linux", action="store_true",
                        help="Use Linux PulseAudio/pacat instead of sounddevice")
    parser.add_argument("--sink", default="jarvis_voice",
                        help="Linux: PulseAudio sink name (default: jarvis_voice)")

    args = parser.parse_args()

    print("=== Jarvis OBS Audio Receiver ===")

    is_windows = sys.platform == "win32"

    if args.linux or not is_windows:
        run_linux(args.port, args.sink)
    else:
        run_windows(args.port, args.device)


if __name__ == "__main__":
    main()
