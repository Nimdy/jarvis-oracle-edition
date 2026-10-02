#!/usr/bin/env bash
# Install the Jarvis desktop fan curves on a fresh Pop!_OS load.
# Users running Pop!_OS should follow this guide. It may work on other distributions, but they are not officially supported.
# This script can be updated based on your hardware configuration and fan preferences.
# Machine: ASUS ROG STRIX X670E-E GAMING WIFI, Ryzen 9 7950X.
# Fans are the onboard Nuvoton NCT6799, not a USB AIO controller.
#
# Copy this file onto the fresh install and run:
#   sudo ./setup-desktop-fans.sh
#
# Re-running is safe. It writes the same curve the 2026-10-02 reboot
# already confirmed, then enables it for every later boot.
#
# What it does:
#   fan7 was locked at PWM 255 (about 2150 RPM). It now idles near 900 RPM.
#   fan2 stalls below PWM 80, so its floor is 110 (about 1900 RPM).
#   Chassis fans 1,3,4,5,6 idle near 750 RPM and climb with the CPU.
#   The control sensor is temp 8, "PECI/TSI Agent 0 Calibration".
#   It tracks CPU Tctl about 10C low. TSI0 matches Tctl but this chip
#   cannot use it for fan control. Do not write pwm*_floor to a non-zero
#   value: that arms the "fan may stop" bit.
#
# BIOS still runs the fans hard for a few seconds during POST. This
# service pulls them back down once the system is up.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root: sudo $0" >&2
  exit 1
fi

BOARD=$(cat /sys/class/dmi/id/board_name 2>/dev/null || true)
if [ "$BOARD" != "ROG STRIX X670E-E GAMING WIFI" ] && [ "${1:-}" != "--force" ]; then
  echo "This curve is for ROG STRIX X670E-E GAMING WIFI. This board is: ${BOARD:-unknown}" >&2
  echo "Pass --force only if you mean to install it here anyway." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y lm-sensors

if ! modprobe nct6775; then
  echo "nct6775 did not load. If dmesg says the Super I/O ports conflict with ACPI:" >&2
  echo "  kernelstub -a acpi_enforce_resources=lax && reboot" >&2
  echo "Then run this script again." >&2
  exit 1
fi

install -d -m 0755 /usr/local/sbin /etc/modules-load.d /etc/systemd/system

printf '%s\n' nct6775 > /etc/modules-load.d/nct6775.conf

cat > /usr/local/sbin/jarvis-fan-curve.sh << 'SCRIPT'
#!/bin/bash
# Smart Fan IV on the ASUS X670E-E NCT6799.
# pwm_temp_sel 8 is "PECI/TSI Agent 0 Calibration". It tracks CPU Tctl
# about 10C low and is the hottest source this chip will accept.
# TSI0 (temp13) matches Tctl exactly but the fan engine cannot select it.
# Do not write pwm*_floor: a non-zero write sets the "fan may stop" bit.
set -euo pipefail

modprobe nct6775

H=""
for _try in $(seq 1 20); do
  for d in /sys/class/hwmon/hwmon*; do
    name=$(cat "$d/name" 2>/dev/null || true)
    if [ "$name" = "nct6799" ]; then
      H=$d
      break
    fi
  done
  [ -n "$H" ] && break
  sleep 1
done
if [ -z "$H" ]; then
  echo "nct6799 hwmon not found" >&2
  exit 1
fi

BAK=/var/lib/jarvis-fan-curve/bios-defaults.txt

dump_state() {
  local i p
  for i in 1 2 3 4 5 6 7; do
    echo "pwm${i}_enable=$(cat "$H/pwm${i}_enable")"
    echo "pwm${i}_mode=$(cat "$H/pwm${i}_mode")"
    echo "pwm${i}_temp_sel=$(cat "$H/pwm${i}_temp_sel")"
    echo "pwm${i}_floor=$(cat "$H/pwm${i}_floor")"
    for p in 1 2 3 4 5; do
      echo "pwm${i}_auto_point${p}_temp=$(cat "$H/pwm${i}_auto_point${p}_temp")"
      echo "pwm${i}_auto_point${p}_pwm=$(cat "$H/pwm${i}_auto_point${p}_pwm")"
    done
  done
}

restore_backup() {
  [ -f "$BAK" ] || return 1
  while IFS='=' read -r key val; do
    case "$key" in
      ''|\#*) continue ;;
    esac
    echo "$val" > "$H/$key" || true
  done < "$BAK"
}

if [ "${1:-}" = "--restore" ]; then
  restore_backup
  echo "restored BIOS fan defaults from $BAK"
  exit 0
fi

if [ ! -f "$BAK" ]; then
  install -d -m 0755 /var/lib/jarvis-fan-curve
  {
    echo "# captured $(date -Is) before the first jarvis-fan-curve apply"
    dump_state
  } > "$BAK"
  chmod 644 "$BAK"
fi

# Undo an earlier floor write on fan1 that armed "fan may stop".
echo 0 > "$H/pwm1_floor" || true

apply_points() {
  local i=$1
  shift
  local p=1
  while [ $# -ge 2 ]; do
    echo "$2" > "$H/pwm${i}_auto_point${p}_pwm"
    echo "$1" > "$H/pwm${i}_auto_point${p}_temp"
    p=$((p + 1))
    shift 2
  done
  echo 1 > "$H/pwm${i}_mode"
  echo 8 > "$H/pwm${i}_temp_sel"
  echo 5 > "$H/pwm${i}_enable"
}

# Chassis. Idle stays near the old ~800 RPM and they now climb with CPU load.
for i in 1 3 4 5 6; do
  apply_points "$i" \
    30000 55 \
    50000 100 \
    65000 160 \
    78000 210 \
    88000 255
done

# Fast header. Measured stall at PWM 80, stable at 100. Floor is 110.
apply_points 2 \
  28000 110 \
  45000 150 \
  60000 195 \
  75000 230 \
  88000 255

# Header that BIOS had locked at PWM 255.
apply_points 7 \
  28000 60 \
  45000 105 \
  60000 160 \
  75000 215 \
  88000 255

sleep 6
bad=0
temp8=$(cat "$H/temp8_input")
echo "hwmon=$H cpu_sensor_temp8=${temp8}"
for i in 1 2 3 4 5 6 7; do
  rpm=$(cat "$H/fan${i}_input")
  pwm=$(cat "$H/pwm${i}")
  sel=$(cat "$H/pwm${i}_temp_sel")
  en=$(cat "$H/pwm${i}_enable")
  echo "fan${i} rpm=${rpm} pwm=${pwm} temp_sel=${sel} enable=${en}"
  if [ "$rpm" -eq 0 ]; then
    bad=1
  fi
done
if [ "$bad" -ne 0 ]; then
  echo "a fan stopped; restoring BIOS defaults" >&2
  restore_backup
  exit 1
fi
logger -t jarvis-fan-curve "applied NCT6799 curves on $H temp8=${temp8}"
SCRIPT
chmod 755 /usr/local/sbin/jarvis-fan-curve.sh

cat > /etc/systemd/system/jarvis-fan-curve.service << 'UNIT'
[Unit]
Description=Apply motherboard AIO and CPU fan curves
After=systemd-modules-load.service
Wants=systemd-modules-load.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/jarvis-fan-curve.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now jarvis-fan-curve.service

echo
echo "Fan curve installed and enabled for boot."
systemctl --no-pager --full status jarvis-fan-curve.service | head -15
