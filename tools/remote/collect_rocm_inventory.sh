#!/usr/bin/env bash
set -uo pipefail

# Public inventory output is intentionally allowlisted. Do not add broad host,
# filesystem, process, network, environment, or device-identifier dumps.
echo "## captured-at"
date -u +"%Y-%m-%dT%H:%M:%SZ"

echo "## kernel"
printf 'system: '
uname -s
printf 'release: '
uname -r
printf 'architecture: '
uname -m

echo "## operating-system"
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  printf 'id: %s\n' "${ID:-not_available}"
  printf 'version_id: %s\n' "${VERSION_ID:-not_available}"
fi

echo "## cpu-topology"
lscpu --parse=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ 2>&1 || true

echo "## memory-capacity"
free --bytes --total 2>&1 || true

echo "## hip-compiler"
hipcc --version 2>&1 || /opt/rocm/bin/hipcc --version 2>&1 || true

echo "## rocm-device-summary"
rocm-smi --showproductname --showdriverversion --showmeminfo vram --showpower --showclocks 2>&1 \
  || /opt/rocm/bin/rocm-smi --showproductname --showdriverversion --showmeminfo vram --showpower --showclocks 2>&1 \
  || true
