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

echo "## cuda-compiler"
nvcc --version 2>&1 || true

echo "## gpu-summary"
nvidia-smi --query-gpu=name,driver_version,compute_cap,pci.bus_id,memory.total --format=csv,noheader 2>&1 || true
