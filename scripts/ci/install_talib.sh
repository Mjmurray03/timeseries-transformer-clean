#!/usr/bin/env bash
set -euo pipefail

# Inputs
TA_VERSION="${TA_VERSION:-0.4.0}"
PREFIX="${PREFIX:-/usr/local}"

echo "[ci] Installing TA-Lib ${TA_VERSION} into ${PREFIX}"

# Prereqs
sudo apt-get update -y
sudo apt-get install -y build-essential wget ca-certificates

# Cache directory inside workspace (restored by actions/cache)
CACHE_DIR="${GITHUB_WORKSPACE:-$PWD}/.ci-cache/talib-${TA_VERSION}"
mkdir -p "${CACHE_DIR}"

if [ -f "${CACHE_DIR}/installed.marker" ]; then
  echo "[ci] TA-Lib already built in cache; syncing to ${PREFIX}"
  sudo rsync -a "${CACHE_DIR}/usr_local/" "${PREFIX}/"
  sudo ldconfig
  exit 0
fi

# Build from source
cd "$(mktemp -d)"
wget -q "https://prdownloads.sourceforge.net/ta-lib/ta-lib-${TA_VERSION}-src.tar.gz" -O ta-lib.tgz
tar -xzf ta-lib.tgz
cd "ta-lib-${TA_VERSION}"

./configure --prefix="${PREFIX}"
make -j"$(nproc)"
sudo make install
sudo ldconfig

# Save into cache snapshot
mkdir -p "${CACHE_DIR}/usr_local"
sudo rsync -a "${PREFIX}/" "${CACHE_DIR}/usr_local/"
touch "${CACHE_DIR}/installed.marker"

# Export include/lib for python wrapper if needed
echo "TA_LIBRARY_PATH=${PREFIX}/lib" >> "${GITHUB_ENV}"
echo "TA_INCLUDE_PATH=${PREFIX}/include" >> "${GITHUB_ENV}"
echo "LD_LIBRARY_PATH=${PREFIX}/lib:${LD_LIBRARY_PATH:-}" >> "${GITHUB_ENV}"

echo "[ci] TA-Lib ${TA_VERSION} installed."