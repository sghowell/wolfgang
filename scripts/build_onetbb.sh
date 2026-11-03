#!/usr/bin/env bash
# Build the oneTBB runtime that release wheels bundle, from a pinned commit.
#
# Usage: scripts/build_onetbb.sh <install-prefix>
#
# cibuildwheel runs this once per platform (before-all). The clone is checked
# against ONETBB_COMMIT, so a moved tag fails the build instead of shipping
# different code. Only the core tbb library is built: Wolfgang links TBB::tbb
# and nothing else. auditwheel (Linux) or delocate (macOS) then copies the
# shared library into each wheel; the license files are copied next to the
# install so CMake can place them in the wheel.
set -euo pipefail

ONETBB_TAG="v2022.3.0"
ONETBB_COMMIT="f1862f38f83568d96e814e469ab61f88336cc595"
ONETBB_REPOSITORY="https://github.com/uxlfoundation/oneTBB.git"

prefix=${1:?usage: build_onetbb.sh <install-prefix>}
work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT

git clone --quiet --depth 1 --branch "${ONETBB_TAG}" "${ONETBB_REPOSITORY}" "${work}/src"
actual=$(git -C "${work}/src" rev-parse HEAD)
if [ "${actual}" != "${ONETBB_COMMIT}" ]; then
  echo "oneTBB ${ONETBB_TAG} resolved to ${actual}, expected ${ONETBB_COMMIT}" >&2
  exit 1
fi

cmake_args=(
  -S "${work}/src"
  -B "${work}/build"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="${prefix}"
  -DCMAKE_INSTALL_LIBDIR=lib
  -DTBB_TEST=OFF
  -DTBB_EXAMPLES=OFF
  -DTBB_STRICT=OFF
  -DTBBMALLOC_BUILD=OFF
  -DTBB_DISABLE_HWLOC_AUTOMATIC_SEARCH=ON
)
if [ "$(uname -s)" = "Darwin" ]; then
  # Match the wheel's minimum macOS so delocate accepts the bundled library.
  cmake_args+=(-DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}")
fi
cmake "${cmake_args[@]}"
cmake --build "${work}/build" --parallel
cmake --install "${work}/build"

mkdir -p "${prefix}/share/wolfgang-notices"
cp "${work}/src/LICENSE.txt" "${prefix}/share/wolfgang-notices/oneTBB-LICENSE.txt"
cp "${work}/src/third-party-programs.txt" "${prefix}/share/wolfgang-notices/oneTBB-third-party-programs.txt"
echo "oneTBB ${ONETBB_TAG} (${ONETBB_COMMIT}) installed to ${prefix}"
