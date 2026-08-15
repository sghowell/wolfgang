# Apple Metal Bring-Up Report - 2026-05-01

## Summary

FastPauli now has a target-specific Apple Metal source-build lane:

```text
FASTPAULI_ENABLE_METAL=ON
accelerator_build_mode: metal_only
compiled_backends: cpu, metal
backend identity: metal
implemented operations: transfers, pairwise commutation, retained commutation matrix, compact count consumers
```

The local source build succeeded on an Apple M4 Pro Mac mini. Non-elevated
Codex commands still run under a sandbox profile that hides Metal devices from
`MTLCreateSystemDefaultDevice()`, but the same commands run with elevated Codex
execution see the Apple M4 Pro device and pass Metal runtime validation. Full
Xcode and the downloadable Metal Toolchain component are now installed, and a
short Metal System Trace capture succeeded for the FastPauli Metal benchmark.
This means the earlier runtime blocker was process sandboxing rather than GPU
utilization or missing hardware, and the earlier profiler blocker was missing
Xcode tooling rather than a FastPauli limitation.

## Evidence

Host:

```text
Model: Mac mini Mac16,11
SoC: Apple M4 Pro
CPU cores: 12
GPU: Apple M4 Pro, 16 cores
Metal support from system_profiler: Supported
macOS: Version 26.2 (Build 25C56)
Xcode: 26.4.1 (Build 17E202)
Objective-C++ compiler: AppleClang 21.0.0.21000099
SDK version from DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun --sdk macosx --show-sdk-version: 26.4
Metal Toolchain component: installed
Metal Toolchain identifier: com.apple.dt.toolchain.Metal.32023.883
Metal compiler version: Apple metal version 32023.883 (metalfe-32023.883)
Git revision at benchmark evidence capture: 8c6741d
Metal device name from FastPauli runtime under elevated Codex execution: Apple M4 Pro
Metal capability summary from FastPauli runtime under elevated Codex execution: unified_memory=true; low_power=false; headless=false; removable=false; recommended_max_working_set_size_bytes=40200896512
Metal storage mode: MTLResourceStorageModeShared
Metal commutation threadgroup size: 256
```

Commands run:

```bash
uv pip install -e '.[test]' \
  --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_METAL=ON

.venv/bin/python -m pytest tests/test_apple_metal_foundation.py -q
.venv/bin/python -m pytest tests/test_apple_metal_foundation.py tests/test_apple_accelerator_planning.py tests/test_backend_neutral_campaign9_plan.py -q
.venv/bin/python benchmarks/bench_metal_kernels.py --smoke --repeat 1 --json
FASTPAULI_VALIDATE_METAL=1 TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 PATH=.venv/bin:$PATH .venv/bin/python benchmarks/bench_metal_kernels.py --smoke --repeat 1 --json --output docs/benchmarks/data/apple_metal_bringup_2026-05-01/raw/metal_benchmark_smoke.json
swift -e 'import Metal; print(MTLCreateSystemDefaultDevice() as Any)'
swift -e 'import Metal; print(MTLCopyAllDevices())'
FASTPAULI_VALIDATE_METAL=1 .venv/bin/python scripts/validate.py
system_profiler SPDisplaysDataType
xcodebuild -version
xcodebuild -showComponent MetalToolchain
xcrun --sdk macosx --show-sdk-path
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun --sdk macosx --show-sdk-version
xcrun --find metal
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 xcrun metal -v
xcrun xctrace list devices
xcrun xctrace record --template 'Metal System Trace' --time-limit 8s --output /tmp/fastpauli-metal-allprocess-20260501-8c6741d.trace --all-processes --no-prompt
xcrun xctrace export --input /tmp/fastpauli-metal-allprocess-20260501-8c6741d.trace --toc --output /tmp/fastpauli-metal-allprocess-20260501-8c6741d-toc.xml
```

Results:

```text
Metal source build: passed
Apple Metal foundation pytest under metal_only build: 11 passed, 1 skipped
Metal-enabled semantic pytest under metal_only build: 275 passed, 90 skipped
Benchmark smoke: CPU and Metal rows emitted with metal_status.runtime_available == true
Sandboxed MTLCreateSystemDefaultDevice(): nil
Sandboxed MTLCopyAllDevices(): []
Elevated MTLCreateSystemDefaultDevice(): Optional(<AGXG16SDevice ... name = Apple M4 Pro>)
Elevated MTLCopyAllDevices(): [<AGXG16SDevice ... name = Apple M4 Pro>]
system_profiler SPDisplaysDataType: Apple M4 Pro GPU, Metal: Supported
plain xcrun --find metal: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal
validation-selected xcrun --find metal: /var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.5.188.0.lAJDnj/Metal.xctoolchain/usr/bin/metal
plain xcrun metal -v: unavailable without selecting the installed Metal Toolchain component
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 xcrun metal -v: Apple metal version 32023.883
xcrun xctrace list devices: local Mac mini
Metal System Trace: captured all-process trace with FastPauli python3.12 process present
Sanitized Metal System Trace summary: 50 Metal/GPU/graphics/MPS schemas, including command-buffer submission/completion and GPU interval schemas; device UUID, host display name, full process inventory, and local absolute paths omitted
```

Measured CPU rows from:

```bash
FASTPAULI_VALIDATE_METAL=1 TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 PATH=.venv/bin:$PATH .venv/bin/python benchmarks/bench_metal_kernels.py --smoke --repeat 1 --json --output docs/benchmarks/data/apple_metal_bringup_2026-05-01/raw/metal_benchmark_smoke.json
```

| Variant | Object backend | Boundary | Active CPU backend | Status | Median seconds |
| --- | --- | --- | --- | --- | ---: |
| cpu_default | cpu | host_materialized | scalar | ok | 1.3959011994302273e-05 |
| cpu_scalar | cpu | host_materialized | scalar | ok | 3.245798870921135e-05 |
| cpu_neon | cpu | host_materialized | neon | ok | 1.3249984476715326e-05 |

Measured Metal rows from elevated validation:

| Variant | Object backend | Boundary | Active CPU backend | Status | Median seconds |
| --- | --- | --- | --- | --- | ---: |
| metal_transfer_inclusive | metal | transfer_inclusive | n/a | ok | 0.0043651669984683394 |
| metal_device_resident | metal | device_resident | n/a | ok | 0.00012520901509560645 |
| metal_device_matrix | metal | device_resident | n/a | ok | 0.0008132910006679595 |
| metal_compact_consumer | metal | compact_consumer | n/a | ok | 2.4579931050539017e-06 |

Checked data:

```text
Benchmark JSON: docs/benchmarks/data/apple_metal_bringup_2026-05-01/raw/metal_benchmark_smoke.json
Summary JSON: docs/benchmarks/data/apple_metal_bringup_2026-05-01/summary.json
Sanitized Metal System Trace summary: docs/benchmarks/data/apple_metal_bringup_2026-05-01/profiler/metal_system_trace_summary.json
Local binary trace bundle and raw all-process TOC: /tmp/fastpauli-metal-allprocess-20260501-8c6741d.trace and /tmp/fastpauli-metal-allprocess-20260501-8c6741d-toc.xml (intentionally not checked in)
```

## Implemented Surface

Build and dispatch:

```text
FASTPAULI_ENABLE_METAL is OFF by default
FASTPAULI_ENABLE_METAL rejects CUDA or HIP target flags at configure time
CPU-only builds include no Metal framework dependency
Metal source builds use private Objective-C++ translation units under src/metal
public headers remain Apple-framework-free
```

Runtime API:

```text
PauliSum.to_device(backend="metal")
DevicePauliSum.backend == "metal"
DevicePauliSum.to_host()
DeviceCommutationMatrix.empty(..., backend="metal")
DeviceCommutationMatrix.backend == "metal"
DevicePauliSum.commutes_with()
DevicePauliSum.commutes_with_into()
DevicePauliSum.commutes_with_device()
DevicePauliSum.commutes_with_device(..., output=...)
DeviceCommutationMatrix.count_commuting(axis=None|0|1)
DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)
```

Out-of-scope surfaces remain unavailable:

```text
Metal wheels
MPSGraph-first sparse kernels
raw Metal buffer export
DLPack export
CUDA Array Interface export
public async or command queue APIs
mixed CUDA/HIP/Metal source builds
```

## Runtime And Profiler Notes

The host hardware reports a Metal-capable Apple M4 Pro GPU, and elevated Codex
execution sees the same device through the Metal runtime. The non-elevated
Codex command sandbox still reports `nil` and an empty device list. That makes
Metal validation in this environment an elevated-command requirement, not a
hardware-availability or high-utilization issue.

Full Xcode and `xctrace` are now available. `xcodebuild -showComponent
MetalToolchain` reports the Metal Toolchain as installed, but this Xcode
installation still requires an explicit toolchain selector for the standalone
compiler: `TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 xcrun metal -v`
works, while plain `xcrun metal -v` reports that the Metal Toolchain is
missing. The repo-local validation entrypoint now discovers the installed
Metal Toolchain identifier and applies that selector to its Metal build,
runtime, benchmark, SDK, and compiler-version checks without changing the
machine-wide `xcode-select` setting.

The retained profiler evidence is a short all-process Metal System Trace. A
targeted `xctrace --launch` probe was rejected for this environment because
`xctrace` resolves the virtualenv interpreter through a Python app wrapper and
loses the virtualenv packages before importing NumPy. The all-process capture
avoids that launch-environment issue and records the FastPauli `.venv`
`python3.12` process in a sanitized trace summary. The binary `.trace` bundle
and raw all-process TOC remain local because they are large and include
unrelated process inventory; the checked summary records only the template,
duration, redacted FastPauli process presence, privacy omissions, and available
Metal/GPU schemas. The default template has GPU counter profile and shader
timeline disabled, so deeper Apple GPU counter work remains a separate future
profiling task rather than a bring-up blocker.
