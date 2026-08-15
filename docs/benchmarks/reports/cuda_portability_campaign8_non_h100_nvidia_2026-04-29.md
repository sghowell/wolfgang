# Campaign 8 Non-H100 NVIDIA Portability Report

Date: 2026-04-29

Status: blocked, with blocker recorded.

Campaign 8 required a second NVIDIA architecture before widening claims beyond the H100 SM90 host. No A100, RTX 6000 Ada, L4, or A10 host was available during this execution window, so no non-H100 source build, validation command, or portability benchmark command was run.

## Hardware Identifier

```text
blocked_no_non_h100_nvidia_host
```

The identifier was recorded before any portability command was attempted. Since there was no selected host, GPU name, compute capability, driver, toolkit, compiled architecture, validation command, benchmark command, and runtime result are all unavailable.

## Required Host Priority

```text
1. A100, SM80
2. RTX 6000 Ada, SM89
3. L4, SM89
4. A10, SM86
```

## Result

Campaign 8 claims remain H100-only for the retained private benchmark consumers:

```text
device_resident_graph_status: retained on H100
public_grouping_api_status: deferred
dlpack_interop_status: deferred
non_h100_portability_status: blocked
stream_graph_status: deferred
scatter_tuning_status: rejected_no_consumer
```

The next portability run must build from source on the selected host with the matching `FASTPAULI_CUDA_ARCHITECTURES` value, run `scripts/validate.py` with `FASTPAULI_VALIDATE_CUDA=1`, then run `benchmarks/bench_cuda_scaling.py --profile campaign8-portability --repeat 5 --warmup 1 --json`.
