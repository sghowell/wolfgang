# CUDA Campaign 7 Non-H100 NVIDIA Portability

Status: infrastructure blocker recorded.

Campaign 7 required one retained-consumer run on an NVIDIA architecture other
than H100 before broadening GPU claims. No A100, RTX 6000 Ada, L4, A10, or
other non-H100 NVIDIA host was available during this execution slice.

## Result

```text
hardware: unavailable
compute capability: unavailable
driver: unavailable
CUDA toolkit: unavailable
compiled architectures: unavailable
retained Campaign 7 consumer run: not run
README claim broadening: rejected; H100 source-build evidence only
```

The blocker is recorded in
`docs/benchmarks/data/cuda_portability_campaign7_non_h100_nvidia_2026-04-29/metadata/blocker.txt`.

## Next Action

Run `docs/plans/h100_deep_optimization_campaign7_plan.md` Task 7 on a non-H100
NVIDIA host, preferably A100 SM80, and replace this blocker with raw benchmark
JSON, metadata, validation logs, and an updated portability report.
