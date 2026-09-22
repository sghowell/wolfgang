# API reference

Wolfgang's public Python API is intentionally compact. The authoritative behavioral overview is the [Python API guide](../guide/python-api.md); native docstrings are available through `help(wolfgang_quantum.PauliSum)` and related classes.

```python
import wolfgang_quantum
help(wolfgang_quantum.PauliSum)
```

## Public package exports

- `wolfgang_quantum.PauliSum`
- `wolfgang_quantum.DevicePauliSum`
- `wolfgang_quantum.DeviceCommutationMatrix`
- `wolfgang_quantum.capabilities`
- `wolfgang_quantum.WolfgangCapabilities`
- `wolfgang_quantum.CpuCapabilities`
- `wolfgang_quantum.BackendCapabilities`
- `wolfgang_quantum.cuda_available` / `wolfgang_quantum.cuda_devices`
- `wolfgang_quantum.hip_available` / `wolfgang_quantum.hip_devices`
- `wolfgang_quantum.metal_available` / `wolfgang_quantum.metal_devices`
- `wolfgang_quantum.__version__`

Optional adapter methods are available through the base class with lazy dependency checks.

## Operation capabilities

`capabilities().accelerator("metal").execution_for("simplify")` returns
`"host_bridge"`. The immutable `operations` pairs describe the public path;
check `runtime_available` separately before executing an operation.

| Operation | CUDA / HIP | Metal |
| --- | --- | --- |
| `commutes_with`, `commutes_with_device` | device | device |
| `simplify` | device | host_bridge |
| `count_commuting`, `conflict_degrees` | device | host_shared_memory |
| `expectation_statevector`, `matmul` | device | unsupported |

`device` identifies where the principal kernel executes, not an absence of host
work or synchronization. CUDA/HIP total counts currently finish a partial-sum
reduction on the host. Metal count consumers scan shared memory on the CPU;
Metal simplify downloads, simplifies on the CPU, and uploads. Private benchmark
selectors do not change these public capability records. Unknown operation names
raise `ValueError`.

## Native API

Headers under `include/wolfgang/` document the current source-level C++ surface. Binary ABI stability is not promised before a deliberate native-library release. Consult the [API stability policy](../architecture/api_stability.md) before depending on pre-1.0 behavior.

## Private surfaces

Names beginning with `_`, internal extension modules, benchmark hooks, test helpers, campaign reports, and environment variables labeled benchmark-only are not public API. Their presence in a source checkout does not create a compatibility promise.
