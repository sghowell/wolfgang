# API reference

Wolfgang's public Python API is small by design. The [Python API guide](../guide/python-api.md) is the authoritative description of its behavior. Native docstrings are available through `help(wolfgang_quantum.PauliSum)` and the other exported classes.

```python
import wolfgang_quantum
help(wolfgang_quantum.PauliSum)
```

## Public package exports

- `wolfgang_quantum.PauliSum`
- `wolfgang_quantum.DevicePauliSum`
- `wolfgang_quantum.DeviceCommutationMatrix`
- `wolfgang_quantum.DeviceArrayView` (returned by `DevicePauliSum.device_arrays()`)
- `wolfgang_quantum.capabilities`
- `wolfgang_quantum.WolfgangCapabilities`
- `wolfgang_quantum.CpuCapabilities`
- `wolfgang_quantum.BackendCapabilities`
- `wolfgang_quantum.cuda_available` / `wolfgang_quantum.cuda_devices`
- `wolfgang_quantum.hip_available` / `wolfgang_quantum.hip_devices`
- `wolfgang_quantum.metal_available` / `wolfgang_quantum.metal_devices`
- `wolfgang_quantum.set_num_threads` / `wolfgang_quantum.get_num_threads`
- `wolfgang_quantum.measurement` (shot allocation, diagonalizing circuits and estimates from counts for commuting groups)
- `wolfgang_quantum.__version__`

Optional adapter methods are defined on the base class and check for their
dependency only when called.

## Operation capabilities

Each accelerator capability record lists, in its immutable `operations` pairs,
where the public path for each operation executes. For example,
`capabilities().accelerator("metal").execution_for("simplify")` returns
`"host_bridge"`. The `operations` pairs describe the public path, not runtime
availability; check `runtime_available` separately before executing an operation.

| Operation | CUDA / HIP | Metal |
| --- | --- | --- |
| `commutes_with`, `commutes_with_device` | device | device |
| `simplify` | device | host_bridge |
| `count_commuting`, `conflict_degrees` | device | host_shared_memory |
| `expectation_statevector`, `matmul` | device | unsupported |

`device` means the principal kernel executes on the device. It does not mean the
operation does no host work or synchronization: CUDA/HIP total counts currently
finish a partial-sum reduction on the host.

On Metal, count consumers scan shared memory on the CPU. Metal simplify
downloads the operator, simplifies it on the CPU, and uploads the result. For
one-word operators of at least 12,288 terms it instead sorts keys on the GPU and
reduces in float64 on the CPU over shared memory; the results are identical.

Private benchmark selectors do not change these public capability records.
Unknown operation names raise `ValueError`.

## Native API

Headers under `include/wolfgang/` document the current source-level C++ surface. Binary ABI stability is not promised before a deliberate native-library release. Consult the [API stability policy](../architecture/api_stability.md) before depending on pre-1.0 behavior.

## Private surfaces

Names beginning with `_`, internal extension modules, benchmark hooks, test helpers, campaign reports, and environment variables labeled benchmark-only are not public API. Their presence in a source checkout does not create a compatibility promise.
