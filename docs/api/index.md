# API reference

Wolfgang's public Python API is intentionally compact. The authoritative behavioral overview is the [Python API guide](../guide/python-api.md); native docstrings are available through `help(wolfgang_quantum.PauliSum)` and related classes. The legacy `fastpauli` import remains as a one-transition compatibility shim.

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

## Native API

Headers under `include/wolfgang/` document the current source-level C++ surface. Legacy `include/fastpauli/` headers are a one-transition forwarding layer. Binary ABI stability is not promised before a deliberate native-library release. Consult the [API stability policy](../architecture/api_stability.md) before depending on pre-1.0 behavior.

## Private surfaces

Names beginning with `_`, internal extension modules, benchmark hooks, test helpers, campaign reports, and environment variables labeled benchmark-only are not public API. Their presence in a source checkout does not create a compatibility promise.
