# CUDA Async And Stream API Review

Status: public async/stream API deferred; benchmark-only private probes rejected for Campaign 6.

## Existing Invariant: Public CUDA Methods Synchronize Before Returning

FastPauli's public CUDA methods use the default stream and synchronize before
returning to Python. Callers may immediately inspect returned host values, reuse
input objects, or observe Python exceptions without reasoning about outstanding
device work.

Campaign 6 preserves that invariant for `DevicePauliSum.simplify()`,
`DevicePauliSum.matmul()`, `DevicePauliSum.expectation_statevector()`,
`DevicePauliSum.commutes_with()`, `DevicePauliSum.commutes_with_into()`, and
`DevicePauliSum.commutes_with_device()`.

## Candidate Stream Handle Forms

The candidate public forms considered were:

```text
Python integer stream pointer argument
Python object exposing a CuPy-compatible stream pointer
FastPauli-owned Stream class
FastPauli-owned AsyncResult/Event class
```

Raw integer stream pointers were rejected for public API because they cannot
encode ownership, device ordinal, capture state, or lifetime. A CuPy stream
object contract is attractive for interop, but making it public would couple
FastPauli's synchronization semantics to optional package behavior. A
FastPauli-owned stream wrapper is the most controllable future option, but it
requires broader lifetime and error-propagation design than this campaign needs.

## Candidate Event Ownership Model

A future async API should treat completion events as owned by FastPauli result
objects, not by temporary Python call frames. The event owner must keep all
input and output device allocations alive until the event has completed or an
explicit synchronization/error query has transferred ownership back to the
caller.

Campaign 6 does not add such an owner. Without a retained owner type, private
enqueue-only timing would not represent a safe public contract and would be
easy to confuse with complete operation latency.

## Stream Capture Behavior

Public stream-aware methods must state whether they are graph-capture safe.
Capture safety requires no host synchronization, no hidden allocation in the
captured region, deterministic stream association, and no Python exception path
that depends on deferred CUDA errors.

Current public CUDA methods intentionally call synchronization APIs and may
allocate internal temporary storage. They are therefore not documented as stream
capture safe. Campaign 6 defers graph-capture support.

## Host Synchronization Semantics

The retained Campaign 6 rule is:

```text
all public CUDA methods synchronize before returning
benchmark timings that are not synchronized end-to-end must remain private and labeled as incomplete enqueue or event boundaries
no README or user-facing performance claim may compare enqueue-only timing against synchronous public CPU or CUDA timing
```

Because Campaign 6's primary consumer work is compact summary reduction rather
than launch-overhead reduction, it does not add private stream/event probes.

## Python Object Lifetime Rules

Any future async result object must keep these objects alive until completion:

```text
DevicePauliSum operands
DeviceCommutationMatrix outputs
CUDA-array-interface input arrays accepted by FastPauli
workspace or temporary-storage owners
completion events
the CUDA stream object when FastPauli owns or references it
```

The current public synchronous API does not need extra Python lifetime handles
because all device work has completed before returning.

## Exception And Error Propagation

Current CUDA errors are checked before the public method returns. A future async
API would need a defined point where deferred launch, copy, and synchronization
errors become Python exceptions. Candidate points are:

```text
AsyncResult.wait()
AsyncResult.result()
explicit stream synchronization helper
object destruction as a best-effort log-only fallback, not as the primary error surface
```

Campaign 6 does not introduce deferred error propagation.

## Benchmark-Only Private Prototype Rules

Private stream/event probes are rejected for this campaign. A later campaign may
add them only when the benchmark explicitly records:

```text
private_benchmark_only boundary label
stream ownership and device ordinal
whether CUDA graph capture is attempted or rejected
whether the timing is enqueue-only, event-elapsed, synchronize-only, or end-to-end
which public synchronous timing remains the apples-to-apples comparison
```

## Retained Public Surfaces

None.

## Rejected Public Surfaces

```text
public raw stream pointer arguments
public raw event pointer arguments
public enqueue-only methods
public async return objects without an owner/lifetime contract
public stream-capture support claims
```

## Deferred Public Surfaces

```text
FastPauli-owned Stream wrapper
FastPauli-owned AsyncResult/Event wrapper
CuPy stream interop
CUDA graph capture support
stream-ordered workspace ownership
```
