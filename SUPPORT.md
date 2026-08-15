# Wolfgang Support

## Start here

1. Read the [installation guide](docs/getting-started/installation.md).
2. Check the [release support matrix](docs/release/support_matrix.md).
3. Run `python -c "import fastpauli; print(fastpauli.__version__)"`.
4. For source builds, retain the complete CMake configure output and `_build_info()` report.

## Asking for help

Use a GitHub Discussion or a regular issue for installation questions, API clarification, reproducible correctness defects, and documentation gaps. Use the hardware-support issue form for CUDA, ROCm/HIP, Metal, compiler, or wheel requests. Use the performance-regression form only when the same operation, data, timing boundary, and environment can be compared.

A useful report includes:

- Wolfgang and Python versions;
- OS, architecture, and compiler;
- install/build command;
- relevant CMake options;
- accelerator model, runtime, driver, and target architecture where applicable;
- minimal code and complete error text;
- whether the scalar CPU path reproduces the problem.

Sanitize usernames, addresses, tokens, hostnames, absolute paths, and profiler databases before posting.

## Support boundaries

CPU wheels and source-build accelerator paths have different evidence levels. A backend being implemented does not imply that wheels exist for it. Windows, combined accelerator binaries, arbitrary GPU architectures, and asynchronous stream semantics are not supported unless the current support matrix explicitly says otherwise.

Security concerns belong in the private process described by [SECURITY.md](SECURITY.md), not in public support channels.
