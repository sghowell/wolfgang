# Security Policy

Wolfgang combines native C++20 code, Python buffer protocols, and optional accelerator runtimes. Memory-safety, device-pointer ownership, malformed packed data, build provenance, and package integrity are security-relevant even though Wolfgang is a local numerical library.

## Supported versions

Security fixes are provided for the latest released minor series. Pre-release and source-build accelerator features are supported on a best-effort basis unless the release support matrix says otherwise. See [`docs/release/support_matrix.md`](docs/release/support_matrix.md) for the exact current boundary.

## Privately report a vulnerability

**Do not open a public issue for a suspected vulnerability.** Privately report it through GitHub's private vulnerability reporting for this repository. If that channel is unavailable, email `sean.g.howell@gmail.com` with the subject `Wolfgang security report`.

Include, when safe:

- affected version or revision;
- platform, compiler, Python, and accelerator runtime;
- a minimal reproducer;
- expected and observed behavior;
- impact and whether untrusted input is required;
- sanitizer, debugger, or profiler output with credentials and personal paths removed.

Do not send live credentials, proprietary datasets, raw environment dumps, or unredacted cloud-profiler databases.

## Response process

The maintainer will acknowledge a report within seven days, reproduce it privately, assess affected versions, prepare a regression test and minimal fix, and coordinate disclosure. Confirmed vulnerabilities in a released version will receive a patch release and release-note advisory. Public disclosure waits until a fix is available unless continued secrecy creates greater risk.

## Security design and scope

Wolfgang treats public Python/C++ inputs and external array protocols as trust boundaries. Release gates include checked allocation arithmetic, property tests, native sanitizers where supported, dependency scanning, immutable release automation, and artifact privacy scanning. More detail is in [`docs/quality/security_and_supply_chain.md`](docs/quality/security_and_supply_chain.md).
