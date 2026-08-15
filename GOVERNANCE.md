# Governance

Wolfgang is currently a maintainer-led open-source research engineering project.

## Maintainer

Sean Howell is the founding maintainer and release owner. The maintainer sets the public API and support boundary, reviews changes, manages releases, and adjudicates benchmark and correctness claims.

## Decision principles

Decisions prioritize, in order:

1. numerical and memory correctness;
2. honest evidence and reproducibility;
3. a portable scalar CPU baseline;
4. clear public contracts and maintainability;
5. measured accelerator performance;
6. feature breadth.

Performance changes require a correctness oracle, named timing boundary, retained environment description, and repeatable evidence. Hardware-specific optimization may not weaken portable imports or silently alter semantics.

## Contributions and review

All substantive changes use pull requests. A change must pass required CI and receive maintainer approval. Public API, packed representation, interop ownership, synchronization, release policy, and benchmark-schema changes require explicit design review. Automated or agent-authored changes are held to the same evidence and review standard as human-authored changes and must identify generated artifacts honestly.

## Releases

The maintainer owns versioning, signed/tagged release decisions, package publication, support-matrix wording, and security patches. Release artifacts must be built from the tagged revision through reviewed automation.

## Evolution

If recurring contributors emerge, maintainership may be extended based on sustained high-quality review and implementation. Governance changes are proposed by pull request. If the founding maintainer becomes inactive, established maintainers may appoint a release owner by documented consensus.
