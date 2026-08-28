# Legal provenance and clean-room policy

This repository is an independently authored MIT-licensed implementation.

## Rules

1. Public specifications, API documentation, benchmark methodology, architectural ideas, and observable behavior may be studied.
2. Code from other projects is not copied into this repository unless a maintainer intentionally imports a permissively licensed fragment and records its exact source, license, copyright notice, and modification history below.
3. GPL, AGPL, SSPL, BUSL, source-available, non-commercial, and otherwise incompatible implementation code must not be copied, translated line-for-line, or used as a code donor.
4. Public C APIs and Vulkan/HIP function signatures may be represented as required for interoperability. Generated or handwritten bindings remain subject to this repository's own implementation and review process.
5. New dependencies must pass `cargo deny check licenses` and the repository license policy before merge.

## References studied for architecture and interoperability

These references inform design only. No source code has been copied into the initial implementation.

| Project / specification | Why it is studied | Upstream license / status |
| --- | --- | --- |
| AMD HIP documentation | HIP runtime model, external-memory and external-semaphore API behavior | Documentation/reference material |
| ROCm `rocm-examples` Vulkan interop sample | Confirms the supported Vulkan/HIP external-resource architecture | MIT |
| ROCm HIP repository | HIP API surface and runtime behavior | MIT |
| Khronos Vulkan specification | Vulkan external-memory/semaphore semantics | Specification |
| `ash` | Idiomatic low-level Vulkan binding architecture for Rust | MIT OR Apache-2.0 |
| `abi_stable` | Reference for versioned plugin-boundary concepts; not required by the initial ABI | MIT OR Apache-2.0 |
| ZLUDA | High-level compatibility-layer architecture ideas only | MIT OR Apache-2.0 at time reviewed; no code copied |

## Imported code ledger

None.

If code is ever imported, add an entry with:

- exact upstream URL and commit;
- original file/path;
- license and required notices;
- copied line/range or generated artifact description;
- local destination;
- modifications made.

Absence of an entry means the code must be independently authored.

## Dependency licensing

The initial Rust dependency policy allows permissive licenses normally compatible with MIT distribution, including MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Unicode-3.0, and Zlib. Dependency licenses are audited in CI with `cargo-deny`.

This file is an engineering provenance record, not legal advice.
