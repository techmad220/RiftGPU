# Merge Security Contract

RiftGPU `main` is protected by evidence-bearing merge gates. A pull request is eligible to merge only when the configured required checks are successful for its latest exact head revision.

The software proof verifies formatting, debug and release regression suites, Clippy, release builds, dependency/license policy, platform certification, dynamic plugin integration tests, and full reachable-history secret scanning.

The AMD hardware proof is produced on the trusted TECH host for same-repository pull-request heads only. It checks out and verifies the immutable PR head SHA, runs release regressions, executes the Vulkan-to-HIP hardware stress path including the 32-cycle and 64 MiB transfer proof, executes shared rocBLAS GEMM against the CPU correctness oracle, and re-reads the PR after execution so a moved head cannot inherit an earlier hardware result.

Fork-originated code is never executed by the trusted self-hosted hardware gate.

These gates establish evidence for the invariants they test; they do not constitute a mathematical guarantee that software can contain no undiscovered defect.
