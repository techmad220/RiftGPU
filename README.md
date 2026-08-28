# Vulkan ROCm Bridge

A modular, MIT-licensed Rust GPU runtime for coordinating Vulkan and AMD ROCm/HIP on the same machine.

The project is intentionally **not** a Vulkan-to-HIP emulator. HIP work is submitted to the native HIP runtime; Vulkan work is submitted to the native Vulkan driver. The bridge owns discovery, resource interoperability, synchronization, capability routing, plugin loading, and benchmark-driven backend selection.

## Why

A single workload may have operations that are best served by different GPU stacks. The goal is to keep data resident on the GPU and let the runtime route work by measured capability instead of forcing the whole application into one backend.

```text
Application / inference engine
            |
      vrb-core runtime
            |
    capability scheduler
       /           \
  Vulkan           HIP/ROCm
       \           /
      shared GPU resources
             |
           AMD GPU
```

## Architecture rule

`vrb-core` remains a small infrastructure layer. New compute features are built **on top** through dependency-injected services, versioned plugin contracts, operator crates, and integrations. Model-specific kernels, inference-engine bindings, schedulers, telemetry, cache policy, and framework adapters do not belong in the transport core.

That rule is deliberate: keep the bridge replaceable, testable, and stable while higher-level capabilities evolve independently. See [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Current v0.1 architecture

- `vrb-plugin-api` — versioned C ABI for dynamic backend/operator plugins. No Rust trait objects cross a DLL boundary.
- `vrb-core` — dependency-injected backend registry, capability model, performance records, and routing policy.
- `vrb-backends` — CPU reference implementation plus Vulkan, HIP, and dynamic-plugin backends.
- `vrb` — diagnostics, routing, and benchmark CLI.

The built-in Vulkan and HIP backends dynamically discover the local runtime and required external-resource APIs. They deliberately advertise only transport/runtime capabilities. Optimized GEMM, attention, quantization, and model-specific kernels belong in operator plugins and can evolve independently of the core.

## Commands

Probe the machine:

```console
cargo run -p vrb -- doctor
```

Machine-readable probe:

```console
cargo run -p vrb -- doctor --json
```

Ask the scheduler to select a compatible backend:

```console
cargo run -p vrb -- route custom unknown --zero-copy
```

Run the portable correctness benchmark:

```console
cargo run -p vrb -- bench-cpu --elements 1048576 --iterations 25
```

Load an external backend plugin:

```console
cargo run -p vrb -- doctor --plugin path/to/backend.dll
```

## Native-speed model

ROCm is not interpreted by Vulkan. A HIP plugin executes through the real HIP runtime, so its kernels retain native HIP execution characteristics. The bridge's cost is at resource import, synchronization, scheduling, and command submission boundaries. Keeping allocations imported and resident amortizes those costs.

The design therefore prefers:

```text
shared VRAM -> HIP work -> GPU sync -> Vulkan work -> GPU sync -> HIP work
```

over host round trips:

```text
VRAM -> system RAM -> VRAM -> system RAM -> VRAM
```

## Plugins and dependency injection

Inside the process, dependencies are injected through Rust traits and `Arc<dyn ...>` interfaces. Dynamic plugins use the smaller versioned `extern "C"` ABI in `vrb-plugin-api` so compiler-specific Rust layout is never part of the compatibility contract.

A plugin can advertise backend type and capabilities, probe its devices, and execute requests. The host validates ABI version and descriptor size before accepting callbacks and serializes calls through the plugin state boundary.

## Routing

The default policy is `FastestCompatible`:

1. reject unavailable or capability-incompatible backends;
2. prefer measured median latency when benchmark data exists;
3. otherwise use a bootstrap ordering of HIP, Vulkan, Hybrid, Plugin, then CPU;
4. allow callers to inject a completely different routing policy.

The bootstrap ordering is not treated as a performance fact. Hardware measurements override it.

## Legal / provenance policy

This is an independently authored MIT implementation. We study public specifications, public API documentation, observable behavior, benchmark methodology, and architectural ideas. We do not copy incompatible implementation code.

See [`LEGAL_PROVENANCE.md`](LEGAL_PROVENANCE.md) for the clean-room rules and source ledger. CI runs `cargo-deny` to reject dependencies and sources outside the allowlisted policy.

## Build

Minimum supported Rust version: **1.88**.

```console
cargo build --workspace --release
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

ROCm is loaded dynamically at runtime. Building the portable workspace does not require a ROCm SDK installation.

## Interop acceptance criteria

A hardware bridge is considered working only after the following passes on a real machine:

1. Vulkan and HIP both discover a usable AMD GPU.
2. Vulkan confirms the platform external-memory capability.
3. HIP exports the required external-memory entry points.
4. Vulkan creates an exportable device allocation.
5. HIP imports and maps that exact allocation.
6. HIP modifies the shared allocation without a CPU relay copy.
7. Vulkan observes the result and correctness validation passes.
8. synchronization is explicit and repeatable.
9. repeated runs do not leak handles or GPU resources.
10. benchmark receipts compare shared-resource transitions against copy-based fallback.

No benchmark result is hard-coded into routing as a claim about hardware it has not measured.

## License

MIT. See [`LICENSE`](LICENSE).
