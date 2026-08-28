# Architecture

## Non-negotiable boundary

`vrb-core` is infrastructure. It owns backend discovery, capability/routing policy, performance records, shared-resource coordination contracts, and dependency-injection seams.

Higher-level compute features are built on top and must not be hard-wired into the core runtime.

## Dependency direction

```text
applications / framework adapters
            |
      operator services
            |
      vrb-operators <------ vrb-operator-loader
            |                       |
         vrb-core          vrb-operator-plugin-api
        /        \                    ^
vrb-backends   vrb-plugin-api          |
                                    dynamic
                                  operator DLLs

versioned operator protocols (GEMM, attention, ...)
            |
reference / HIP / Vulkan operator implementations
```

Dynamic operator libraries depend only on stable operator-facing contracts rather than on `vrb-core` internals. The loader adapts those libraries into `vrb-operators::Operator` instances and injects them into an operator registry. `vrb-core` does not know that operator plugins exist.

Dependencies point downward. `vrb-core` must never depend on framework adapters, model integrations, operator loaders, operator plugin APIs, operator protocols, or concrete operator implementations.

## Extension policy

New capabilities should normally be introduced as one of:

1. an injectable Rust service trait inside a higher-level crate;
2. a versioned dynamic plugin contract when ABI stability is required;
3. an adapter crate for an external framework or inference engine;
4. an operator implementation crate for GEMM, attention, quantization, transforms, or model-specific kernels.

## Operator protocol boundary

Operator semantics are versioned separately from the generic dynamic-plugin ABI. The plugin ABI answers **how an operator is discovered and invoked**; an operator protocol answers **what the invocation bytes mean**.

For example, the GEMM protocol is an independent crate defining a portable little-endian request/response format for `C = alpha * A * B + beta * C`. A CPU reference implementation provides the correctness oracle. Future HIP/Vulkan implementations must consume the same protocol and match the reference semantics rather than inventing backend-specific request formats.

Protocol decoders must validate magic, version, header size, flags, dimensions, arithmetic overflow, and exact encoded length before allocating based on message contents. Concrete implementations may impose stricter injectable resource/work limits.

## Operator plugin boundary

Dynamic compute operators use a dedicated ABI rather than extending the v0.1 backend-plugin ABI. This keeps transport/backend evolution independent from compute-kernel evolution.

The operator plugin host must:

- validate ABI version and descriptor size before accepting callbacks;
- query operator metadata into host-owned structures;
- use raw integer tags at the C boundary so unknown future values degrade safely;
- reject duplicate operator IDs and malformed descriptors;
- serialize callbacks that share plugin state;
- bound plugin-directed host allocations through an injectable load policy;
- keep the dynamic library loaded until the optional shutdown callback completes;
- advertise only capabilities the host adapter actually preserves.

The initial host-byte adapter therefore does not advertise zero-copy. A future shared-resource invocation path must prove actual shared-resource semantics before enabling that capability.

## Core admission test

A feature belongs in `vrb-core` only if all of these are true:

- it is backend-agnostic infrastructure;
- it is needed by multiple independent higher-level consumers;
- it cannot be expressed cleanly through an injected service or plugin contract;
- including it does not create a dependency from core onto a framework, model, or concrete compute kernel.

If any condition fails, build it above the core.

## Compatibility policy

- Rust-internal DI may use traits and `Arc<dyn Trait + Send + Sync>`.
- Dynamic libraries cross only versioned C ABI boundaries.
- No Rust trait object crosses a DLL/shared-library ABI boundary.
- Plugin descriptors are size- and version-checked before callbacks are accepted.
- New optional capabilities must not break older plugins.
- Backend-plugin and operator-plugin ABIs version independently.
- Individual operator protocols version independently from the generic operator-plugin ABI.

## Release policy

Certified release tags remain immutable. New work is developed on branches and merged only after CI and applicable hardware certification pass.
