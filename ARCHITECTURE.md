# Architecture

## Non-negotiable boundary

`vrb-core` is infrastructure. It owns backend discovery, capability/routing policy, performance records, and dependency-injection seams.

Higher-level compute features and resource-specific invocation contracts are built on top and must not be hard-wired into the core runtime.

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

shared-resource operator services
            |
   vrb-shared-operators <------ vrb-shared-operator-loader
            |                              |
      vrb-operators             vrb-shared-operator-plugin-api

versioned operator protocols (GEMM, attention, ...)
            |
reference / HIP / Vulkan operator implementations
```

Dynamic operator libraries depend only on stable operator-facing contracts rather than on `vrb-core` internals. Loaders adapt those libraries into injected operator services. `vrb-core` does not know that operator plugins or shared-resource operator plugins exist.

Dependencies point downward. `vrb-core` must never depend on framework adapters, model integrations, operator loaders, operator plugin APIs, shared-resource operator APIs, operator protocols, or concrete operator implementations.

## Extension policy

New capabilities should normally be introduced as one of:

1. an injectable Rust service trait inside a higher-level crate;
2. a versioned dynamic plugin contract when ABI stability is required;
3. an adapter crate for an external framework or inference engine;
4. an operator implementation crate for GEMM, attention, quantization, transforms, or model-specific kernels.

## Operator protocol boundary

Operator semantics are versioned separately from generic dynamic-plugin ABIs. A plugin ABI answers **how an operator is discovered and invoked**; an operator protocol answers **what the control metadata means**.

For example, the GEMM protocol is an independent crate defining portable semantics for `C = alpha * A * B + beta * C`. A CPU reference implementation provides the correctness oracle. Future HIP/Vulkan implementations must match those semantics rather than inventing backend-specific behavior.

Protocol decoders must validate magic, version, header size, flags, dimensions, arithmetic overflow, and exact encoded length before allocating based on message contents. Concrete implementations may impose stricter injectable resource/work limits.

## Host-byte operator plugin boundary

Dynamic host-byte compute operators use a dedicated ABI rather than extending the v0.1 backend-plugin ABI. This keeps transport/backend evolution independent from compute-kernel evolution.

The host-byte operator plugin host must:

- validate ABI version and descriptor size before accepting callbacks;
- query operator metadata into host-owned structures;
- use raw integer tags at the C boundary so unknown future values degrade safely;
- reject duplicate operator IDs and malformed descriptors;
- serialize callbacks that share plugin state;
- bound plugin-directed host allocations through an injectable load policy;
- keep the dynamic library loaded until the optional shutdown callback completes;
- advertise only capabilities the host adapter actually preserves.

The host-byte adapter never advertises zero-copy because its bulk payload crosses the ABI as host bytes.

## Shared-resource operator boundary

Shared-resource invocation is a separate Rust contract and a separately versioned C ABI. It passes small host-side control metadata plus borrowed external-memory regions and synchronization points. Bulk tensor payloads remain in externally shared allocations.

The shared-resource host must:

- validate every region before dispatch, including non-zero handle, non-zero length, checked range arithmetic, and allocation bounds;
- treat native handles as borrowed for callback duration only; plugins may not close or retain them;
- keep memory-handle kind, access mode, allocation size, offset, and length explicit;
- keep wait and signal synchronization points explicit and independently validated;
- bound metadata bytes, resource count, synchronization count, operator count, and host receipt size;
- use only host-owned ABI descriptor arrays during callbacks;
- reject contradictory capability claims;
- keep `EXTERNAL_RESOURCE` support distinct from `PROVEN_ZERO_COPY`.

`PROVEN_ZERO_COPY` is an evidence-bearing capability. A plugin may set it only when its actual execution path directly consumes the shared allocation without a host-relay copy for bulk tensor data. Merely receiving an external handle is insufficient. Generic software E2E tests validate the contract and loader; hardware certification is required before a real HIP/Vulkan operator may advertise `PROVEN_ZERO_COPY`.

## Core admission test

A feature belongs in `vrb-core` only if all of these are true:

- it is backend-agnostic infrastructure;
- it is needed by multiple independent higher-level consumers;
- it cannot be expressed cleanly through an injected service or plugin contract;
- including it does not create a dependency from core onto a framework, model, resource-specific invocation ABI, or concrete compute kernel.

If any condition fails, build it above the core.

## Compatibility policy

- Rust-internal DI may use traits and `Arc<dyn Trait + Send + Sync>`.
- Dynamic libraries cross only versioned C ABI boundaries.
- No Rust trait object crosses a DLL/shared-library ABI boundary.
- Plugin descriptors are size- and version-checked before callbacks are accepted.
- New optional capabilities must not break older plugins.
- Backend-plugin, host-byte operator-plugin, and shared-resource operator-plugin ABIs version independently.
- Individual operator protocols version independently from all plugin ABIs.

## Release policy

Certified release tags remain immutable. New work is developed on branches and merged only after CI and applicable hardware certification pass.
