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
      vrb-operators
            |
         vrb-core
        /        \
vrb-backends   vrb-plugin-api
```

Dependencies point downward. `vrb-core` must never depend on framework adapters, model integrations, or concrete operator implementations.

## Extension policy

New capabilities should normally be introduced as one of:

1. an injectable Rust service trait inside a higher-level crate;
2. a versioned dynamic plugin contract when ABI stability is required;
3. an adapter crate for an external framework or inference engine;
4. an operator implementation crate for GEMM, attention, quantization, transforms, or model-specific kernels.

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

## Release policy

Certified release tags remain immutable. New work is developed on branches and merged only after CI and applicable hardware certification pass.
