# vrb-operators

Higher-level compute operator contracts for Vulkan ROCm Bridge.

This crate sits above `vrb-core`. It provides dependency-injected operator discovery/selection and deliberately does not own Vulkan/HIP transport, framework bindings, or model-specific kernels.

Concrete implementations should live in separate crates or dynamic plugins.

Planned implementation families:

- GEMM
- attention
- quantize/dequantize
- transforms
- model-specific/custom operators

The registry is policy-driven: callers may inject a different selection policy without changing operator implementations or the core bridge.
