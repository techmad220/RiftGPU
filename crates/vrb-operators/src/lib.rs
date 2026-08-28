//! Operator-layer abstractions built on top of `vrb-core`.
//!
//! This crate intentionally contains no transport implementation. It provides a
//! dependency-injected operator registry and selection contract so GEMM,
//! attention, quantization, transforms, and model-specific kernels can evolve
//! independently of the bridge core.

use std::{collections::BTreeMap, sync::Arc};

use thiserror::Error;
use vrb_core::BackendKind;

/// Stable logical classes of compute that may be provided by operator plugins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperatorKind {
    Gemm,
    Attention,
    Quantize,
    Dequantize,
    Transform,
    Custom,
}

/// Backend-neutral request metadata used for operator selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorRequest {
    pub kind: OperatorKind,
    pub preferred_backend: Option<BackendKind>,
    pub requires_zero_copy: bool,
}

impl OperatorRequest {
    #[must_use]
    pub const fn new(kind: OperatorKind) -> Self {
        Self {
            kind,
            preferred_backend: None,
            requires_zero_copy: false,
        }
    }
}

/// Declarative capabilities of one operator implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorCapabilities {
    pub kind: OperatorKind,
    pub backend: BackendKind,
    pub supports_zero_copy: bool,
}

/// Backend-neutral invocation payload.
///
/// The initial contract uses opaque bytes on purpose. Typed tensor views and
/// shared-resource handles can be layered on without coupling this crate to a
/// framework or model runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorInvocation<'a> {
    pub input: &'a [u8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorOutput {
    pub bytes: Vec<u8>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum OperatorError {
    #[error("operator is unavailable: {0}")]
    Unavailable(String),
    #[error("operator execution failed: {0}")]
    Execution(String),
    #[error("no compatible operator registered for {0:?}")]
    NoCompatibleOperator(OperatorKind),
}

/// Injectable compute operator.
pub trait Operator: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> OperatorCapabilities;
    fn execute(&self, invocation: OperatorInvocation<'_>) -> Result<OperatorOutput, OperatorError>;
}

/// Injectable selection policy. Applications can replace this without changing
/// the registry or any concrete operator.
pub trait OperatorSelectionPolicy: Send + Sync {
    fn select(
        &self,
        request: &OperatorRequest,
        candidates: &[Arc<dyn Operator>],
    ) -> Option<Arc<dyn Operator>>;
}

/// Conservative default policy: filter for compatibility, honor a requested
/// backend when available, and otherwise preserve registration order.
#[derive(Debug, Default)]
pub struct FirstCompatible;

impl OperatorSelectionPolicy for FirstCompatible {
    fn select(
        &self,
        request: &OperatorRequest,
        candidates: &[Arc<dyn Operator>],
    ) -> Option<Arc<dyn Operator>> {
        let compatible = |operator: &&Arc<dyn Operator>| {
            let capabilities = operator.capabilities();
            capabilities.kind == request.kind
                && (!request.requires_zero_copy || capabilities.supports_zero_copy)
        };

        if let Some(preferred_backend) = request.preferred_backend {
            if let Some(operator) = candidates
                .iter()
                .filter(compatible)
                .find(|operator| operator.capabilities().backend == preferred_backend)
            {
                return Some(Arc::clone(operator));
            }
        }

        candidates.iter().filter(compatible).next().cloned()
    }
}

/// DI-driven operator registry. It owns no concrete kernels and has no knowledge
/// of Vulkan, HIP APIs, inference engines, or model formats.
pub struct OperatorRegistry {
    operators: BTreeMap<OperatorKind, Vec<Arc<dyn Operator>>>,
    policy: Arc<dyn OperatorSelectionPolicy>,
}

impl OperatorRegistry {
    #[must_use]
    pub fn new(policy: Arc<dyn OperatorSelectionPolicy>) -> Self {
        Self {
            operators: BTreeMap::new(),
            policy,
        }
    }

    pub fn register(&mut self, operator: Arc<dyn Operator>) {
        self.operators
            .entry(operator.capabilities().kind)
            .or_default()
            .push(operator);
    }

    pub fn execute(
        &self,
        request: &OperatorRequest,
        invocation: OperatorInvocation<'_>,
    ) -> Result<OperatorOutput, OperatorError> {
        let candidates = self
            .operators
            .get(&request.kind)
            .map(Vec::as_slice)
            .unwrap_or_default();

        let operator = self
            .policy
            .select(request, candidates)
            .ok_or(OperatorError::NoCompatibleOperator(request.kind))?;

        operator.execute(invocation)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.operators.values().map(Vec::len).sum()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoOperator {
        backend: BackendKind,
        zero_copy: bool,
    }

    impl Operator for EchoOperator {
        fn name(&self) -> &str {
            "echo"
        }

        fn capabilities(&self) -> OperatorCapabilities {
            OperatorCapabilities {
                kind: OperatorKind::Custom,
                backend: self.backend,
                supports_zero_copy: self.zero_copy,
            }
        }

        fn execute(
            &self,
            invocation: OperatorInvocation<'_>,
        ) -> Result<OperatorOutput, OperatorError> {
            Ok(OperatorOutput {
                bytes: invocation.input.to_vec(),
            })
        }
    }

    #[test]
    fn registry_executes_injected_operator() {
        let mut registry = OperatorRegistry::new(Arc::new(FirstCompatible));
        registry.register(Arc::new(EchoOperator {
            backend: BackendKind::Cpu,
            zero_copy: false,
        }));

        let result = registry
            .execute(
                &OperatorRequest::new(OperatorKind::Custom),
                OperatorInvocation { input: b"vrb" },
            )
            .expect("registered operator should execute");

        assert_eq!(result.bytes, b"vrb");
    }

    #[test]
    fn policy_honors_zero_copy_requirement() {
        let mut registry = OperatorRegistry::new(Arc::new(FirstCompatible));
        registry.register(Arc::new(EchoOperator {
            backend: BackendKind::Cpu,
            zero_copy: false,
        }));

        let request = OperatorRequest {
            kind: OperatorKind::Custom,
            preferred_backend: None,
            requires_zero_copy: true,
        };

        assert_eq!(
            registry.execute(&request, OperatorInvocation { input: b"vrb" }),
            Err(OperatorError::NoCompatibleOperator(OperatorKind::Custom))
        );
    }
}
