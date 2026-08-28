//! Shared-resource operator abstractions layered above the generic operator API.
//!
//! This crate contains no Vulkan, HIP, OS-handle, or dynamic-loading code. It
//! models borrowed external-memory regions and synchronization points so a
//! concrete transport can inject them without copying tensor payloads through
//! host memory. Ownership of every native handle remains with the caller.

use std::{collections::BTreeMap, sync::Arc};

use thiserror::Error;
use vrb_core::BackendKind;
use vrb_operators::OperatorKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExternalMemoryHandleKind {
    Win32Kmt,
    Win32Nt,
    OpaqueFd,
    DmaBuf,
    Custom(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedResourceRegion {
    pub handle_kind: ExternalMemoryHandleKind,
    /// Borrowed native-handle token. The operator must not close or retain it.
    pub handle: u64,
    pub allocation_size: u64,
    pub offset: u64,
    pub length: u64,
    pub access: ResourceAccess,
}

impl SharedResourceRegion {
    pub fn validate(&self) -> Result<(), SharedOperatorError> {
        if self.handle == 0 {
            return Err(SharedOperatorError::InvalidResource(
                "external-memory handle must be non-zero".to_owned(),
            ));
        }
        if self.allocation_size == 0 || self.length == 0 {
            return Err(SharedOperatorError::InvalidResource(
                "shared-resource allocation and region length must be non-zero".to_owned(),
            ));
        }
        let end = self.offset.checked_add(self.length).ok_or_else(|| {
            SharedOperatorError::InvalidResource("shared-resource range overflow".to_owned())
        })?;
        if end > self.allocation_size {
            return Err(SharedOperatorError::InvalidResource(format!(
                "shared-resource range [{}, {}) exceeds allocation size {}",
                self.offset, end, self.allocation_size
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExternalSyncHandleKind {
    Win32Opaque,
    OpaqueFd,
    Timeline,
    Custom(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedSyncPoint {
    pub handle_kind: ExternalSyncHandleKind,
    /// Borrowed native-handle token. The operator must not close or retain it.
    pub handle: u64,
    /// Timeline/fence value when applicable; zero is valid for binary primitives.
    pub value: u64,
}

impl SharedSyncPoint {
    pub fn validate(&self) -> Result<(), SharedOperatorError> {
        if self.handle == 0 {
            return Err(SharedOperatorError::InvalidSynchronization(
                "external synchronization handle must be non-zero".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedOperatorRequest {
    pub kind: OperatorKind,
    pub preferred_backend: Option<BackendKind>,
    pub required_memory_kind: Option<ExternalMemoryHandleKind>,
    pub requires_synchronization: bool,
    pub requires_proven_zero_copy: bool,
}

impl SharedOperatorRequest {
    #[must_use]
    pub const fn new(kind: OperatorKind) -> Self {
        Self {
            kind,
            preferred_backend: None,
            required_memory_kind: None,
            requires_synchronization: false,
            requires_proven_zero_copy: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedOperatorCapabilities {
    pub kind: OperatorKind,
    pub backend: BackendKind,
    pub memory_kinds: Vec<ExternalMemoryHandleKind>,
    pub sync_kinds: Vec<ExternalSyncHandleKind>,
    pub supports_external_synchronization: bool,
    /// True only for implementations whose execution path actually consumes the
    /// shared allocation directly rather than relaying tensor bytes through host memory.
    pub proven_zero_copy: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedOperatorInvocation<'a> {
    /// Small host-side control metadata only; bulk tensor payloads belong in resources.
    pub metadata: &'a [u8],
    pub resources: &'a [SharedResourceRegion],
    pub waits: &'a [SharedSyncPoint],
    pub signals: &'a [SharedSyncPoint],
}

impl SharedOperatorInvocation<'_> {
    pub fn validate(&self) -> Result<(), SharedOperatorError> {
        for resource in self.resources {
            resource.validate()?;
        }
        for wait in self.waits {
            wait.validate()?;
        }
        for signal in self.signals {
            signal.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedOperatorOutput {
    /// Small host-side execution receipt/metadata. Bulk output remains in shared resources.
    pub receipt: Vec<u8>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SharedOperatorError {
    #[error("shared operator is unavailable: {0}")]
    Unavailable(String),
    #[error("shared operator execution failed: {0}")]
    Execution(String),
    #[error("invalid shared resource: {0}")]
    InvalidResource(String),
    #[error("invalid external synchronization point: {0}")]
    InvalidSynchronization(String),
    #[error("no compatible shared operator registered for {0:?}")]
    NoCompatibleOperator(OperatorKind),
}

pub trait SharedOperator: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> SharedOperatorCapabilities;
    fn execute_shared(
        &self,
        invocation: SharedOperatorInvocation<'_>,
    ) -> Result<SharedOperatorOutput, SharedOperatorError>;
}

pub trait SharedOperatorSelectionPolicy: Send + Sync {
    fn select(
        &self,
        request: &SharedOperatorRequest,
        invocation: &SharedOperatorInvocation<'_>,
        candidates: &[Arc<dyn SharedOperator>],
    ) -> Option<Arc<dyn SharedOperator>>;
}

#[derive(Debug, Default)]
pub struct FirstCompatibleShared;

impl SharedOperatorSelectionPolicy for FirstCompatibleShared {
    fn select(
        &self,
        request: &SharedOperatorRequest,
        invocation: &SharedOperatorInvocation<'_>,
        candidates: &[Arc<dyn SharedOperator>],
    ) -> Option<Arc<dyn SharedOperator>> {
        let compatible = |operator: &&Arc<dyn SharedOperator>| {
            let capabilities = operator.capabilities();
            let invocation_uses_sync =
                !invocation.waits.is_empty() || !invocation.signals.is_empty();
            capabilities.kind == request.kind
                && request
                    .required_memory_kind
                    .is_none_or(|kind| capabilities.memory_kinds.contains(&kind))
                && invocation
                    .resources
                    .iter()
                    .all(|resource| capabilities.memory_kinds.contains(&resource.handle_kind))
                && (!request.requires_synchronization
                    || capabilities.supports_external_synchronization)
                && (!invocation_uses_sync || capabilities.supports_external_synchronization)
                && invocation
                    .waits
                    .iter()
                    .chain(invocation.signals.iter())
                    .all(|point| capabilities.sync_kinds.contains(&point.handle_kind))
                && (!request.requires_proven_zero_copy || capabilities.proven_zero_copy)
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

        candidates.iter().find(compatible).cloned()
    }
}

pub struct SharedOperatorRegistry {
    operators: BTreeMap<OperatorKind, Vec<Arc<dyn SharedOperator>>>,
    policy: Arc<dyn SharedOperatorSelectionPolicy>,
}

impl SharedOperatorRegistry {
    #[must_use]
    pub fn new(policy: Arc<dyn SharedOperatorSelectionPolicy>) -> Self {
        Self {
            operators: BTreeMap::new(),
            policy,
        }
    }

    pub fn register(&mut self, operator: Arc<dyn SharedOperator>) {
        self.operators
            .entry(operator.capabilities().kind)
            .or_default()
            .push(operator);
    }

    pub fn execute(
        &self,
        request: &SharedOperatorRequest,
        invocation: SharedOperatorInvocation<'_>,
    ) -> Result<SharedOperatorOutput, SharedOperatorError> {
        invocation.validate()?;
        let candidates = self
            .operators
            .get(&request.kind)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let operator = self
            .policy
            .select(request, &invocation, candidates)
            .ok_or(SharedOperatorError::NoCompatibleOperator(request.kind))?;
        operator.execute_shared(invocation)
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

    struct ReceiptOperator;

    impl SharedOperator for ReceiptOperator {
        fn name(&self) -> &str {
            "receipt"
        }

        fn capabilities(&self) -> SharedOperatorCapabilities {
            SharedOperatorCapabilities {
                kind: OperatorKind::Custom,
                backend: BackendKind::Plugin,
                memory_kinds: vec![ExternalMemoryHandleKind::Win32Kmt],
                sync_kinds: vec![ExternalSyncHandleKind::Win32Opaque],
                supports_external_synchronization: true,
                proven_zero_copy: false,
            }
        }

        fn execute_shared(
            &self,
            invocation: SharedOperatorInvocation<'_>,
        ) -> Result<SharedOperatorOutput, SharedOperatorError> {
            Ok(SharedOperatorOutput {
                receipt: invocation.metadata.to_vec(),
            })
        }
    }

    fn valid_kmt_resource() -> SharedResourceRegion {
        SharedResourceRegion {
            handle_kind: ExternalMemoryHandleKind::Win32Kmt,
            handle: 1,
            allocation_size: 64,
            offset: 0,
            length: 64,
            access: ResourceAccess::ReadWrite,
        }
    }

    #[test]
    fn invalid_region_fails_before_operator_execution() {
        let mut registry = SharedOperatorRegistry::new(Arc::new(FirstCompatibleShared));
        registry.register(Arc::new(ReceiptOperator));
        let resource = SharedResourceRegion {
            offset: 48,
            length: 32,
            ..valid_kmt_resource()
        };
        let error = registry
            .execute(
                &SharedOperatorRequest::new(OperatorKind::Custom),
                SharedOperatorInvocation {
                    metadata: b"x",
                    resources: &[resource],
                    waits: &[],
                    signals: &[],
                },
            )
            .expect_err("out-of-range shared region must be rejected");
        assert!(matches!(error, SharedOperatorError::InvalidResource(_)));
    }

    #[test]
    fn actual_resource_kind_is_part_of_selection() {
        let mut registry = SharedOperatorRegistry::new(Arc::new(FirstCompatibleShared));
        registry.register(Arc::new(ReceiptOperator));
        let resource = SharedResourceRegion {
            handle_kind: ExternalMemoryHandleKind::DmaBuf,
            ..valid_kmt_resource()
        };
        assert_eq!(
            registry.execute(
                &SharedOperatorRequest::new(OperatorKind::Custom),
                SharedOperatorInvocation {
                    metadata: b"x",
                    resources: &[resource],
                    waits: &[],
                    signals: &[],
                },
            ),
            Err(SharedOperatorError::NoCompatibleOperator(
                OperatorKind::Custom
            ))
        );
    }

    #[test]
    fn actual_sync_kind_is_part_of_selection() {
        let mut registry = SharedOperatorRegistry::new(Arc::new(FirstCompatibleShared));
        registry.register(Arc::new(ReceiptOperator));
        let resource = valid_kmt_resource();
        let wait = SharedSyncPoint {
            handle_kind: ExternalSyncHandleKind::Timeline,
            handle: 2,
            value: 7,
        };
        assert_eq!(
            registry.execute(
                &SharedOperatorRequest::new(OperatorKind::Custom),
                SharedOperatorInvocation {
                    metadata: b"x",
                    resources: &[resource],
                    waits: &[wait],
                    signals: &[],
                },
            ),
            Err(SharedOperatorError::NoCompatibleOperator(
                OperatorKind::Custom
            ))
        );
    }

    #[test]
    fn proven_zero_copy_is_not_inferred_from_shared_resource_support() {
        let mut registry = SharedOperatorRegistry::new(Arc::new(FirstCompatibleShared));
        registry.register(Arc::new(ReceiptOperator));
        let request = SharedOperatorRequest {
            kind: OperatorKind::Custom,
            preferred_backend: None,
            required_memory_kind: Some(ExternalMemoryHandleKind::Win32Kmt),
            requires_synchronization: true,
            requires_proven_zero_copy: true,
        };
        let resource = valid_kmt_resource();
        assert_eq!(
            registry.execute(
                &request,
                SharedOperatorInvocation {
                    metadata: b"x",
                    resources: &[resource],
                    waits: &[],
                    signals: &[],
                },
            ),
            Err(SharedOperatorError::NoCompatibleOperator(
                OperatorKind::Custom
            ))
        );
    }
}
