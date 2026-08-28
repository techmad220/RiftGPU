#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BackendId(String);

impl BackendId {
    pub fn new(value: impl Into<String>) -> Result<Self, RuntimeError> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(RuntimeError::InvalidBackendId);
        }
        Ok(Self(trimmed.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for BackendId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
    Vulkan,
    Hip,
    Hybrid,
    Plugin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    Copy,
    VectorAdd,
    Gemv,
    Gemm,
    Softmax,
    Attention,
    RmsNorm,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    F32,
    F16,
    Bf16,
    I8,
    Q4,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet {
    pub operations: Vec<OperationKind>,
    pub data_types: Vec<DataType>,
    pub external_memory: bool,
    pub external_semaphore: bool,
    pub zero_copy: bool,
}

impl CapabilitySet {
    pub fn supports(&self, request: &RouteRequest) -> bool {
        self.operations.contains(&request.operation)
            && self.data_types.contains(&request.data_type)
            && (!request.requires_external_memory || self.external_memory)
            && (!request.requires_external_semaphore || self.external_semaphore)
            && (!request.requires_zero_copy || self.zero_copy)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendProbe {
    pub id: BackendId,
    pub kind: BackendKind,
    pub name: String,
    pub vendor: String,
    pub available: bool,
    pub device_count: u32,
    pub detail: String,
    pub capabilities: CapabilitySet,
}

pub trait ComputeBackend: Send + Sync {
    fn id(&self) -> &BackendId;
    fn kind(&self) -> BackendKind;
    fn probe(&self) -> Result<BackendProbe, BackendError>;
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend unavailable: {0}")]
    Unavailable(String),
    #[error("backend probe failed: {0}")]
    Probe(String),
    #[error("backend operation unsupported: {0}")]
    Unsupported(String),
    #[error("backend internal error: {0}")]
    Internal(String),
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("backend id cannot be empty")]
    InvalidBackendId,
    #[error("backend '{0}' is already registered")]
    DuplicateBackend(BackendId),
    #[error("no compatible backend is available")]
    NoCompatibleBackend,
    #[error("routing policy failed: {0}")]
    Routing(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteRequest {
    pub operation: OperationKind,
    pub data_type: DataType,
    pub requires_external_memory: bool,
    pub requires_external_semaphore: bool,
    pub requires_zero_copy: bool,
}

impl RouteRequest {
    pub const fn new(operation: OperationKind, data_type: DataType) -> Self {
        Self {
            operation,
            data_type,
            requires_external_memory: false,
            requires_external_semaphore: false,
            requires_zero_copy: false,
        }
    }

    pub const fn zero_copy(mut self) -> Self {
        self.requires_external_memory = true;
        self.requires_external_semaphore = true;
        self.requires_zero_copy = true;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceRecord {
    pub backend: BackendId,
    pub operation: OperationKind,
    pub data_type: DataType,
    pub median_microseconds: f64,
    pub samples: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerformanceTable {
    pub records: Vec<PerformanceRecord>,
}

impl PerformanceTable {
    pub fn record(&mut self, record: PerformanceRecord) {
        if let Some(existing) = self.records.iter_mut().find(|existing| {
            existing.backend == record.backend
                && existing.operation == record.operation
                && existing.data_type == record.data_type
        }) {
            *existing = record;
        } else {
            self.records.push(record);
        }
    }

    pub fn median_us(
        &self,
        backend: &BackendId,
        operation: OperationKind,
        data_type: DataType,
    ) -> Option<f64> {
        self.records
            .iter()
            .find(|record| {
                &record.backend == backend
                    && record.operation == operation
                    && record.data_type == data_type
            })
            .map(|record| record.median_microseconds)
    }
}

pub trait RoutingPolicy: Send + Sync {
    fn select(
        &self,
        request: &RouteRequest,
        candidates: &[BackendProbe],
        performance: &PerformanceTable,
    ) -> Result<BackendId, RuntimeError>;
}

#[derive(Debug, Default)]
pub struct FastestCompatible;

impl RoutingPolicy for FastestCompatible {
    fn select(
        &self,
        request: &RouteRequest,
        candidates: &[BackendProbe],
        performance: &PerformanceTable,
    ) -> Result<BackendId, RuntimeError> {
        let compatible: Vec<&BackendProbe> = candidates
            .iter()
            .filter(|probe| probe.available && probe.capabilities.supports(request))
            .collect();

        if compatible.is_empty() {
            return Err(RuntimeError::NoCompatibleBackend);
        }

        if let Some(best) = compatible
            .iter()
            .filter_map(|probe| {
                performance
                    .median_us(&probe.id, request.operation, request.data_type)
                    .filter(|value| value.is_finite() && *value >= 0.0)
                    .map(|value| (*probe, value))
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))
        {
            return Ok(best.0.id.clone());
        }

        // When no benchmark exists, prefer native GPU compute, then portable GPU,
        // then hybrid/plugin implementations, and CPU last. Hardware measurements
        // replace this bootstrap ordering as soon as they are available.
        let rank = |kind: BackendKind| match kind {
            BackendKind::Hip => 0_u8,
            BackendKind::Vulkan => 1,
            BackendKind::Hybrid => 2,
            BackendKind::Plugin => 3,
            BackendKind::Cpu => 4,
        };

        compatible
            .into_iter()
            .min_by_key(|probe| rank(probe.kind))
            .map(|probe| probe.id.clone())
            .ok_or(RuntimeError::NoCompatibleBackend)
    }
}

#[derive(Default)]
pub struct BackendRegistry {
    backends: BTreeMap<BackendId, Arc<dyn ComputeBackend>>,
}

impl BackendRegistry {
    pub fn register(&mut self, backend: Arc<dyn ComputeBackend>) -> Result<(), RuntimeError> {
        let id = backend.id().clone();
        if self.backends.contains_key(&id) {
            return Err(RuntimeError::DuplicateBackend(id));
        }
        self.backends.insert(id, backend);
        Ok(())
    }

    pub fn get(&self, id: &BackendId) -> Option<Arc<dyn ComputeBackend>> {
        self.backends.get(id).cloned()
    }

    pub fn probes(&self) -> Vec<BackendProbe> {
        self.backends
            .values()
            .map(|backend| match backend.probe() {
                Ok(probe) => probe,
                Err(error) => BackendProbe {
                    id: backend.id().clone(),
                    kind: backend.kind(),
                    name: backend.id().to_string(),
                    vendor: String::new(),
                    available: false,
                    device_count: 0,
                    detail: error.to_string(),
                    capabilities: CapabilitySet {
                        operations: Vec::new(),
                        data_types: Vec::new(),
                        external_memory: false,
                        external_semaphore: false,
                        zero_copy: false,
                    },
                },
            })
            .collect()
    }
}

pub struct Runtime {
    registry: BackendRegistry,
    policy: Arc<dyn RoutingPolicy>,
    performance: PerformanceTable,
}

impl Runtime {
    pub fn probes(&self) -> Vec<BackendProbe> {
        self.registry.probes()
    }

    pub fn route(&self, request: &RouteRequest) -> Result<BackendId, RuntimeError> {
        let probes = self.registry.probes();
        self.policy.select(request, &probes, &self.performance)
    }

    pub fn backend(&self, id: &BackendId) -> Option<Arc<dyn ComputeBackend>> {
        self.registry.get(id)
    }

    pub fn performance(&self) -> &PerformanceTable {
        &self.performance
    }
}

pub struct RuntimeBuilder {
    registry: BackendRegistry,
    policy: Arc<dyn RoutingPolicy>,
    performance: PerformanceTable,
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self {
            registry: BackendRegistry::default(),
            policy: Arc::new(FastestCompatible),
            performance: PerformanceTable::default(),
        }
    }
}

impl RuntimeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn backend(mut self, backend: Arc<dyn ComputeBackend>) -> Result<Self, RuntimeError> {
        self.registry.register(backend)?;
        Ok(self)
    }

    pub fn routing_policy(mut self, policy: Arc<dyn RoutingPolicy>) -> Self {
        self.policy = policy;
        self
    }

    pub fn performance_table(mut self, performance: PerformanceTable) -> Self {
        self.performance = performance;
        self
    }

    pub fn build(self) -> Runtime {
        Runtime {
            registry: self.registry,
            policy: self.policy,
            performance: self.performance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeBackend {
        id: BackendId,
        kind: BackendKind,
        zero_copy: bool,
    }

    impl ComputeBackend for FakeBackend {
        fn id(&self) -> &BackendId {
            &self.id
        }

        fn kind(&self) -> BackendKind {
            self.kind
        }

        fn probe(&self) -> Result<BackendProbe, BackendError> {
            Ok(BackendProbe {
                id: self.id.clone(),
                kind: self.kind,
                name: self.id.to_string(),
                vendor: "test".to_owned(),
                available: true,
                device_count: 1,
                detail: "ok".to_owned(),
                capabilities: CapabilitySet {
                    operations: vec![OperationKind::Gemm],
                    data_types: vec![DataType::F32],
                    external_memory: self.zero_copy,
                    external_semaphore: self.zero_copy,
                    zero_copy: self.zero_copy,
                },
            })
        }
    }

    fn fake(id: &str, kind: BackendKind, zero_copy: bool) -> Arc<dyn ComputeBackend> {
        Arc::new(FakeBackend {
            id: BackendId::new(id).unwrap(),
            kind,
            zero_copy,
        })
    }

    #[test]
    fn benchmark_data_overrides_bootstrap_preference() {
        let mut table = PerformanceTable::default();
        table.record(PerformanceRecord {
            backend: BackendId::new("vulkan").unwrap(),
            operation: OperationKind::Gemm,
            data_type: DataType::F32,
            median_microseconds: 10.0,
            samples: 20,
        });
        table.record(PerformanceRecord {
            backend: BackendId::new("hip").unwrap(),
            operation: OperationKind::Gemm,
            data_type: DataType::F32,
            median_microseconds: 14.0,
            samples: 20,
        });

        let runtime = RuntimeBuilder::new()
            .backend(fake("hip", BackendKind::Hip, true))
            .unwrap()
            .backend(fake("vulkan", BackendKind::Vulkan, true))
            .unwrap()
            .performance_table(table)
            .build();

        let selected = runtime
            .route(&RouteRequest::new(OperationKind::Gemm, DataType::F32))
            .unwrap();
        assert_eq!(selected.as_str(), "vulkan");
    }

    #[test]
    fn zero_copy_requirement_filters_incompatible_backends() {
        let runtime = RuntimeBuilder::new()
            .backend(fake("cpu", BackendKind::Cpu, false))
            .unwrap()
            .backend(fake("hip", BackendKind::Hip, true))
            .unwrap()
            .build();

        let selected = runtime
            .route(&RouteRequest::new(OperationKind::Gemm, DataType::F32).zero_copy())
            .unwrap();
        assert_eq!(selected.as_str(), "hip");
    }

    #[test]
    fn duplicate_backend_ids_fail_closed() {
        let result = RuntimeBuilder::new()
            .backend(fake("same", BackendKind::Cpu, false))
            .unwrap()
            .backend(fake("same", BackendKind::Vulkan, false));
        assert!(matches!(result, Err(RuntimeError::DuplicateBackend(_))));
    }
}
