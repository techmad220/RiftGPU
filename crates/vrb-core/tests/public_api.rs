use std::sync::Arc;
use vrb_core::{
    BackendError, BackendId, BackendKind, BackendProbe, BackendRegistry, CapabilitySet,
    ComputeBackend, DataType, FastestCompatible, OperationKind, PerformanceRecord,
    PerformanceTable, RouteRequest, RoutingPolicy, RuntimeBuilder, RuntimeError,
};

#[derive(Clone)]
struct FixtureBackend {
    id: BackendId,
    kind: BackendKind,
    available: bool,
    fail_probe: bool,
    capabilities: CapabilitySet,
}

impl ComputeBackend for FixtureBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn kind(&self) -> BackendKind {
        self.kind
    }

    fn probe(&self) -> Result<BackendProbe, BackendError> {
        if self.fail_probe {
            return Err(BackendError::Probe("intentional certification failure".to_owned()));
        }
        Ok(BackendProbe {
            id: self.id.clone(),
            kind: self.kind,
            name: format!("{} fixture", self.id),
            vendor: "certification".to_owned(),
            available: self.available,
            device_count: u32::from(self.available),
            detail: "fixture".to_owned(),
            capabilities: self.capabilities.clone(),
        })
    }
}

fn caps(zero_copy: bool) -> CapabilitySet {
    CapabilitySet {
        operations: vec![OperationKind::Copy, OperationKind::Gemm],
        data_types: vec![DataType::F32, DataType::I8],
        external_memory: zero_copy,
        external_semaphore: zero_copy,
        zero_copy,
    }
}

fn fixture(id: &str, kind: BackendKind, available: bool, zero_copy: bool) -> Arc<dyn ComputeBackend> {
    Arc::new(FixtureBackend {
        id: BackendId::new(id).expect("valid fixture id"),
        kind,
        available,
        fail_probe: false,
        capabilities: caps(zero_copy),
    })
}

#[test]
fn backend_id_trims_rejects_empty_and_displays() {
    let id = BackendId::new("  hip-main  ").expect("trimmed id should be valid");
    assert_eq!(id.as_str(), "hip-main");
    assert_eq!(id.to_string(), "hip-main");
    assert!(matches!(BackendId::new("   "), Err(RuntimeError::InvalidBackendId)));
}

#[test]
fn route_request_and_capability_requirements_are_fail_closed() {
    let normal = RouteRequest::new(OperationKind::Gemm, DataType::F32);
    let zero = normal.zero_copy();
    assert!(!normal.requires_external_memory);
    assert!(zero.requires_external_memory);
    assert!(zero.requires_external_semaphore);
    assert!(zero.requires_zero_copy);
    assert!(caps(true).supports(&zero));
    assert!(!caps(false).supports(&zero));
    assert!(!caps(true).supports(&RouteRequest::new(OperationKind::Attention, DataType::F32)));
    assert!(!caps(true).supports(&RouteRequest::new(OperationKind::Gemm, DataType::F16)));
}

#[test]
fn performance_table_inserts_updates_and_queries_exact_keys() {
    let backend = BackendId::new("hip").unwrap();
    let mut table = PerformanceTable::default();
    table.record(PerformanceRecord {
        backend: backend.clone(),
        operation: OperationKind::Gemm,
        data_type: DataType::F32,
        median_microseconds: 20.0,
        samples: 5,
    });
    table.record(PerformanceRecord {
        backend: backend.clone(),
        operation: OperationKind::Gemm,
        data_type: DataType::F32,
        median_microseconds: 11.0,
        samples: 9,
    });

    assert_eq!(table.records.len(), 1);
    assert_eq!(table.median_us(&backend, OperationKind::Gemm, DataType::F32), Some(11.0));
    assert_eq!(table.median_us(&backend, OperationKind::Copy, DataType::F32), None);
}

#[test]
fn fastest_compatible_prefers_measurement_then_kind_and_ignores_bad_measurements() {
    let hip = fixture("hip", BackendKind::Hip, true, true).probe().unwrap();
    let vulkan = fixture("vulkan", BackendKind::Vulkan, true, true).probe().unwrap();
    let cpu = fixture("cpu", BackendKind::Cpu, true, false).probe().unwrap();
    let request = RouteRequest::new(OperationKind::Gemm, DataType::F32);
    let policy = FastestCompatible;

    let bootstrap = policy
        .select(&request, &[cpu.clone(), vulkan.clone(), hip.clone()], &PerformanceTable::default())
        .unwrap();
    assert_eq!(bootstrap.as_str(), "hip");

    let mut measured = PerformanceTable::default();
    measured.record(PerformanceRecord {
        backend: hip.id.clone(),
        operation: OperationKind::Gemm,
        data_type: DataType::F32,
        median_microseconds: f64::NAN,
        samples: 4,
    });
    measured.record(PerformanceRecord {
        backend: vulkan.id.clone(),
        operation: OperationKind::Gemm,
        data_type: DataType::F32,
        median_microseconds: 8.0,
        samples: 4,
    });
    assert_eq!(policy.select(&request, &[hip, vulkan], &measured).unwrap().as_str(), "vulkan");
}

#[test]
fn fastest_compatible_rejects_unavailable_or_incompatible_candidates() {
    let unavailable = fixture("hip", BackendKind::Hip, false, true).probe().unwrap();
    let cpu = fixture("cpu", BackendKind::Cpu, true, false).probe().unwrap();
    let request = RouteRequest::new(OperationKind::Gemm, DataType::F32).zero_copy();
    let result = FastestCompatible.select(&request, &[unavailable, cpu], &PerformanceTable::default());
    assert!(matches!(result, Err(RuntimeError::NoCompatibleBackend)));
}

#[test]
fn registry_get_probes_and_probe_failure_conversion_are_certified() {
    let mut registry = BackendRegistry::default();
    registry.register(fixture("cpu", BackendKind::Cpu, true, false)).unwrap();
    let broken: Arc<dyn ComputeBackend> = Arc::new(FixtureBackend {
        id: BackendId::new("broken").unwrap(),
        kind: BackendKind::Plugin,
        available: false,
        fail_probe: true,
        capabilities: caps(false),
    });
    registry.register(broken).unwrap();

    assert!(registry.get(&BackendId::new("cpu").unwrap()).is_some());
    assert!(registry.get(&BackendId::new("missing").unwrap()).is_none());
    let probes = registry.probes();
    assert_eq!(probes.len(), 2);
    let broken_probe = probes.iter().find(|probe| probe.id.as_str() == "broken").unwrap();
    assert!(!broken_probe.available);
    assert_eq!(broken_probe.device_count, 0);
    assert!(broken_probe.detail.contains("intentional certification failure"));
}

struct ForceCpu;

impl RoutingPolicy for ForceCpu {
    fn select(
        &self,
        _request: &RouteRequest,
        candidates: &[BackendProbe],
        _performance: &PerformanceTable,
    ) -> Result<BackendId, RuntimeError> {
        candidates
            .iter()
            .find(|probe| probe.id.as_str() == "cpu")
            .map(|probe| probe.id.clone())
            .ok_or_else(|| RuntimeError::Routing("cpu missing".to_owned()))
    }
}

#[test]
fn runtime_builder_runtime_accessors_custom_policy_and_duplicate_detection_work() {
    let mut perf = PerformanceTable::default();
    perf.record(PerformanceRecord {
        backend: BackendId::new("cpu").unwrap(),
        operation: OperationKind::Copy,
        data_type: DataType::I8,
        median_microseconds: 3.0,
        samples: 3,
    });

    let runtime = RuntimeBuilder::new()
        .backend(fixture("cpu", BackendKind::Cpu, true, false))
        .unwrap()
        .backend(fixture("hip", BackendKind::Hip, true, true))
        .unwrap()
        .routing_policy(Arc::new(ForceCpu))
        .performance_table(perf.clone())
        .build();

    assert_eq!(runtime.probes().len(), 2);
    assert_eq!(runtime.performance(), &perf);
    assert!(runtime.backend(&BackendId::new("hip").unwrap()).is_some());
    assert_eq!(runtime.route(&RouteRequest::new(OperationKind::Gemm, DataType::F32)).unwrap().as_str(), "cpu");

    let duplicate = RuntimeBuilder::new()
        .backend(fixture("same", BackendKind::Cpu, true, false))
        .unwrap()
        .backend(fixture("same", BackendKind::Hip, true, true));
    assert!(matches!(duplicate, Err(RuntimeError::DuplicateBackend(_))));
}
