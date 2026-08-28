use vrb_backends::{CpuBackend, HipBackend, VulkanBackend};
use vrb_core::{BackendError, BackendKind, ComputeBackend, DataType, OperationKind};

#[test]
fn cpu_backend_public_surface_succeeds_and_fails_closed() {
    let backend = CpuBackend::new();
    let default_backend = CpuBackend::default();
    assert_eq!(backend.id().as_str(), "cpu");
    assert_eq!(default_backend.id().as_str(), "cpu");
    assert_eq!(backend.kind(), BackendKind::Cpu);

    let probe = backend.probe().expect("CPU probe must succeed");
    assert!(probe.available);
    assert_eq!(probe.device_count, 1);
    assert!(probe.capabilities.operations.contains(&OperationKind::Copy));
    assert!(probe
        .capabilities
        .operations
        .contains(&OperationKind::VectorAdd));
    assert!(probe.capabilities.data_types.contains(&DataType::F32));
    assert!(!probe.capabilities.zero_copy);

    let left = [1.0_f32, -2.0, 3.5, 0.0];
    let right = [2.0_f32, 8.0, -1.5, 4.0];
    let mut output = [0.0_f32; 4];
    backend.vector_add_f32(&left, &right, &mut output).unwrap();
    assert_eq!(output, [3.0, 6.0, 2.0, 4.0]);

    let mut short_output = [0.0_f32; 3];
    let error = backend
        .vector_add_f32(&left, &right, &mut short_output)
        .expect_err("length mismatch must fail");
    assert!(matches!(error, BackendError::Internal(_)));
}

#[test]
fn hip_backend_constructor_identity_and_probe_contract_are_stable() {
    let backend = HipBackend::new();
    let default_backend = HipBackend::default();
    assert_eq!(backend.id().as_str(), "hip");
    assert_eq!(default_backend.id().as_str(), "hip");
    assert_eq!(backend.kind(), BackendKind::Hip);

    match backend.runtime_info() {
        Ok(info) => {
            assert!(!info.library.is_empty());
            assert!(info.runtime_version_raw > 0);
            assert!(!info.devices.is_empty());
        }
        Err(error) => {
            assert!(matches!(
                error,
                BackendError::Unavailable(_) | BackendError::Probe(_)
            ));
        }
    }

    match backend.probe() {
        Ok(probe) => {
            assert_eq!(probe.id.as_str(), "hip");
            assert!(probe.available);
            assert!(probe.device_count > 0);
        }
        Err(error) => {
            assert!(matches!(
                error,
                BackendError::Unavailable(_) | BackendError::Probe(_)
            ));
        }
    }
}

#[test]
fn vulkan_backend_constructor_identity_and_probe_contract_are_stable() {
    let backend = VulkanBackend::new();
    let default_backend = VulkanBackend::default();
    assert_eq!(backend.id().as_str(), "vulkan");
    assert_eq!(default_backend.id().as_str(), "vulkan");
    assert_eq!(backend.kind(), BackendKind::Vulkan);

    match backend.runtime_info() {
        Ok(info) => {
            assert!(info.loader_available);
            assert!(!info.devices.is_empty());
            if let Some(preferred) = info.preferred_compute_device() {
                assert!(preferred.compute_queue);
                assert!(info
                    .devices
                    .iter()
                    .filter(|device| device.compute_queue)
                    .all(|device| preferred.bridge_score() <= device.bridge_score()));
            }
        }
        Err(error) => {
            assert!(matches!(
                error,
                BackendError::Unavailable(_) | BackendError::Probe(_)
            ));
        }
    }

    match backend.probe() {
        Ok(probe) => {
            assert_eq!(probe.id.as_str(), "vulkan");
            assert_eq!(probe.available, probe.device_count > 0);
        }
        Err(error) => {
            assert!(matches!(
                error,
                BackendError::Unavailable(_) | BackendError::Probe(_)
            ));
        }
    }
}
