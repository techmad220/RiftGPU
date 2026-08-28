use std::path::{Path, PathBuf};
use vrb_backends::{DynamicPluginBackend, PluginLoadError};
use vrb_core::{BackendKind, ComputeBackend, DataType, OperationKind};
use vrb_plugin_api::{VrbExecutionRequestV1, VrbStatus};

fn required_plugin_path() -> Option<PathBuf> {
    match std::env::var_os("VRB_TEST_PLUGIN_PATH") {
        Some(path) => Some(PathBuf::from(path)),
        None if std::env::var_os("VRB_REQUIRE_PLUGIN_E2E").is_some() => {
            panic!("VRB_TEST_PLUGIN_PATH is required when VRB_REQUIRE_PLUGIN_E2E is set")
        }
        None => None,
    }
}

#[test]
fn missing_plugin_path_fails_closed() {
    let missing = Path::new("definitely-not-a-real-vrb-plugin.dll");
    let error = DynamicPluginBackend::load(missing).expect_err("missing plugin must fail");
    assert!(matches!(error, PluginLoadError::Load { .. }));
}

#[test]
fn dynamic_plugin_load_probe_execute_error_and_shutdown_are_end_to_end() {
    let Some(path) = required_plugin_path() else {
        return;
    };
    assert!(path.is_file(), "fixture plugin does not exist: {}", path.display());

    let marker = std::env::temp_dir().join(format!(
        "vrb-plugin-shutdown-{}-{}.txt",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let _ = std::fs::remove_file(&marker);
    std::env::set_var("VRB_TEST_PLUGIN_SHUTDOWN_MARKER", &marker);

    {
        let backend = DynamicPluginBackend::load(&path).expect("fixture plugin must load");
        assert_eq!(backend.id().as_str(), "plugin:certification-fixture");
        assert_eq!(backend.kind(), BackendKind::Hybrid);

        let probe = backend.probe().expect("fixture probe must succeed");
        assert!(probe.available);
        assert_eq!(probe.device_count, 1);
        assert_eq!(probe.name, "VRB E2E Fixture");
        assert_eq!(probe.vendor, "Techmad Certification");
        assert!(probe.capabilities.operations.contains(&OperationKind::Custom));
        assert!(probe.capabilities.data_types.contains(&DataType::F32));
        assert!(probe.capabilities.external_memory);
        assert!(probe.capabilities.external_semaphore);
        assert!(probe.capabilities.zero_copy);

        let success = VrbExecutionRequestV1 {
            struct_size: std::mem::size_of::<VrbExecutionRequestV1>() as u32,
            operation: 42,
            data_type: 1,
            flags: 0,
            element_count: 1024,
            input_handle: 11,
            output_handle: 22,
            opaque: 33,
        };
        backend.execute_raw(&success).expect("operation 42 must succeed");

        let unsupported = VrbExecutionRequestV1 {
            operation: 7,
            ..success
        };
        let error = backend
            .execute_raw(&unsupported)
            .expect_err("operation 7 must propagate Unsupported");
        assert!(matches!(error, PluginLoadError::PluginStatus(VrbStatus::Unsupported)));

        let invalid = VrbExecutionRequestV1 {
            struct_size: 0,
            operation: 42,
            ..success
        };
        let error = backend
            .execute_raw(&invalid)
            .expect_err("undersized request must fail closed");
        assert!(matches!(error, PluginLoadError::PluginStatus(VrbStatus::InvalidArgument)));
        assert!(!marker.exists(), "shutdown must not run before Drop");
    }

    assert_eq!(std::fs::read(&marker).expect("shutdown marker must exist"), b"shutdown-called\n");
    let _ = std::fs::remove_file(&marker);
    std::env::remove_var("VRB_TEST_PLUGIN_SHUTDOWN_MARKER");
}
