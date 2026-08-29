use std::path::PathBuf;
use vrb_core::BackendKind;
use vrb_operators::OperatorKind;
use vrb_shared_operator_loader::LoadedSharedOperatorLibrary;
use vrb_shared_operators::{ExternalMemoryHandleKind, SharedOperator};

#[test]
fn dynamic_hip_shared_gemm_descriptor_loads_without_runtime_link_dependency() {
    let required = std::env::var("VRB_REQUIRE_HIP_SHARED_GEMM_LOAD_E2E").as_deref() == Ok("1");
    let Some(path) = std::env::var_os("VRB_HIP_SHARED_GEMM_PLUGIN_PATH").map(PathBuf::from) else {
        assert!(
            !required,
            "VRB_HIP_SHARED_GEMM_PLUGIN_PATH is required for dynamic load E2E"
        );
        return;
    };
    assert!(
        path.is_file(),
        "HIP shared GEMM DLL is missing: {}",
        path.display()
    );

    let library = LoadedSharedOperatorLibrary::load(&path)
        .expect("HIP shared GEMM DLL should load without ROCm runtime until execution");
    assert_eq!(library.plugin_name(), "vrb-hip-shared-gemm");
    assert_eq!(library.operators().len(), 1);
    let capabilities = library.operators()[0].capabilities();
    assert_eq!(capabilities.kind, OperatorKind::Gemm);
    assert_eq!(capabilities.backend, BackendKind::Hip);
    assert!(capabilities
        .memory_kinds
        .contains(&ExternalMemoryHandleKind::Win32Kmt));
    assert!(capabilities.proven_zero_copy);
    assert!(!capabilities.supports_external_synchronization);
}
