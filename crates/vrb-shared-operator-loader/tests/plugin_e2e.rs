use std::{path::PathBuf, sync::Arc};
use vrb_operators::OperatorKind;
use vrb_shared_operator_loader::LoadedSharedOperatorLibrary;
use vrb_shared_operators::{
    ExternalMemoryHandleKind, ExternalSyncHandleKind, FirstCompatibleShared, ResourceAccess,
    SharedOperator, SharedOperatorInvocation, SharedOperatorRegistry, SharedOperatorRequest,
    SharedResourceRegion, SharedSyncPoint,
};

#[test]
fn dynamic_shared_operator_round_trips_borrowed_descriptors_without_claiming_zero_copy() {
    let required = std::env::var("VRB_REQUIRE_SHARED_OPERATOR_E2E").as_deref() == Ok("1");
    let Some(path) = std::env::var_os("VRB_TEST_SHARED_OPERATOR_PLUGIN_PATH").map(PathBuf::from)
    else {
        assert!(
            !required,
            "VRB_TEST_SHARED_OPERATOR_PLUGIN_PATH is required for shared operator E2E"
        );
        return;
    };
    assert!(
        path.is_file(),
        "shared operator plugin does not exist: {}",
        path.display()
    );

    let library = LoadedSharedOperatorLibrary::load(&path)
        .expect("shared operator test plugin should load");
    assert_eq!(library.plugin_name(), "vrb-test-shared-operator");
    assert_eq!(library.operators().len(), 1);
    let capabilities = library.operators()[0].capabilities();
    assert!(capabilities.supports_external_synchronization);
    assert!(!capabilities.proven_zero_copy);
    assert!(
        capabilities
            .memory_kinds
            .contains(&ExternalMemoryHandleKind::Win32Kmt)
    );

    let mut registry = SharedOperatorRegistry::new(Arc::new(FirstCompatibleShared));
    library.register_into(&mut registry);

    let resource = SharedResourceRegion {
        handle_kind: ExternalMemoryHandleKind::Win32Kmt,
        handle: 0x1234,
        allocation_size: 4096,
        offset: 256,
        length: 1024,
        access: ResourceAccess::ReadWrite,
    };
    let wait = SharedSyncPoint {
        handle_kind: ExternalSyncHandleKind::Win32Opaque,
        handle: 0x2001,
        value: 0,
    };
    let signal = SharedSyncPoint {
        handle_kind: ExternalSyncHandleKind::Win32Opaque,
        handle: 0x2002,
        value: 0,
    };
    let invocation = SharedOperatorInvocation {
        metadata: b"vrb-shared",
        resources: &[resource],
        waits: &[wait],
        signals: &[signal],
    };
    let request = SharedOperatorRequest {
        kind: OperatorKind::Custom,
        preferred_backend: None,
        required_memory_kind: Some(ExternalMemoryHandleKind::Win32Kmt),
        requires_synchronization: true,
        requires_proven_zero_copy: false,
    };

    let output = registry
        .execute(&request, invocation.clone())
        .expect("shared operator should execute through DI registry");
    assert_eq!(output.receipt, b"shared-contract-ok");

    let proven_request = SharedOperatorRequest {
        requires_proven_zero_copy: true,
        ..request
    };
    let error = registry
        .execute(&proven_request, invocation)
        .expect_err("fixture must not satisfy a proven-zero-copy request");
    assert!(error.to_string().contains("no compatible shared operator"));
}
