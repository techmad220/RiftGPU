use std::{path::PathBuf, sync::Arc};
use vrb_core::BackendKind;
use vrb_gemm_protocol::{decode_response, encode_request};
use vrb_operator_loader::LoadedOperatorLibrary;
use vrb_operators::{
    FirstCompatible, OperatorInvocation, OperatorKind, OperatorRegistry, OperatorRequest,
};

#[test]
fn reference_gemm_dynamic_plugin_executes_through_di_registry() {
    let required = std::env::var("VRB_REQUIRE_REFERENCE_GEMM_E2E").as_deref() == Ok("1");
    let Some(path) = std::env::var_os("VRB_REFERENCE_GEMM_PLUGIN_PATH").map(PathBuf::from) else {
        assert!(
            !required,
            "VRB_REFERENCE_GEMM_PLUGIN_PATH is required for reference GEMM E2E"
        );
        return;
    };
    assert!(
        path.is_file(),
        "reference GEMM plugin does not exist: {}",
        path.display()
    );

    let library = LoadedOperatorLibrary::load(&path).expect("reference GEMM plugin should load");
    assert_eq!(library.plugin_name(), "vrb-reference-gemm");
    assert_eq!(library.operators().len(), 1);

    let mut registry = OperatorRegistry::new(Arc::new(FirstCompatible));
    library.register_into(&mut registry);

    let request_bytes = encode_request(
        2,
        2,
        3,
        1.0,
        0.0,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        None,
    )
    .expect("GEMM request should encode");
    let request = OperatorRequest {
        kind: OperatorKind::Gemm,
        preferred_backend: Some(BackendKind::Cpu),
        requires_zero_copy: false,
    };

    let output = registry
        .execute(
            &request,
            OperatorInvocation {
                input: &request_bytes,
            },
        )
        .expect("DI registry should route to the dynamic reference GEMM operator");
    let response = decode_response(&output.bytes).expect("GEMM response should decode");

    assert_eq!((response.m, response.n), (2, 2));
    assert_eq!(response.values, [58.0, 64.0, 139.0, 154.0]);
}
