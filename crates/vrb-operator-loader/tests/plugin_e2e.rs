use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use vrb_core::BackendKind;
use vrb_operator_loader::LoadedOperatorLibrary;
use vrb_operators::{Operator, OperatorInvocation, OperatorKind};

#[test]
fn dynamic_operator_plugin_round_trip_and_shutdown() {
    let required = std::env::var("VRB_REQUIRE_OPERATOR_PLUGIN_E2E").as_deref() == Ok("1");
    let Some(path) = std::env::var_os("VRB_TEST_OPERATOR_PLUGIN_PATH").map(PathBuf::from) else {
        assert!(!required, "VRB_TEST_OPERATOR_PLUGIN_PATH is required for operator-plugin E2E");
        return;
    };

    assert!(path.is_file(), "operator plugin does not exist: {}", path.display());

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    let marker = std::env::temp_dir().join(format!(
        "vrb-operator-plugin-shutdown-{}-{nonce}.txt",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&marker);
    std::env::set_var("VRB_TEST_OPERATOR_PLUGIN_SHUTDOWN_MARKER", &marker);

    {
        let library = LoadedOperatorLibrary::load(&path).expect("operator plugin should load");
        assert_eq!(library.plugin_name(), "operator-certification-fixture");
        assert_eq!(library.operators().len(), 1);

        let operator = &library.operators()[0];
        assert_eq!(operator.name(), "fixture-gemm");
        assert_eq!(operator.capabilities().kind, OperatorKind::Gemm);
        assert_eq!(operator.capabilities().backend, BackendKind::Cpu);
        assert!(!operator.capabilities().supports_zero_copy);

        let output = operator
            .execute(OperatorInvocation {
                input: b"operator-plugin-e2e",
            })
            .expect("fixture operator should execute");
        assert_eq!(output.bytes, b"operator-plugin-e2e");
    }

    std::env::remove_var("VRB_TEST_OPERATOR_PLUGIN_SHUTDOWN_MARKER");
    let shutdown = std::fs::read_to_string(&marker)
        .expect("dropping the final operator library reference should call shutdown");
    assert_eq!(shutdown, "shutdown-called\n");
    let _ = std::fs::remove_file(marker);
}
