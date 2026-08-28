use std::sync::Arc;

use vrb_core::BackendKind;
use vrb_operators::{
    FirstCompatible, Operator, OperatorCapabilities, OperatorError, OperatorInvocation,
    OperatorKind, OperatorOutput, OperatorRegistry, OperatorRequest,
};

struct TestOperator {
    name: &'static str,
    backend: BackendKind,
    zero_copy: bool,
}

impl Operator for TestOperator {
    fn name(&self) -> &str {
        self.name
    }

    fn capabilities(&self) -> OperatorCapabilities {
        OperatorCapabilities {
            kind: OperatorKind::Gemm,
            backend: self.backend,
            supports_zero_copy: self.zero_copy,
        }
    }

    fn execute(&self, invocation: OperatorInvocation<'_>) -> Result<OperatorOutput, OperatorError> {
        Ok(OperatorOutput {
            bytes: invocation.input.to_vec(),
        })
    }
}

#[test]
fn prefers_requested_backend_when_compatible() {
    let mut registry = OperatorRegistry::new(Arc::new(FirstCompatible));
    registry.register(Arc::new(TestOperator {
        name: "cpu-gemm",
        backend: BackendKind::Cpu,
        zero_copy: false,
    }));
    registry.register(Arc::new(TestOperator {
        name: "hip-gemm",
        backend: BackendKind::Hip,
        zero_copy: true,
    }));

    let request = OperatorRequest {
        kind: OperatorKind::Gemm,
        preferred_backend: Some(BackendKind::Hip),
        requires_zero_copy: true,
    };

    let result = registry
        .execute(&request, OperatorInvocation { input: b"matrix" })
        .expect("HIP operator should satisfy the request");

    assert_eq!(result.bytes, b"matrix");
}
