use thiserror::Error;
use vrb_core::BackendKind;
use vrb_gemm_protocol::{
    decode_request, encode_response, encoded_response_len, inspect_request, GemmProtocolError,
    GemmRequestMeta,
};
use vrb_operators::{
    Operator, OperatorCapabilities, OperatorError, OperatorInvocation, OperatorKind, OperatorOutput,
};

pub const REFERENCE_GEMM_NAME: &str = "cpu-reference-gemm";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmLimits {
    pub max_matrix_elements: u64,
    pub max_multiply_adds: u64,
}

impl Default for GemmLimits {
    fn default() -> Self {
        Self {
            max_matrix_elements: 16 * 1024 * 1024,
            max_multiply_adds: 1_000_000_000,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ReferenceGemmError {
    #[error(transparent)]
    Protocol(#[from] GemmProtocolError),
    #[error("GEMM {resource} requires {actual} units, exceeding configured limit {maximum}")]
    ResourceLimit {
        resource: &'static str,
        actual: u128,
        maximum: u128,
    },
    #[error("GEMM dimensions cannot be represented in this process address space")]
    AddressSpaceOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuReferenceGemm {
    limits: GemmLimits,
}

impl Default for CpuReferenceGemm {
    fn default() -> Self {
        Self::new(GemmLimits::default())
    }
}

impl CpuReferenceGemm {
    #[must_use]
    pub const fn new(limits: GemmLimits) -> Self {
        Self { limits }
    }

    #[must_use]
    pub const fn limits(&self) -> GemmLimits {
        self.limits
    }

    pub fn required_output_len(&self, input: &[u8]) -> Result<u64, ReferenceGemmError> {
        required_output_len(input, self.limits)
    }

    pub fn execute_bytes(&self, input: &[u8]) -> Result<Vec<u8>, ReferenceGemmError> {
        execute_gemm_bytes(input, self.limits)
    }
}

impl Operator for CpuReferenceGemm {
    fn name(&self) -> &str {
        REFERENCE_GEMM_NAME
    }

    fn capabilities(&self) -> OperatorCapabilities {
        OperatorCapabilities {
            kind: OperatorKind::Gemm,
            backend: BackendKind::Cpu,
            supports_zero_copy: false,
        }
    }

    fn execute(&self, invocation: OperatorInvocation<'_>) -> Result<OperatorOutput, OperatorError> {
        self.execute_bytes(invocation.input)
            .map(|bytes| OperatorOutput { bytes })
            .map_err(|error| OperatorError::Execution(error.to_string()))
    }
}

pub fn required_output_len(input: &[u8], limits: GemmLimits) -> Result<u64, ReferenceGemmError> {
    let meta = inspect_request(input)?;
    validate_limits(&meta, limits)?;
    Ok(encoded_response_len(meta.m, meta.n)?)
}

pub fn execute_gemm_bytes(input: &[u8], limits: GemmLimits) -> Result<Vec<u8>, ReferenceGemmError> {
    let meta = inspect_request(input)?;
    validate_limits(&meta, limits)?;
    let request = decode_request(input)?;

    let m = usize::try_from(request.meta.m).map_err(|_| ReferenceGemmError::AddressSpaceOverflow)?;
    let n = usize::try_from(request.meta.n).map_err(|_| ReferenceGemmError::AddressSpaceOverflow)?;
    let k = usize::try_from(request.meta.k).map_err(|_| ReferenceGemmError::AddressSpaceOverflow)?;
    let output_elements = usize::try_from(request.meta.c_elements)
        .map_err(|_| ReferenceGemmError::AddressSpaceOverflow)?;

    let mut output = vec![0.0_f32; output_elements];
    if request.meta.beta != 0.0 {
        if let Some(c) = request.c.as_deref() {
            for (destination, source) in output.iter_mut().zip(c.iter().copied()) {
                *destination = request.meta.beta * source;
            }
        }
    }

    if request.meta.alpha != 0.0 && m != 0 && n != 0 && k != 0 {
        for row in 0..m {
            let a_row = row * k;
            let output_row = row * n;
            for depth in 0..k {
                let scaled_a = request.meta.alpha * request.a[a_row + depth];
                let b_row = depth * n;
                for column in 0..n {
                    output[output_row + column] += scaled_a * request.b[b_row + column];
                }
            }
        }
    }

    Ok(encode_response(request.meta.m, request.meta.n, &output)?)
}

fn validate_limits(meta: &GemmRequestMeta, limits: GemmLimits) -> Result<(), ReferenceGemmError> {
    for (resource, elements) in [
        ("matrix A elements", meta.a_elements),
        ("matrix B elements", meta.b_elements),
        ("matrix C/output elements", meta.c_elements),
    ] {
        if elements > limits.max_matrix_elements {
            return Err(ReferenceGemmError::ResourceLimit {
                resource,
                actual: u128::from(elements),
                maximum: u128::from(limits.max_matrix_elements),
            });
        }
    }

    let operations = u128::from(meta.m) * u128::from(meta.n) * u128::from(meta.k);
    if operations > u128::from(limits.max_multiply_adds) {
        return Err(ReferenceGemmError::ResourceLimit {
            resource: "multiply-add count",
            actual: operations,
            maximum: u128::from(limits.max_multiply_adds),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vrb_gemm_protocol::{decode_response, encode_request};

    #[test]
    fn reference_gemm_matches_known_product() {
        let request = encode_request(
            2,
            2,
            3,
            1.0,
            0.0,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            None,
        )
        .expect("request should encode");

        let response = execute_gemm_bytes(&request, GemmLimits::default())
            .expect("reference GEMM should execute");
        let decoded = decode_response(&response).expect("response should decode");

        assert_eq!(decoded.values, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn alpha_beta_and_c_are_applied() {
        let request = encode_request(
            1,
            2,
            2,
            0.5,
            2.0,
            &[2.0, 4.0],
            &[1.0, 3.0, 2.0, 5.0],
            Some(&[10.0, 20.0]),
        )
        .expect("request should encode");

        let response = CpuReferenceGemm::default()
            .execute_bytes(&request)
            .expect("reference GEMM should execute");
        let decoded = decode_response(&response).expect("response should decode");

        assert_eq!(decoded.values, [25.0, 53.0]);
    }

    #[test]
    fn configured_limits_reject_excessive_work_before_decode_allocation() {
        let request = encode_request(2, 2, 2, 1.0, 0.0, &[1.0; 4], &[1.0; 4], None)
            .expect("request should encode");
        let limits = GemmLimits {
            max_matrix_elements: 16,
            max_multiply_adds: 7,
        };

        let error = required_output_len(&request, limits)
            .expect_err("work above configured operation limit must fail");
        assert_eq!(
            error,
            ReferenceGemmError::ResourceLimit {
                resource: "multiply-add count",
                actual: 8,
                maximum: 7,
            }
        );
    }
}
