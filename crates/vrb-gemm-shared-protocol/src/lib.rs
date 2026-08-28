use thiserror::Error;

pub const SHARED_GEMM_PROTOCOL_VERSION: u32 = 1;
pub const SHARED_GEMM_MAGIC: [u8; 8] = *b"VRBSGM01";
pub const SHARED_GEMM_HEADER_LEN: usize = 64;
pub const SHARED_GEMM_RESOURCE_COUNT: usize = 3;
pub const A_RESOURCE_INDEX: usize = 0;
pub const B_RESOURCE_INDEX: usize = 1;
pub const C_RESOURCE_INDEX: usize = 2;
const F32_BYTES: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharedGemmControl {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub alpha: f32,
    pub beta: f32,
    pub hip_device_index: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedGemmResourceLengths {
    pub a_bytes: u64,
    pub b_bytes: u64,
    pub c_bytes: u64,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SharedGemmProtocolError {
    #[error("shared GEMM metadata is truncated: need {minimum} bytes, got {actual}")]
    Truncated { minimum: usize, actual: usize },
    #[error("invalid shared GEMM metadata magic")]
    InvalidMagic,
    #[error("unsupported shared GEMM protocol version {actual}; expected {expected}")]
    UnsupportedVersion { actual: u32, expected: u32 },
    #[error("invalid shared GEMM header length {actual}; expected {expected}")]
    InvalidHeaderLength { actual: u32, expected: usize },
    #[error("shared GEMM metadata length {actual} does not match required length {expected}")]
    LengthMismatch { expected: usize, actual: usize },
    #[error("shared GEMM dimensions overflow while computing {0}")]
    DimensionOverflow(&'static str),
    #[error("shared GEMM byte length overflow while computing {0}")]
    ByteLengthOverflow(&'static str),
    #[error("HIP device index must be non-negative, got {0}")]
    InvalidDeviceIndex(i32),
}

#[must_use]
pub fn encode_control(control: SharedGemmControl) -> Result<Vec<u8>, SharedGemmProtocolError> {
    validate_control(control)?;
    let mut output = Vec::with_capacity(SHARED_GEMM_HEADER_LEN);
    output.extend_from_slice(&SHARED_GEMM_MAGIC);
    output.extend_from_slice(&SHARED_GEMM_PROTOCOL_VERSION.to_le_bytes());
    output.extend_from_slice(&(SHARED_GEMM_HEADER_LEN as u32).to_le_bytes());
    output.extend_from_slice(&control.m.to_le_bytes());
    output.extend_from_slice(&control.n.to_le_bytes());
    output.extend_from_slice(&control.k.to_le_bytes());
    output.extend_from_slice(&control.alpha.to_le_bytes());
    output.extend_from_slice(&control.beta.to_le_bytes());
    output.extend_from_slice(&control.hip_device_index.to_le_bytes());
    output.extend_from_slice(&(SHARED_GEMM_RESOURCE_COUNT as u32).to_le_bytes());
    output.extend_from_slice(&[0_u8; 8]);
    debug_assert_eq!(output.len(), SHARED_GEMM_HEADER_LEN);
    Ok(output)
}

pub fn decode_control(input: &[u8]) -> Result<SharedGemmControl, SharedGemmProtocolError> {
    if input.len() < SHARED_GEMM_HEADER_LEN {
        return Err(SharedGemmProtocolError::Truncated {
            minimum: SHARED_GEMM_HEADER_LEN,
            actual: input.len(),
        });
    }
    if input.len() != SHARED_GEMM_HEADER_LEN {
        return Err(SharedGemmProtocolError::LengthMismatch {
            expected: SHARED_GEMM_HEADER_LEN,
            actual: input.len(),
        });
    }
    if input[..8] != SHARED_GEMM_MAGIC {
        return Err(SharedGemmProtocolError::InvalidMagic);
    }
    let version = read_u32(input, 8);
    if version != SHARED_GEMM_PROTOCOL_VERSION {
        return Err(SharedGemmProtocolError::UnsupportedVersion {
            actual: version,
            expected: SHARED_GEMM_PROTOCOL_VERSION,
        });
    }
    let header_len = read_u32(input, 12);
    if header_len as usize != SHARED_GEMM_HEADER_LEN {
        return Err(SharedGemmProtocolError::InvalidHeaderLength {
            actual: header_len,
            expected: SHARED_GEMM_HEADER_LEN,
        });
    }
    let resource_count = read_u32(input, 52);
    if resource_count != SHARED_GEMM_RESOURCE_COUNT as u32 {
        return Err(SharedGemmProtocolError::LengthMismatch {
            expected: SHARED_GEMM_RESOURCE_COUNT,
            actual: resource_count as usize,
        });
    }
    let control = SharedGemmControl {
        m: read_u64(input, 16),
        n: read_u64(input, 24),
        k: read_u64(input, 32),
        alpha: read_f32(input, 40),
        beta: read_f32(input, 44),
        hip_device_index: read_i32(input, 48),
    };
    validate_control(control)?;
    Ok(control)
}

pub fn expected_resource_lengths(
    control: SharedGemmControl,
) -> Result<SharedGemmResourceLengths, SharedGemmProtocolError> {
    validate_control(control)?;
    Ok(SharedGemmResourceLengths {
        a_bytes: matrix_bytes(control.m, control.k, "matrix A")?,
        b_bytes: matrix_bytes(control.k, control.n, "matrix B")?,
        c_bytes: matrix_bytes(control.m, control.n, "matrix C")?,
    })
}

fn validate_control(control: SharedGemmControl) -> Result<(), SharedGemmProtocolError> {
    if control.hip_device_index < 0 {
        return Err(SharedGemmProtocolError::InvalidDeviceIndex(
            control.hip_device_index,
        ));
    }
    let _ = expected_elements(control.m, control.k, "matrix A")?;
    let _ = expected_elements(control.k, control.n, "matrix B")?;
    let _ = expected_elements(control.m, control.n, "matrix C")?;
    Ok(())
}

fn expected_elements(
    rows: u64,
    columns: u64,
    label: &'static str,
) -> Result<u64, SharedGemmProtocolError> {
    rows.checked_mul(columns)
        .ok_or(SharedGemmProtocolError::DimensionOverflow(label))
}

fn matrix_bytes(
    rows: u64,
    columns: u64,
    label: &'static str,
) -> Result<u64, SharedGemmProtocolError> {
    expected_elements(rows, columns, label)?
        .checked_mul(F32_BYTES)
        .ok_or(SharedGemmProtocolError::ByteLengthOverflow(label))
}

fn read_u32(input: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(input[offset..offset + 4].try_into().expect("validated header"))
}

fn read_i32(input: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(input[offset..offset + 4].try_into().expect("validated header"))
}

fn read_u64(input: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(input[offset..offset + 8].try_into().expect("validated header"))
}

fn read_f32(input: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(input[offset..offset + 4].try_into().expect("validated header"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_round_trip_preserves_shape_scalars_and_device() {
        let control = SharedGemmControl {
            m: 2,
            n: 4,
            k: 3,
            alpha: 0.5,
            beta: 2.0,
            hip_device_index: 1,
        };
        let encoded = encode_control(control).expect("valid control should encode");
        assert_eq!(encoded.len(), SHARED_GEMM_HEADER_LEN);
        assert_eq!(decode_control(&encoded), Ok(control));
    }

    #[test]
    fn resource_lengths_match_row_major_fp32_matrices() {
        let lengths = expected_resource_lengths(SharedGemmControl {
            m: 2,
            n: 4,
            k: 3,
            alpha: 1.0,
            beta: 0.0,
            hip_device_index: 0,
        })
        .expect("valid dimensions should produce lengths");
        assert_eq!(lengths.a_bytes, 24);
        assert_eq!(lengths.b_bytes, 48);
        assert_eq!(lengths.c_bytes, 32);
    }

    #[test]
    fn negative_device_index_is_rejected() {
        let error = encode_control(SharedGemmControl {
            m: 1,
            n: 1,
            k: 1,
            alpha: 1.0,
            beta: 0.0,
            hip_device_index: -1,
        })
        .expect_err("negative HIP device must fail");
        assert_eq!(error, SharedGemmProtocolError::InvalidDeviceIndex(-1));
    }
}
