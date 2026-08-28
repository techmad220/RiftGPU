use thiserror::Error;

pub const GEMM_PROTOCOL_VERSION: u32 = 1;
pub const GEMM_REQUEST_MAGIC: [u8; 8] = *b"VRBGEMM1";
pub const GEMM_RESPONSE_MAGIC: [u8; 8] = *b"VRBGEMR1";
pub const GEMM_REQUEST_HEADER_LEN: usize = 56;
pub const GEMM_RESPONSE_HEADER_LEN: usize = 32;
pub const GEMM_FLAG_HAS_C: u32 = 1 << 0;
const GEMM_KNOWN_FLAGS: u32 = GEMM_FLAG_HAS_C;
const F32_BYTES: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GemmRequest<'a> {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub alpha: f32,
    pub beta: f32,
    pub a: &'a [f32],
    pub b: &'a [f32],
    pub c: Option<&'a [f32]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GemmRequestMeta {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub alpha: f32,
    pub beta: f32,
    pub has_c: bool,
    pub a_elements: u64,
    pub b_elements: u64,
    pub c_elements: u64,
    pub encoded_len: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedGemmRequest {
    pub meta: GemmRequestMeta,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedGemmResponse {
    pub m: u64,
    pub n: u64,
    pub values: Vec<f32>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GemmProtocolError {
    #[error("GEMM message is truncated: need at least {minimum} bytes, got {actual}")]
    Truncated { minimum: usize, actual: usize },
    #[error("invalid GEMM message magic")]
    InvalidMagic,
    #[error("unsupported GEMM protocol version {actual}; expected {expected}")]
    UnsupportedVersion { actual: u32, expected: u32 },
    #[error("invalid GEMM header length {actual}; expected {expected}")]
    InvalidHeaderLength { actual: u32, expected: usize },
    #[error("unsupported GEMM request flags 0x{0:08x}")]
    UnsupportedFlags(u32),
    #[error("GEMM dimensions overflow while computing {0}")]
    DimensionOverflow(&'static str),
    #[error("GEMM message length overflow")]
    LengthOverflow,
    #[error("GEMM message length {actual} does not match required length {expected}")]
    LengthMismatch { expected: u64, actual: u64 },
    #[error("matrix {matrix} contains {actual} elements; expected {expected}")]
    MatrixLengthMismatch {
        matrix: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error("GEMM message cannot be represented in this process address space")]
    AddressSpaceOverflow,
}

pub fn encoded_response_len(m: u64, n: u64) -> Result<u64, GemmProtocolError> {
    let elements = checked_elements(m, n, "response matrix")?;
    let payload = elements
        .checked_mul(F32_BYTES)
        .ok_or(GemmProtocolError::LengthOverflow)?;
    (GEMM_RESPONSE_HEADER_LEN as u64)
        .checked_add(payload)
        .ok_or(GemmProtocolError::LengthOverflow)
}

pub fn inspect_request(input: &[u8]) -> Result<GemmRequestMeta, GemmProtocolError> {
    require_len(input, GEMM_REQUEST_HEADER_LEN)?;
    if input[..8] != GEMM_REQUEST_MAGIC {
        return Err(GemmProtocolError::InvalidMagic);
    }

    let version = read_u32(input, 8)?;
    if version != GEMM_PROTOCOL_VERSION {
        return Err(GemmProtocolError::UnsupportedVersion {
            actual: version,
            expected: GEMM_PROTOCOL_VERSION,
        });
    }

    let header_len = read_u32(input, 12)?;
    if header_len as usize != GEMM_REQUEST_HEADER_LEN {
        return Err(GemmProtocolError::InvalidHeaderLength {
            actual: header_len,
            expected: GEMM_REQUEST_HEADER_LEN,
        });
    }

    let flags = read_u32(input, 16)?;
    if flags & !GEMM_KNOWN_FLAGS != 0 {
        return Err(GemmProtocolError::UnsupportedFlags(flags));
    }

    let m = read_u64(input, 24)?;
    let n = read_u64(input, 32)?;
    let k = read_u64(input, 40)?;
    let alpha = read_f32(input, 48)?;
    let beta = read_f32(input, 52)?;
    let has_c = flags & GEMM_FLAG_HAS_C != 0;

    let a_elements = checked_elements(m, k, "matrix A")?;
    let b_elements = checked_elements(k, n, "matrix B")?;
    let c_elements = checked_elements(m, n, "matrix C")?;
    let payload_elements = a_elements
        .checked_add(b_elements)
        .and_then(|value| {
            if has_c {
                value.checked_add(c_elements)
            } else {
                Some(value)
            }
        })
        .ok_or(GemmProtocolError::LengthOverflow)?;
    let payload_bytes = payload_elements
        .checked_mul(F32_BYTES)
        .ok_or(GemmProtocolError::LengthOverflow)?;
    let encoded_len = (GEMM_REQUEST_HEADER_LEN as u64)
        .checked_add(payload_bytes)
        .ok_or(GemmProtocolError::LengthOverflow)?;
    let actual_len =
        u64::try_from(input.len()).map_err(|_| GemmProtocolError::AddressSpaceOverflow)?;
    if actual_len != encoded_len {
        return Err(GemmProtocolError::LengthMismatch {
            expected: encoded_len,
            actual: actual_len,
        });
    }

    Ok(GemmRequestMeta {
        m,
        n,
        k,
        alpha,
        beta,
        has_c,
        a_elements,
        b_elements,
        c_elements,
        encoded_len,
    })
}

pub fn decode_request(input: &[u8]) -> Result<DecodedGemmRequest, GemmProtocolError> {
    let meta = inspect_request(input)?;
    let mut cursor = GEMM_REQUEST_HEADER_LEN;
    let a = read_f32_values(input, &mut cursor, meta.a_elements)?;
    let b = read_f32_values(input, &mut cursor, meta.b_elements)?;
    let c = if meta.has_c {
        Some(read_f32_values(input, &mut cursor, meta.c_elements)?)
    } else {
        None
    };

    Ok(DecodedGemmRequest { meta, a, b, c })
}

pub fn encode_request(request: GemmRequest<'_>) -> Result<Vec<u8>, GemmProtocolError> {
    let a_elements = checked_elements(request.m, request.k, "matrix A")?;
    let b_elements = checked_elements(request.k, request.n, "matrix B")?;
    let c_elements = checked_elements(request.m, request.n, "matrix C")?;
    validate_slice_len("A", a_elements, request.a.len())?;
    validate_slice_len("B", b_elements, request.b.len())?;
    if let Some(c) = request.c {
        validate_slice_len("C", c_elements, c.len())?;
    }

    let payload_elements = a_elements
        .checked_add(b_elements)
        .and_then(|value| {
            if request.c.is_some() {
                value.checked_add(c_elements)
            } else {
                Some(value)
            }
        })
        .ok_or(GemmProtocolError::LengthOverflow)?;
    let total_len = (GEMM_REQUEST_HEADER_LEN as u64)
        .checked_add(
            payload_elements
                .checked_mul(F32_BYTES)
                .ok_or(GemmProtocolError::LengthOverflow)?,
        )
        .ok_or(GemmProtocolError::LengthOverflow)?;
    let capacity =
        usize::try_from(total_len).map_err(|_| GemmProtocolError::AddressSpaceOverflow)?;

    let mut output = Vec::with_capacity(capacity);
    output.extend_from_slice(&GEMM_REQUEST_MAGIC);
    output.extend_from_slice(&GEMM_PROTOCOL_VERSION.to_le_bytes());
    output.extend_from_slice(&(GEMM_REQUEST_HEADER_LEN as u32).to_le_bytes());
    let flags = if request.c.is_some() {
        GEMM_FLAG_HAS_C
    } else {
        0
    };
    output.extend_from_slice(&flags.to_le_bytes());
    output.extend_from_slice(&0_u32.to_le_bytes());
    output.extend_from_slice(&request.m.to_le_bytes());
    output.extend_from_slice(&request.n.to_le_bytes());
    output.extend_from_slice(&request.k.to_le_bytes());
    output.extend_from_slice(&request.alpha.to_le_bytes());
    output.extend_from_slice(&request.beta.to_le_bytes());
    write_f32_values(&mut output, request.a);
    write_f32_values(&mut output, request.b);
    if let Some(c) = request.c {
        write_f32_values(&mut output, c);
    }
    debug_assert_eq!(output.len(), capacity);
    Ok(output)
}

pub fn encode_response(m: u64, n: u64, values: &[f32]) -> Result<Vec<u8>, GemmProtocolError> {
    let expected_elements = checked_elements(m, n, "response matrix")?;
    validate_slice_len("response", expected_elements, values.len())?;
    let total_len = encoded_response_len(m, n)?;
    let capacity =
        usize::try_from(total_len).map_err(|_| GemmProtocolError::AddressSpaceOverflow)?;

    let mut output = Vec::with_capacity(capacity);
    output.extend_from_slice(&GEMM_RESPONSE_MAGIC);
    output.extend_from_slice(&GEMM_PROTOCOL_VERSION.to_le_bytes());
    output.extend_from_slice(&(GEMM_RESPONSE_HEADER_LEN as u32).to_le_bytes());
    output.extend_from_slice(&m.to_le_bytes());
    output.extend_from_slice(&n.to_le_bytes());
    write_f32_values(&mut output, values);
    debug_assert_eq!(output.len(), capacity);
    Ok(output)
}

pub fn decode_response(input: &[u8]) -> Result<DecodedGemmResponse, GemmProtocolError> {
    require_len(input, GEMM_RESPONSE_HEADER_LEN)?;
    if input[..8] != GEMM_RESPONSE_MAGIC {
        return Err(GemmProtocolError::InvalidMagic);
    }

    let version = read_u32(input, 8)?;
    if version != GEMM_PROTOCOL_VERSION {
        return Err(GemmProtocolError::UnsupportedVersion {
            actual: version,
            expected: GEMM_PROTOCOL_VERSION,
        });
    }
    let header_len = read_u32(input, 12)?;
    if header_len as usize != GEMM_RESPONSE_HEADER_LEN {
        return Err(GemmProtocolError::InvalidHeaderLength {
            actual: header_len,
            expected: GEMM_RESPONSE_HEADER_LEN,
        });
    }

    let m = read_u64(input, 16)?;
    let n = read_u64(input, 24)?;
    let expected_len = encoded_response_len(m, n)?;
    let actual_len =
        u64::try_from(input.len()).map_err(|_| GemmProtocolError::AddressSpaceOverflow)?;
    if expected_len != actual_len {
        return Err(GemmProtocolError::LengthMismatch {
            expected: expected_len,
            actual: actual_len,
        });
    }

    let mut cursor = GEMM_RESPONSE_HEADER_LEN;
    let values = read_f32_values(
        input,
        &mut cursor,
        checked_elements(m, n, "response matrix")?,
    )?;
    Ok(DecodedGemmResponse { m, n, values })
}

fn checked_elements(
    rows: u64,
    columns: u64,
    label: &'static str,
) -> Result<u64, GemmProtocolError> {
    rows.checked_mul(columns)
        .ok_or(GemmProtocolError::DimensionOverflow(label))
}

fn validate_slice_len(
    matrix: &'static str,
    expected: u64,
    actual: usize,
) -> Result<(), GemmProtocolError> {
    let actual = u64::try_from(actual).map_err(|_| GemmProtocolError::AddressSpaceOverflow)?;
    if actual != expected {
        return Err(GemmProtocolError::MatrixLengthMismatch {
            matrix,
            expected,
            actual,
        });
    }
    Ok(())
}

fn require_len(input: &[u8], minimum: usize) -> Result<(), GemmProtocolError> {
    if input.len() < minimum {
        return Err(GemmProtocolError::Truncated {
            minimum,
            actual: input.len(),
        });
    }
    Ok(())
}

fn read_u32(input: &[u8], offset: usize) -> Result<u32, GemmProtocolError> {
    let bytes = read_array::<4>(input, offset)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(input: &[u8], offset: usize) -> Result<u64, GemmProtocolError> {
    let bytes = read_array::<8>(input, offset)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(input: &[u8], offset: usize) -> Result<f32, GemmProtocolError> {
    let bytes = read_array::<4>(input, offset)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_array<const N: usize>(input: &[u8], offset: usize) -> Result<[u8; N], GemmProtocolError> {
    let end = offset
        .checked_add(N)
        .ok_or(GemmProtocolError::LengthOverflow)?;
    let slice = input.get(offset..end).ok_or(GemmProtocolError::Truncated {
        minimum: end,
        actual: input.len(),
    })?;
    let mut output = [0_u8; N];
    output.copy_from_slice(slice);
    Ok(output)
}

fn read_f32_values(
    input: &[u8],
    cursor: &mut usize,
    count: u64,
) -> Result<Vec<f32>, GemmProtocolError> {
    let count = usize::try_from(count).map_err(|_| GemmProtocolError::AddressSpaceOverflow)?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(read_f32(input, *cursor)?);
        *cursor = (*cursor)
            .checked_add(4)
            .ok_or(GemmProtocolError::LengthOverflow)?;
    }
    Ok(values)
}

fn write_f32_values(output: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trip_preserves_gemm_inputs() {
        let encoded = encode_request(GemmRequest {
            m: 2,
            n: 2,
            k: 3,
            alpha: 0.5,
            beta: 2.0,
            a: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            b: &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            c: Some(&[1.0, 2.0, 3.0, 4.0]),
        })
        .expect("valid request should encode");
        let decoded = decode_request(&encoded).expect("encoded request should decode");

        assert_eq!(decoded.meta.m, 2);
        assert_eq!(decoded.meta.n, 2);
        assert_eq!(decoded.meta.k, 3);
        assert_eq!(decoded.meta.alpha, 0.5);
        assert_eq!(decoded.meta.beta, 2.0);
        assert_eq!(decoded.a, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(decoded.b, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        assert_eq!(decoded.c.as_deref(), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    }

    #[test]
    fn response_round_trip_preserves_shape_and_values() {
        let encoded = encode_response(2, 2, &[58.0, 64.0, 139.0, 154.0])
            .expect("valid response should encode");
        let decoded = decode_response(&encoded).expect("encoded response should decode");
        assert_eq!((decoded.m, decoded.n), (2, 2));
        assert_eq!(decoded.values, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn malformed_dimensions_are_rejected_without_wrapping() {
        let error = encode_request(GemmRequest {
            m: u64::MAX,
            n: 1,
            k: 2,
            alpha: 1.0,
            beta: 0.0,
            a: &[],
            b: &[],
            c: None,
        })
        .expect_err("overflowing matrix dimensions must fail");
        assert_eq!(error, GemmProtocolError::DimensionOverflow("matrix A"));
    }
}
