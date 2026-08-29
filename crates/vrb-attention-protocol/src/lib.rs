#![forbid(unsafe_code)]

//! Versioned control protocol for shared-resource scaled dot-product attention.
//!
//! Tensor payloads never appear in this message. The fixed-size header only
//! describes shape, datatype, causal behavior, scale, and target device.

use thiserror::Error;

pub const ATTENTION_MAGIC: u32 = 0x4142_5256; // "VRBA" in little-endian bytes.
pub const ATTENTION_PROTOCOL_VERSION: u16 = 1;
pub const ATTENTION_HEADER_LEN: usize = 64;
pub const ATTENTION_RESOURCE_COUNT: u32 = 4;
pub const Q_RESOURCE_INDEX: usize = 0;
pub const K_RESOURCE_INDEX: usize = 1;
pub const V_RESOURCE_INDEX: usize = 2;
pub const O_RESOURCE_INDEX: usize = 3;
pub const FLAG_CAUSAL: u32 = 1;
const KNOWN_FLAGS: u32 = FLAG_CAUSAL;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum AttentionDataType {
    #[default]
    F32 = 1,
    F16 = 2,
    Bf16 = 3,
}

impl AttentionDataType {
    #[must_use]
    pub const fn element_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    fn from_raw(value: u32) -> Result<Self, AttentionProtocolError> {
        match value {
            1 => Ok(Self::F32),
            2 => Ok(Self::F16),
            3 => Ok(Self::Bf16),
            other => Err(AttentionProtocolError::UnsupportedDataType(other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AttentionControl {
    pub flags: u32,
    pub data_type: AttentionDataType,
    pub batch: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub query_len: u32,
    pub kv_len: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub device_index: i32,
}

impl AttentionControl {
    #[must_use]
    pub const fn is_causal(self) -> bool {
        self.flags & FLAG_CAUSAL != 0
    }

    pub fn validate(self) -> Result<(), AttentionProtocolError> {
        if self.flags & !KNOWN_FLAGS != 0 {
            return Err(AttentionProtocolError::UnsupportedFlags(self.flags));
        }
        if self.batch == 0
            || self.query_heads == 0
            || self.kv_heads == 0
            || self.query_len == 0
            || self.kv_len == 0
            || self.head_dim == 0
        {
            return Err(AttentionProtocolError::ZeroDimension);
        }
        if self.kv_heads > self.query_heads || !self.query_heads.is_multiple_of(self.kv_heads) {
            return Err(AttentionProtocolError::InvalidHeadGrouping {
                query_heads: self.query_heads,
                kv_heads: self.kv_heads,
            });
        }
        if self.is_causal() && self.kv_len < self.query_len {
            return Err(AttentionProtocolError::InvalidCausalLengths {
                query_len: self.query_len,
                kv_len: self.kv_len,
            });
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(AttentionProtocolError::InvalidScale(self.scale));
        }
        if self.device_index < 0 {
            return Err(AttentionProtocolError::InvalidDeviceIndex(
                self.device_index,
            ));
        }
        expected_resource_lengths(self)?;
        Ok(())
    }
}

pub fn encode_control(
    control: AttentionControl,
) -> Result<[u8; ATTENTION_HEADER_LEN], AttentionProtocolError> {
    control.validate()?;
    let mut output = [0_u8; ATTENTION_HEADER_LEN];
    write_u32(&mut output, 0, ATTENTION_MAGIC);
    write_u16(&mut output, 4, ATTENTION_PROTOCOL_VERSION);
    write_u16(
        &mut output,
        6,
        u16::try_from(ATTENTION_HEADER_LEN).expect("attention header length fits u16"),
    );
    write_u32(&mut output, 8, control.flags);
    write_u32(&mut output, 12, control.data_type as u32);
    write_u32(&mut output, 16, control.batch);
    write_u32(&mut output, 20, control.query_heads);
    write_u32(&mut output, 24, control.kv_heads);
    write_u32(&mut output, 28, control.query_len);
    write_u32(&mut output, 32, control.kv_len);
    write_u32(&mut output, 36, control.head_dim);
    write_u32(&mut output, 40, control.scale.to_bits());
    write_u32(&mut output, 44, control.device_index as u32);
    write_u32(&mut output, 48, ATTENTION_RESOURCE_COUNT);
    write_u32(&mut output, 52, 0);
    write_u64(&mut output, 56, 0);
    Ok(output)
}

pub fn decode_control(input: &[u8]) -> Result<AttentionControl, AttentionProtocolError> {
    if input.len() != ATTENTION_HEADER_LEN {
        return Err(AttentionProtocolError::InvalidLength {
            expected: ATTENTION_HEADER_LEN,
            actual: input.len(),
        });
    }
    if read_u32(input, 0) != ATTENTION_MAGIC {
        return Err(AttentionProtocolError::InvalidMagic);
    }
    let version = read_u16(input, 4);
    if version != ATTENTION_PROTOCOL_VERSION {
        return Err(AttentionProtocolError::UnsupportedVersion(version));
    }
    if usize::from(read_u16(input, 6)) != ATTENTION_HEADER_LEN {
        return Err(AttentionProtocolError::InvalidHeaderLength(read_u16(
            input, 6,
        )));
    }
    let resource_count = read_u32(input, 48);
    if resource_count != ATTENTION_RESOURCE_COUNT {
        return Err(AttentionProtocolError::InvalidResourceCount(resource_count));
    }
    if read_u32(input, 52) != 0 || read_u64(input, 56) != 0 {
        return Err(AttentionProtocolError::ReservedFieldNonZero);
    }

    let control = AttentionControl {
        flags: read_u32(input, 8),
        data_type: AttentionDataType::from_raw(read_u32(input, 12))?,
        batch: read_u32(input, 16),
        query_heads: read_u32(input, 20),
        kv_heads: read_u32(input, 24),
        query_len: read_u32(input, 28),
        kv_len: read_u32(input, 32),
        head_dim: read_u32(input, 36),
        scale: f32::from_bits(read_u32(input, 40)),
        device_index: read_u32(input, 44) as i32,
    };
    control.validate()?;
    Ok(control)
}

pub fn expected_resource_lengths(
    control: AttentionControl,
) -> Result<[u64; ATTENTION_RESOURCE_COUNT as usize], AttentionProtocolError> {
    let width = control.data_type.element_bytes();
    let q_elements = checked_product(&[
        u64::from(control.batch),
        u64::from(control.query_heads),
        u64::from(control.query_len),
        u64::from(control.head_dim),
    ])?;
    let kv_elements = checked_product(&[
        u64::from(control.batch),
        u64::from(control.kv_heads),
        u64::from(control.kv_len),
        u64::from(control.head_dim),
    ])?;
    let q_bytes = q_elements
        .checked_mul(width)
        .ok_or(AttentionProtocolError::SizeOverflow)?;
    let kv_bytes = kv_elements
        .checked_mul(width)
        .ok_or(AttentionProtocolError::SizeOverflow)?;
    Ok([q_bytes, kv_bytes, kv_bytes, q_bytes])
}

fn checked_product(values: &[u64]) -> Result<u64, AttentionProtocolError> {
    values.iter().try_fold(1_u64, |acc, value| {
        acc.checked_mul(*value)
            .ok_or(AttentionProtocolError::SizeOverflow)
    })
}

fn write_u16(output: &mut [u8], offset: usize, value: u16) {
    output[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(output: &mut [u8], offset: usize, value: u32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(output: &mut [u8], offset: usize, value: u64) {
    output[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_u16(input: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([input[offset], input[offset + 1]])
}

fn read_u32(input: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        input[offset],
        input[offset + 1],
        input[offset + 2],
        input[offset + 3],
    ])
}

fn read_u64(input: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        input[offset],
        input[offset + 1],
        input[offset + 2],
        input[offset + 3],
        input[offset + 4],
        input[offset + 5],
        input[offset + 6],
        input[offset + 7],
    ])
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum AttentionProtocolError {
    #[error("attention control length mismatch: expected {expected}, got {actual}")]
    InvalidLength { expected: usize, actual: usize },
    #[error("attention control magic is invalid")]
    InvalidMagic,
    #[error("unsupported attention protocol version {0}")]
    UnsupportedVersion(u16),
    #[error("invalid attention header length {0}")]
    InvalidHeaderLength(u16),
    #[error("unsupported attention flags 0x{0:08x}")]
    UnsupportedFlags(u32),
    #[error("unsupported attention datatype tag {0}")]
    UnsupportedDataType(u32),
    #[error("attention dimensions must be non-zero")]
    ZeroDimension,
    #[error("invalid attention head grouping: query_heads={query_heads}, kv_heads={kv_heads}")]
    InvalidHeadGrouping { query_heads: u32, kv_heads: u32 },
    #[error(
        "causal attention requires kv_len >= query_len: query_len={query_len}, kv_len={kv_len}"
    )]
    InvalidCausalLengths { query_len: u32, kv_len: u32 },
    #[error("attention scale must be finite and positive, got {0}")]
    InvalidScale(f32),
    #[error("attention device index must be non-negative, got {0}")]
    InvalidDeviceIndex(i32),
    #[error("invalid attention resource count {0}")]
    InvalidResourceCount(u32),
    #[error("reserved attention header field is non-zero")]
    ReservedFieldNonZero,
    #[error("attention tensor size arithmetic overflow")]
    SizeOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control() -> AttentionControl {
        AttentionControl {
            flags: FLAG_CAUSAL,
            data_type: AttentionDataType::F16,
            batch: 2,
            query_heads: 8,
            kv_heads: 2,
            query_len: 16,
            kv_len: 32,
            head_dim: 64,
            scale: 0.125,
            device_index: 0,
        }
    }

    #[test]
    fn control_round_trip_is_exact() {
        let value = control();
        let encoded = encode_control(value).unwrap();
        assert_eq!(encoded.len(), ATTENTION_HEADER_LEN);
        assert_eq!(decode_control(&encoded).unwrap(), value);
    }

    #[test]
    fn resource_lengths_account_for_grouped_query_heads() {
        let lengths = expected_resource_lengths(control()).unwrap();
        assert_eq!(lengths[Q_RESOURCE_INDEX], 2 * 8 * 16 * 64 * 2);
        assert_eq!(lengths[K_RESOURCE_INDEX], 2 * 2 * 32 * 64 * 2);
        assert_eq!(lengths[V_RESOURCE_INDEX], lengths[K_RESOURCE_INDEX]);
        assert_eq!(lengths[O_RESOURCE_INDEX], lengths[Q_RESOURCE_INDEX]);
    }

    #[test]
    fn malformed_head_grouping_is_rejected() {
        let invalid = AttentionControl {
            query_heads: 7,
            kv_heads: 2,
            ..control()
        };
        assert!(matches!(
            invalid.validate(),
            Err(AttentionProtocolError::InvalidHeadGrouping { .. })
        ));
    }

    #[test]
    fn reserved_bytes_fail_closed() {
        let mut encoded = encode_control(control()).unwrap();
        encoded[63] = 1;
        assert_eq!(
            decode_control(&encoded),
            Err(AttentionProtocolError::ReservedFieldNonZero)
        );
    }
}
