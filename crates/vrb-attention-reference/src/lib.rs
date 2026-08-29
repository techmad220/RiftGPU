#![forbid(unsafe_code)]

//! CPU correctness oracle for FP32 scaled dot-product attention.

use thiserror::Error;
use vrb_attention_protocol::{
    expected_resource_lengths, AttentionControl, AttentionDataType, AttentionProtocolError,
    K_RESOURCE_INDEX, O_RESOURCE_INDEX, Q_RESOURCE_INDEX, V_RESOURCE_INDEX,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionLimits {
    pub max_tensor_elements: u64,
    pub max_score_elements: u64,
}

impl Default for AttentionLimits {
    fn default() -> Self {
        Self {
            max_tensor_elements: 64 * 1024 * 1024,
            max_score_elements: 32 * 1024 * 1024,
        }
    }
}

pub fn execute_attention_f32(
    control: AttentionControl,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    limits: AttentionLimits,
) -> Result<Vec<f32>, ReferenceAttentionError> {
    control.validate()?;
    if control.data_type != AttentionDataType::F32 {
        return Err(ReferenceAttentionError::UnsupportedDataType(
            control.data_type,
        ));
    }

    let lengths = expected_resource_lengths(control)?;
    let expected_q = elements_from_f32_bytes(lengths[Q_RESOURCE_INDEX])?;
    let expected_k = elements_from_f32_bytes(lengths[K_RESOURCE_INDEX])?;
    let expected_v = elements_from_f32_bytes(lengths[V_RESOURCE_INDEX])?;
    let expected_o = elements_from_f32_bytes(lengths[O_RESOURCE_INDEX])?;

    check_input_len("q", q.len(), expected_q)?;
    check_input_len("k", k.len(), expected_k)?;
    check_input_len("v", v.len(), expected_v)?;

    let output_elements =
        u64::try_from(expected_o).map_err(|_| ReferenceAttentionError::AddressSpaceOverflow)?;
    if output_elements > limits.max_tensor_elements {
        return Err(ReferenceAttentionError::ResourceLimit {
            what: "output tensor elements",
            requested: output_elements,
            limit: limits.max_tensor_elements,
        });
    }

    let score_elements = checked_product(&[
        u64::from(control.batch),
        u64::from(control.query_heads),
        u64::from(control.query_len),
        u64::from(control.kv_len),
    ])?;
    if score_elements > limits.max_score_elements {
        return Err(ReferenceAttentionError::ResourceLimit {
            what: "attention score elements",
            requested: score_elements,
            limit: limits.max_score_elements,
        });
    }

    let batch = usize::try_from(control.batch)
        .map_err(|_| ReferenceAttentionError::AddressSpaceOverflow)?;
    let query_heads = usize::try_from(control.query_heads)
        .map_err(|_| ReferenceAttentionError::AddressSpaceOverflow)?;
    let kv_heads = usize::try_from(control.kv_heads)
        .map_err(|_| ReferenceAttentionError::AddressSpaceOverflow)?;
    let query_len = usize::try_from(control.query_len)
        .map_err(|_| ReferenceAttentionError::AddressSpaceOverflow)?;
    let kv_len = usize::try_from(control.kv_len)
        .map_err(|_| ReferenceAttentionError::AddressSpaceOverflow)?;
    let head_dim = usize::try_from(control.head_dim)
        .map_err(|_| ReferenceAttentionError::AddressSpaceOverflow)?;
    let heads_per_kv = query_heads / kv_heads;
    let causal_base = kv_len.saturating_sub(query_len);

    let mut output = vec![0.0_f32; expected_o];
    let mut scores = vec![0.0_f32; kv_len];

    for batch_index in 0..batch {
        for query_head in 0..query_heads {
            let kv_head = query_head / heads_per_kv;
            for query_position in 0..query_len {
                let max_key = if control.is_causal() {
                    causal_base + query_position
                } else {
                    kv_len - 1
                };

                let q_base = ((batch_index * query_heads + query_head) * query_len
                    + query_position)
                    * head_dim;
                let mut max_score = f32::NEG_INFINITY;

                for key_position in 0..=max_key {
                    let k_base =
                        ((batch_index * kv_heads + kv_head) * kv_len + key_position) * head_dim;
                    let mut dot = 0.0_f32;
                    for dim in 0..head_dim {
                        dot += q[q_base + dim] * k[k_base + dim];
                    }
                    let score = dot * control.scale;
                    scores[key_position] = score;
                    max_score = max_score.max(score);
                }

                let mut denominator = 0.0_f32;
                for score in &mut scores[..=max_key] {
                    *score = (*score - max_score).exp();
                    denominator += *score;
                }
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err(ReferenceAttentionError::NumericalFailure);
                }

                let o_base = ((batch_index * query_heads + query_head) * query_len
                    + query_position)
                    * head_dim;
                for key_position in 0..=max_key {
                    let weight = scores[key_position] / denominator;
                    let v_base =
                        ((batch_index * kv_heads + kv_head) * kv_len + key_position) * head_dim;
                    for dim in 0..head_dim {
                        output[o_base + dim] += weight * v[v_base + dim];
                    }
                }
            }
        }
    }

    Ok(output)
}

fn elements_from_f32_bytes(bytes: u64) -> Result<usize, ReferenceAttentionError> {
    let elements = bytes
        .checked_div(4)
        .ok_or(ReferenceAttentionError::AddressSpaceOverflow)?;
    usize::try_from(elements).map_err(|_| ReferenceAttentionError::AddressSpaceOverflow)
}

fn check_input_len(
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), ReferenceAttentionError> {
    if actual != expected {
        return Err(ReferenceAttentionError::InputLength {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

fn checked_product(values: &[u64]) -> Result<u64, ReferenceAttentionError> {
    values.iter().try_fold(1_u64, |acc, value| {
        acc.checked_mul(*value)
            .ok_or(ReferenceAttentionError::AddressSpaceOverflow)
    })
}

#[derive(Debug, Error, PartialEq)]
pub enum ReferenceAttentionError {
    #[error(transparent)]
    Protocol(#[from] AttentionProtocolError),
    #[error("CPU reference attention supports FP32 only, got {0:?}")]
    UnsupportedDataType(AttentionDataType),
    #[error("{name} input length mismatch: expected {expected}, got {actual}")]
    InputLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{what} exceeds configured limit: requested {requested}, limit {limit}")]
    ResourceLimit {
        what: &'static str,
        requested: u64,
        limit: u64,
    },
    #[error("attention tensor does not fit the host address space")]
    AddressSpaceOverflow,
    #[error("attention softmax produced a non-finite or zero denominator")]
    NumericalFailure,
}

#[cfg(test)]
mod tests {
    use super::*;
    use vrb_attention_protocol::{AttentionDataType, FLAG_CAUSAL};

    fn base_control() -> AttentionControl {
        AttentionControl {
            flags: 0,
            data_type: AttentionDataType::F32,
            batch: 1,
            query_heads: 1,
            kv_heads: 1,
            query_len: 1,
            kv_len: 2,
            head_dim: 1,
            scale: 1.0,
            device_index: 0,
        }
    }

    #[test]
    fn one_head_attention_matches_known_softmax() {
        let output = execute_attention_f32(
            base_control(),
            &[1.0],
            &[0.0, 1.0],
            &[2.0, 4.0],
            AttentionLimits::default(),
        )
        .unwrap();

        assert!((output[0] - 3.462_117).abs() < 1e-5);
    }

    #[test]
    fn causal_mask_respects_decode_offset() {
        let control = AttentionControl {
            flags: FLAG_CAUSAL,
            query_len: 2,
            kv_len: 2,
            ..base_control()
        };
        let output = execute_attention_f32(
            control,
            &[1.0, 1.0],
            &[0.0, 1.0],
            &[2.0, 4.0],
            AttentionLimits::default(),
        )
        .unwrap();

        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 3.462_117).abs() < 1e-5);
    }

    #[test]
    fn grouped_query_attention_maps_heads_to_kv_groups() {
        let control = AttentionControl {
            query_heads: 4,
            kv_heads: 2,
            kv_len: 1,
            ..base_control()
        };
        let output = execute_attention_f32(
            control,
            &[1.0, 1.0, 1.0, 1.0],
            &[1.0, 10.0],
            &[3.0, 7.0],
            AttentionLimits::default(),
        )
        .unwrap();

        assert_eq!(output, vec![3.0, 3.0, 7.0, 7.0]);
    }
}
