use vrb_core::{
    BackendError, BackendId, BackendKind, BackendProbe, CapabilitySet, ComputeBackend, DataType,
    OperationKind,
};

#[derive(Debug)]
pub struct CpuBackend {
    id: BackendId,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            id: BackendId::new("cpu").expect("static backend id is valid"),
        }
    }

    pub fn vector_add_f32(&self, left: &[f32], right: &[f32], output: &mut [f32]) -> Result<(), BackendError> {
        if left.len() != right.len() || left.len() != output.len() {
            return Err(BackendError::Internal(
                "vector_add_f32 requires equal input and output lengths".to_owned(),
            ));
        }

        for ((out, lhs), rhs) in output.iter_mut().zip(left.iter()).zip(right.iter()) {
            *out = *lhs + *rhs;
        }
        Ok(())
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn probe(&self) -> Result<BackendProbe, BackendError> {
        Ok(BackendProbe {
            id: self.id.clone(),
            kind: BackendKind::Cpu,
            name: "CPU reference backend".to_owned(),
            vendor: std::env::consts::ARCH.to_owned(),
            available: true,
            device_count: 1,
            detail: "Portable correctness/reference backend".to_owned(),
            capabilities: CapabilitySet {
                operations: vec![OperationKind::Copy, OperationKind::VectorAdd],
                data_types: vec![DataType::F32, DataType::I8],
                external_memory: false,
                external_semaphore: false,
                zero_copy: false,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_add_is_correct() {
        let backend = CpuBackend::new();
        let left = [1.0_f32, 2.5, -3.0];
        let right = [4.0_f32, -0.5, 8.0];
        let mut output = [0.0_f32; 3];
        backend.vector_add_f32(&left, &right, &mut output).unwrap();
        assert_eq!(output, [5.0, 2.0, 5.0]);
    }
}
