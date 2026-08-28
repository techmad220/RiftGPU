use std::mem::size_of;
use vrb_plugin_api::{
    capability, expected_backend_info_struct_size, expected_plugin_struct_size, VrbBackendInfoV1,
    VrbBackendKind, VrbExecutionRequestV1, VrbPluginV1, VrbStatus, VRB_PLUGIN_ABI_VERSION,
    VRB_PLUGIN_ENTRY_SYMBOL,
};

#[test]
fn version_symbol_and_status_values_are_stable() {
    assert_eq!(VRB_PLUGIN_ABI_VERSION, 1);
    assert_eq!(VRB_PLUGIN_ENTRY_SYMBOL, b"vrb_plugin_entry_v1\0");
    assert_eq!(VrbStatus::Ok as i32, 0);
    assert_eq!(VrbStatus::InvalidArgument as i32, 1);
    assert_eq!(VrbStatus::Unsupported as i32, 2);
    assert_eq!(VrbStatus::Unavailable as i32, 3);
    assert_eq!(VrbStatus::InternalError as i32, 4);
    assert_eq!(VrbBackendKind::Cpu as u32, 1);
    assert_eq!(VrbBackendKind::Vulkan as u32, 2);
    assert_eq!(VrbBackendKind::Hip as u32, 3);
    assert_eq!(VrbBackendKind::Hybrid as u32, 4);
    assert_eq!(VrbBackendKind::Other as u32, 255);
}

#[test]
fn capability_bits_are_unique_and_non_overlapping() {
    let bits = [
        capability::COMPUTE,
        capability::EXTERNAL_MEMORY,
        capability::EXTERNAL_SEMAPHORE,
        capability::ZERO_COPY,
        capability::TIMESTAMPS,
        capability::FP16,
        capability::FP32,
        capability::INT8,
    ];
    let mut union = 0_u64;
    for bit in bits {
        assert_ne!(bit, 0);
        assert_eq!(bit.count_ones(), 1);
        assert_eq!(union & bit, 0);
        union |= bit;
    }
}

#[test]
fn abi_struct_defaults_and_size_helpers_match_real_layout() {
    let info = VrbBackendInfoV1::default();
    assert_eq!(info.struct_size, size_of::<VrbBackendInfoV1>() as u32);
    assert_eq!(info.backend_kind, VrbBackendKind::Other);
    assert_eq!(info.capability_bits, 0);
    assert_eq!(info.device_count, 0);
    assert!(info.name.iter().all(|value| *value == 0));
    assert!(info.vendor.iter().all(|value| *value == 0));
    assert_eq!(
        expected_backend_info_struct_size(),
        size_of::<VrbBackendInfoV1>() as u32
    );
    assert_eq!(
        expected_plugin_struct_size(),
        size_of::<VrbPluginV1>() as u32
    );

    let request = VrbExecutionRequestV1::default();
    assert_eq!(request.struct_size, 0);
    assert_eq!(request.operation, 0);
    assert_eq!(request.element_count, 0);
}
