#![cfg(target_os = "windows")]

use std::ffi::c_void;
use vrb_backends::{run_zero_copy_smoke, HipBackend, VulkanBackend};
use vrb_core::{BackendError, ComputeBackend};

#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcess() -> *mut c_void;
    fn GetProcessHandleCount(process: *mut c_void, handle_count: *mut u32) -> i32;
}

fn process_handle_count() -> u32 {
    let mut count = 0_u32;
    let process = unsafe { GetCurrentProcess() };
    let ok = unsafe { GetProcessHandleCount(process, &mut count) };
    assert_ne!(ok, 0, "GetProcessHandleCount failed");
    count
}

#[test]
fn amd_vulkan_hip_bridge_full_hardware_e2e() {
    if std::env::var_os("VRB_REQUIRE_AMD_E2E").is_none() {
        return;
    }

    let vulkan = VulkanBackend::new();
    let vulkan_info = vulkan.runtime_info().expect("Vulkan runtime discovery must succeed");
    let preferred = vulkan_info
        .preferred_compute_device()
        .expect("Vulkan must expose a preferred compute device");
    assert_eq!(preferred.vendor_id, 0x1002, "certification requires an AMD Vulkan device");
    assert!(preferred.external_memory, "Vulkan external-memory support is required");
    assert!(preferred.external_semaphore, "Vulkan external-semaphore support is required");
    let vulkan_probe = vulkan.probe().expect("Vulkan probe must succeed");
    assert!(vulkan_probe.available);
    assert!(vulkan_probe.capabilities.external_memory);
    assert!(vulkan_probe.capabilities.external_semaphore);
    assert!(vulkan_probe.capabilities.zero_copy);

    let hip = HipBackend::new();
    let hip_info = hip.runtime_info().expect("HIP runtime discovery must succeed");
    assert!(!hip_info.devices.is_empty(), "HIP must expose at least one device");
    assert!(hip_info.external_memory_api, "HIP external-memory API is required");
    assert!(hip_info.external_semaphore_api, "HIP external-semaphore API is required");
    let hip_probe = hip.probe().expect("HIP probe must succeed");
    assert!(hip_probe.available);
    assert!(hip_probe.capabilities.external_memory);
    assert!(hip_probe.capabilities.external_semaphore);
    assert!(hip_probe.capabilities.zero_copy);

    let zero_error = run_zero_copy_smoke(0, 0).expect_err("zero-byte bridge request must fail");
    assert!(matches!(zero_error, BackendError::Internal(_)));

    // Warm up loader/runtime state before leak accounting so one-time driver caches
    // cannot be mistaken for a per-transfer resource leak.
    let warmup = run_zero_copy_smoke(1024 * 1024, 0x5a).expect("warmup bridge must succeed");
    assert_eq!(warmup.verified_bytes, 1024 * 1024);
    assert_eq!(warmup.vulkan_device, warmup.hip_device);

    let handles_before = process_handle_count();
    for iteration in 0_u8..32 {
        let bytes = match iteration % 4 {
            0 => 1 * 1024 * 1024,
            1 => 4 * 1024 * 1024,
            2 => 8 * 1024 * 1024,
            _ => 16 * 1024 * 1024,
        };
        let pattern = 0x20_u8.wrapping_add(iteration);
        let report = run_zero_copy_smoke(bytes, pattern).unwrap_or_else(|error| {
            panic!("bridge iteration {iteration} failed for {bytes} bytes: {error}")
        });
        assert_eq!(report.bytes, bytes);
        assert_eq!(report.verified_bytes, bytes);
        assert_eq!(report.pattern, pattern);
        assert_eq!(report.vulkan_device, report.hip_device);
        assert!(report.external_memory_handle.contains("WIN32"));
        assert!(!report.synchronization.is_empty());
    }
    let handles_after = process_handle_count();
    let growth = handles_after.saturating_sub(handles_before);
    assert!(
        growth <= 8,
        "process handles grew by {growth} across 32 bridge iterations ({handles_before} -> {handles_after})"
    );

    let large = run_zero_copy_smoke(64 * 1024 * 1024, 0xa5)
        .expect("final 64 MiB bridge must succeed after stress loop");
    assert_eq!(large.verified_bytes, 64 * 1024 * 1024);
    assert_eq!(large.vulkan_device, preferred.name);
    assert_eq!(large.hip_device, preferred.name);
}
