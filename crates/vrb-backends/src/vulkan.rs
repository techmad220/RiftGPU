use ash::{vk, Entry};
use std::collections::BTreeSet;
use std::ffi::{CStr, CString};
use vrb_core::{
    BackendError, BackendId, BackendKind, BackendProbe, CapabilitySet, ComputeBackend, DataType,
    OperationKind,
};

const AMD_PCI_VENDOR_ID: u32 = 0x1002;

#[derive(Debug, Clone)]
pub struct VulkanDeviceInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: u32,
    pub device_type: vk::PhysicalDeviceType,
    pub compute_queue: bool,
    pub external_memory: bool,
    pub external_semaphore: bool,
}

impl VulkanDeviceInfo {
    pub fn is_discrete(&self) -> bool {
        self.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
    }

    pub fn bridge_score(&self) -> (u8, u8, u8, u32) {
        (
            if self.vendor_id == AMD_PCI_VENDOR_ID { 0 } else { 1 },
            if self.is_discrete() { 0 } else { 1 },
            if self.external_memory && self.external_semaphore {
                0
            } else {
                1
            },
            self.device_id,
        )
    }
}

#[derive(Debug, Clone)]
pub struct VulkanRuntimeInfo {
    pub loader_available: bool,
    pub devices: Vec<VulkanDeviceInfo>,
}

impl VulkanRuntimeInfo {
    pub fn preferred_compute_device(&self) -> Option<&VulkanDeviceInfo> {
        self.devices
            .iter()
            .filter(|device| device.compute_queue)
            .min_by_key(|device| device.bridge_score())
    }
}

#[derive(Debug)]
pub struct VulkanBackend {
    id: BackendId,
}

impl VulkanBackend {
    pub fn new() -> Self {
        Self {
            id: BackendId::new("vulkan").expect("static backend id is valid"),
        }
    }

    pub fn runtime_info(&self) -> Result<VulkanRuntimeInfo, BackendError> {
        // SAFETY: ash loads the system Vulkan loader and resolves its published
        // entry points. No application-provided library path is accepted here.
        let entry = unsafe { Entry::load() }
            .map_err(|error| BackendError::Unavailable(format!("Vulkan loader: {error}")))?;

        let app_name = CString::new("vulkan-rocm-bridge")
            .map_err(|error| BackendError::Internal(error.to_string()))?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&app_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_1);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        // SAFETY: create_info only references app_info/app_name for the duration
        // of the call and contains no uninitialized extension chains.
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|error| BackendError::Probe(format!("vkCreateInstance: {error:?}")))?;

        let result = (|| {
            // SAFETY: instance is valid until the cleanup below.
            let physical_devices = unsafe { instance.enumerate_physical_devices() }
                .map_err(|error| BackendError::Probe(format!("enumerate devices: {error:?}")))?;

            let mut devices = Vec::with_capacity(physical_devices.len());
            for device in physical_devices {
                // SAFETY: device was returned by this instance.
                let properties = unsafe { instance.get_physical_device_properties(device) };
                // SAFETY: Vulkan guarantees a NUL-terminated device_name array.
                let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();

                // SAFETY: device was returned by this instance.
                let queue_families =
                    unsafe { instance.get_physical_device_queue_family_properties(device) };
                let compute_queue = queue_families.iter().any(|family| {
                    family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                });

                // SAFETY: device was returned by this instance.
                let extension_properties =
                    unsafe { instance.enumerate_device_extension_properties(device) }.map_err(
                        |error| {
                            BackendError::Probe(format!("enumerate device extensions: {error:?}"))
                        },
                    )?;

                let mut extensions = BTreeSet::new();
                for extension in extension_properties {
                    // SAFETY: Vulkan guarantees NUL termination for extension_name.
                    let extension_name =
                        unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) }
                            .to_string_lossy()
                            .into_owned();
                    extensions.insert(extension_name);
                }

                let (external_memory, external_semaphore) = external_resource_support(&extensions);
                devices.push(VulkanDeviceInfo {
                    name,
                    vendor_id: properties.vendor_id,
                    device_id: properties.device_id,
                    api_version: properties.api_version,
                    device_type: properties.device_type,
                    compute_queue,
                    external_memory,
                    external_semaphore,
                });
            }

            Ok(VulkanRuntimeInfo {
                loader_available: true,
                devices,
            })
        })();

        // SAFETY: no child Vulkan objects survive this probe.
        unsafe { instance.destroy_instance(None) };
        result
    }
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for VulkanBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Vulkan
    }

    fn probe(&self) -> Result<BackendProbe, BackendError> {
        let info = self.runtime_info()?;
        let compute_devices: Vec<&VulkanDeviceInfo> = info
            .devices
            .iter()
            .filter(|device| device.compute_queue)
            .collect();
        let preferred = info.preferred_compute_device();
        let external_memory = preferred.is_some_and(|device| device.external_memory);
        let external_semaphore = preferred.is_some_and(|device| device.external_semaphore);
        let inventory = compute_devices
            .iter()
            .map(|device| {
                format!(
                    "{}[vendor=0x{:04x},device=0x{:04x},type={:?}]",
                    device.name, device.vendor_id, device.device_id, device.device_type
                )
            })
            .collect::<Vec<_>>()
            .join("; ");

        Ok(BackendProbe {
            id: self.id.clone(),
            kind: BackendKind::Vulkan,
            name: preferred
                .map(|device| device.name.clone())
                .unwrap_or_else(|| "Vulkan".to_owned()),
            vendor: preferred
                .map(|device| format!("PCI vendor 0x{:04x}", device.vendor_id))
                .unwrap_or_default(),
            available: preferred.is_some(),
            device_count: compute_devices.len() as u32,
            detail: preferred
                .map(|device| {
                    format!(
                        "selected_device_id=0x{:04x}, selected_type={:?}, api_version={}, external_memory={}, external_semaphore={}; inventory={inventory}",
                        device.device_id,
                        device.device_type,
                        device.api_version,
                        device.external_memory,
                        device.external_semaphore
                    )
                })
                .unwrap_or_else(|| "No compute-capable Vulkan physical device".to_owned()),
            capabilities: CapabilitySet {
                // Like HIP, this built-in layer owns transport/device discovery.
                // Operator plugins advertise concrete optimized operations.
                operations: vec![OperationKind::Copy, OperationKind::Custom],
                data_types: vec![DataType::Unknown],
                external_memory,
                external_semaphore,
                zero_copy: external_memory && external_semaphore,
            },
        })
    }
}

fn external_resource_support(extensions: &BTreeSet<String>) -> (bool, bool) {
    #[cfg(target_os = "windows")]
    {
        (
            extensions.contains("VK_KHR_external_memory_win32"),
            extensions.contains("VK_KHR_external_semaphore_win32"),
        )
    }

    #[cfg(target_os = "linux")]
    {
        (
            extensions.contains("VK_KHR_external_memory_fd"),
            extensions.contains("VK_KHR_external_semaphore_fd"),
        )
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        let _ = extensions;
        (false, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(
        name: &str,
        vendor_id: u32,
        device_id: u32,
        device_type: vk::PhysicalDeviceType,
    ) -> VulkanDeviceInfo {
        VulkanDeviceInfo {
            name: name.to_owned(),
            vendor_id,
            device_id,
            api_version: vk::API_VERSION_1_3,
            device_type,
            compute_queue: true,
            external_memory: true,
            external_semaphore: true,
        }
    }

    #[test]
    fn platform_extension_detection_is_deterministic() {
        let mut extensions = BTreeSet::new();
        #[cfg(target_os = "windows")]
        {
            extensions.insert("VK_KHR_external_memory_win32".to_owned());
            assert_eq!(external_resource_support(&extensions), (true, false));
        }
        #[cfg(target_os = "linux")]
        {
            extensions.insert("VK_KHR_external_memory_fd".to_owned());
            assert_eq!(external_resource_support(&extensions), (true, false));
        }
    }

    #[test]
    fn preferred_device_chooses_discrete_amd_over_integrated_amd() {
        let info = VulkanRuntimeInfo {
            loader_available: true,
            devices: vec![
                device(
                    "AMD Radeon(TM) Graphics",
                    AMD_PCI_VENDOR_ID,
                    0x164e,
                    vk::PhysicalDeviceType::INTEGRATED_GPU,
                ),
                device(
                    "AMD Radeon RX 6800 XT",
                    AMD_PCI_VENDOR_ID,
                    0x73bf,
                    vk::PhysicalDeviceType::DISCRETE_GPU,
                ),
            ],
        };

        assert_eq!(
            info.preferred_compute_device().unwrap().name,
            "AMD Radeon RX 6800 XT"
        );
    }
}
