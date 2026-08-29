use vrb_core::BackendError;

#[derive(Debug, Clone, Copy)]
pub struct SharedBufferSpec<'a> {
    pub bytes: u64,
    pub initial_data: Option<&'a [u8]>,
    pub readback: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportedSharedBuffer {
    /// Borrowed OPAQUE_WIN32_KMT handle valid only for the callback duration.
    pub handle: u64,
    pub allocation_size: u64,
    pub logical_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedBufferSessionOutput<T> {
    pub value: T,
    pub vulkan_device: String,
    pub readbacks: Vec<Option<Vec<u8>>>,
}

pub fn with_exported_shared_buffers<T>(
    specs: &[SharedBufferSpec<'_>],
    operation: impl FnOnce(&[ExportedSharedBuffer]) -> Result<T, BackendError>,
) -> Result<ExportedBufferSessionOutput<T>, BackendError> {
    platform::with_buffers(specs, operation)
}

#[cfg(not(target_os = "windows"))]
mod platform {
    use super::{
        BackendError, ExportedBufferSessionOutput, ExportedSharedBuffer, SharedBufferSpec,
    };

    pub(super) fn with_buffers<T>(
        _specs: &[SharedBufferSpec<'_>],
        _operation: impl FnOnce(&[ExportedSharedBuffer]) -> Result<T, BackendError>,
    ) -> Result<ExportedBufferSessionOutput<T>, BackendError> {
        Err(BackendError::Unsupported(
            "Vulkan shared-buffer export currently implements the Win32 KMT path only".to_owned(),
        ))
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use super::{
        BackendError, ExportedBufferSessionOutput, ExportedSharedBuffer, SharedBufferSpec,
    };
    use ash::{khr, vk, Device, Entry, Instance};
    use std::ffi::{CStr, CString};
    use std::ptr;
    use std::slice;

    const AMD_PCI_VENDOR_ID: u32 = 0x1002;
    const DEFAULT_MAX_SESSION_BYTES: u64 = 1024 * 1024 * 1024;
    const DEFAULT_MAX_BUFFERS: usize = 64;

    pub(super) fn with_buffers<T>(
        specs: &[SharedBufferSpec<'_>],
        operation: impl FnOnce(&[ExportedSharedBuffer]) -> Result<T, BackendError>,
    ) -> Result<ExportedBufferSessionOutput<T>, BackendError> {
        validate_specs(specs)?;
        let context = VulkanContext::create()?;
        let mut buffers = Vec::with_capacity(specs.len());

        for spec in specs {
            let shared = context.create_shared_buffer(spec.bytes)?;
            if let Some(initial) = spec.initial_data {
                let staging = context.create_staging_buffer(spec.bytes)?;
                context.write_staging(&staging, initial)?;
                context.copy_staging_to_device(&staging, &shared)?;
            }
            buffers.push(shared);
        }

        let mut exported = Vec::with_capacity(buffers.len());
        for buffer in &buffers {
            context.release_to_external(buffer)?;
            let handle = context.export_kmt_handle(buffer)?;
            exported.push(ExportedSharedBuffer {
                handle: handle as usize as u64,
                allocation_size: buffer.allocation_size,
                logical_size: buffer.logical_size,
            });
        }

        let operation_result = operation(&exported);

        // The callback contract is synchronous. Whether it succeeds or fails,
        // reclaim Vulkan ownership before any allocation is destroyed.
        let mut reclaim_error = None;
        for buffer in &buffers {
            if let Err(error) = context.acquire_from_external(buffer) {
                if reclaim_error.is_none() {
                    reclaim_error = Some(error);
                }
            }
        }

        let value = operation_result?;
        if let Some(error) = reclaim_error {
            return Err(error);
        }

        let mut readbacks = Vec::with_capacity(buffers.len());
        for (spec, buffer) in specs.iter().zip(&buffers) {
            if spec.readback {
                let staging = context.create_staging_buffer(buffer.logical_size)?;
                context.copy_device_to_staging(buffer, &staging)?;
                readbacks.push(Some(context.read_staging(&staging)?));
            } else {
                readbacks.push(None);
            }
        }

        Ok(ExportedBufferSessionOutput {
            value,
            vulkan_device: context.device_name.clone(),
            readbacks,
        })
    }

    fn validate_specs(specs: &[SharedBufferSpec<'_>]) -> Result<(), BackendError> {
        if specs.is_empty() {
            return Err(BackendError::Internal(
                "shared-buffer session requires at least one buffer".to_owned(),
            ));
        }
        if specs.len() > DEFAULT_MAX_BUFFERS {
            return Err(BackendError::Unsupported(format!(
                "shared-buffer session requests {} buffers, maximum is {DEFAULT_MAX_BUFFERS}",
                specs.len()
            )));
        }
        let mut total = 0_u64;
        for (index, spec) in specs.iter().enumerate() {
            if spec.bytes == 0 {
                return Err(BackendError::Internal(format!(
                    "shared-buffer {index} size must be non-zero"
                )));
            }
            if let Some(initial) = spec.initial_data {
                let actual = u64::try_from(initial.len()).map_err(|_| {
                    BackendError::Unsupported(format!(
                        "shared-buffer {index} initial data length does not fit u64"
                    ))
                })?;
                if actual != spec.bytes {
                    return Err(BackendError::Internal(format!(
                        "shared-buffer {index} initial data is {actual} bytes, expected {}",
                        spec.bytes
                    )));
                }
            }
            total = total.checked_add(spec.bytes).ok_or_else(|| {
                BackendError::Unsupported("shared-buffer session byte count overflow".to_owned())
            })?;
        }
        let maximum = std::env::var("VRB_MAX_SHARED_SESSION_BYTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MAX_SESSION_BYTES);
        if total > maximum {
            return Err(BackendError::Unsupported(format!(
                "shared-buffer session requests {total} bytes, exceeding VRB_MAX_SHARED_SESSION_BYTES={maximum}"
            )));
        }
        Ok(())
    }

    struct VulkanContext {
        _entry: Entry,
        instance: Instance,
        physical_device: vk::PhysicalDevice,
        device: Device,
        queue_family_index: u32,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        external_memory_win32: khr::external_memory_win32::Device,
        device_name: String,
    }

    impl VulkanContext {
        fn create() -> Result<Self, BackendError> {
            // SAFETY: loads the administrator-installed Vulkan loader.
            let entry = unsafe { Entry::load() }
                .map_err(|error| BackendError::Unavailable(format!("Vulkan loader: {error}")))?;
            let app_name = CString::new("vrb-vulkan-shared-buffer")
                .map_err(|error| BackendError::Internal(error.to_string()))?;
            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .application_version(1)
                .engine_name(&app_name)
                .engine_version(1)
                .api_version(vk::API_VERSION_1_1);
            let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
            // SAFETY: create structures live through the call.
            let instance = unsafe { entry.create_instance(&instance_info, None) }
                .map_err(|error| BackendError::Probe(format!("vkCreateInstance: {error:?}")))?;

            let selected = select_vulkan_device(&instance);
            let (physical_device, queue_family_index, device_name) = match selected {
                Ok(value) => value,
                Err(error) => {
                    // SAFETY: no children exist yet.
                    unsafe { instance.destroy_instance(None) };
                    return Err(error);
                }
            };

            let handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT;
            let external_info = vk::PhysicalDeviceExternalBufferInfo::default()
                .flags(vk::BufferCreateFlags::empty())
                .usage(shared_buffer_usage())
                .handle_type(handle_type);
            let mut external_properties = vk::ExternalBufferProperties::default();
            // SAFETY: selected physical device belongs to this instance.
            unsafe {
                instance.get_physical_device_external_buffer_properties(
                    physical_device,
                    &external_info,
                    &mut external_properties,
                )
            };
            if !external_properties
                .external_memory_properties
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
            {
                // SAFETY: no device children exist.
                unsafe { instance.destroy_instance(None) };
                return Err(BackendError::Unsupported(format!(
                    "Vulkan device '{device_name}' cannot export OPAQUE_WIN32_KMT memory"
                )));
            }

            let priorities = [1.0_f32];
            let queue_infos = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)];
            let extension_names = [khr::external_memory_win32::NAME.as_ptr()];
            let device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&extension_names);
            // SAFETY: selection and arrays are valid for this call.
            let device =
                match unsafe { instance.create_device(physical_device, &device_info, None) } {
                    Ok(device) => device,
                    Err(error) => {
                        // SAFETY: instance has no device child on failure.
                        unsafe { instance.destroy_instance(None) };
                        return Err(BackendError::Probe(format!(
                            "vkCreateDevice for '{device_name}': {error:?}"
                        )));
                    }
                };
            // SAFETY: requested queue exists.
            let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
            let pool_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);
            // SAFETY: device and queue family are valid.
            let command_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
                Ok(pool) => pool,
                Err(error) => {
                    // SAFETY: no other device children exist.
                    unsafe {
                        device.destroy_device(None);
                        instance.destroy_instance(None);
                    }
                    return Err(BackendError::Probe(format!(
                        "vkCreateCommandPool: {error:?}"
                    )));
                }
            };
            let external_memory_win32 = khr::external_memory_win32::Device::new(&instance, &device);

            Ok(Self {
                _entry: entry,
                instance,
                physical_device,
                device,
                queue_family_index,
                queue,
                command_pool,
                external_memory_win32,
                device_name,
            })
        }

        fn create_shared_buffer(&self, bytes: u64) -> Result<VulkanBuffer, BackendError> {
            let handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT;
            let mut external =
                vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
            let info = vk::BufferCreateInfo::default()
                .size(bytes)
                .usage(shared_buffer_usage())
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut external);
            // SAFETY: create info is valid.
            let buffer = unsafe { self.device.create_buffer(&info, None) }
                .map_err(|error| BackendError::Internal(format!("vkCreateBuffer: {error:?}")))?;
            // SAFETY: buffer belongs to this device.
            let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
            let memory_properties = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical_device)
            };
            let memory_type_index = find_memory_type(
                &memory_properties,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or_else(|| {
                // SAFETY: live unbound buffer.
                unsafe { self.device.destroy_buffer(buffer, None) };
                BackendError::Unsupported(
                    "no device-local memory type satisfies exportable buffer".to_owned(),
                )
            })?;
            let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
            let allocation_info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index)
                .push_next(&mut dedicated)
                .push_next(&mut export);
            // SAFETY: allocation parameters came from buffer requirements.
            let memory = match unsafe { self.device.allocate_memory(&allocation_info, None) } {
                Ok(memory) => memory,
                Err(error) => {
                    // SAFETY: buffer is unbound.
                    unsafe { self.device.destroy_buffer(buffer, None) };
                    return Err(BackendError::Internal(format!(
                        "vkAllocateMemory(exportable): {error:?}"
                    )));
                }
            };
            // SAFETY: allocation satisfies buffer requirements.
            if let Err(error) = unsafe { self.device.bind_buffer_memory(buffer, memory, 0) } {
                // SAFETY: resources are idle and owned here.
                unsafe {
                    self.device.free_memory(memory, None);
                    self.device.destroy_buffer(buffer, None);
                }
                return Err(BackendError::Internal(format!(
                    "vkBindBufferMemory(exportable): {error:?}"
                )));
            }
            Ok(VulkanBuffer {
                device: self.device.clone(),
                buffer,
                memory,
                logical_size: bytes,
                allocation_size: requirements.size,
            })
        }

        fn create_staging_buffer(&self, bytes: u64) -> Result<VulkanBuffer, BackendError> {
            let info = vk::BufferCreateInfo::default()
                .size(bytes)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            // SAFETY: create info is valid.
            let buffer = unsafe { self.device.create_buffer(&info, None) }.map_err(|error| {
                BackendError::Internal(format!("staging vkCreateBuffer: {error:?}"))
            })?;
            // SAFETY: buffer belongs to device.
            let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
            let properties = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical_device)
            };
            let required =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let memory_type_index =
                find_memory_type(&properties, requirements.memory_type_bits, required).ok_or_else(
                    || {
                        // SAFETY: buffer is unbound.
                        unsafe { self.device.destroy_buffer(buffer, None) };
                        BackendError::Unsupported(
                            "no HOST_VISIBLE|HOST_COHERENT staging memory type is available"
                                .to_owned(),
                        )
                    },
                )?;
            let allocation_info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index);
            // SAFETY: allocation parameters came from requirements.
            let memory = match unsafe { self.device.allocate_memory(&allocation_info, None) } {
                Ok(memory) => memory,
                Err(error) => {
                    // SAFETY: buffer is unbound.
                    unsafe { self.device.destroy_buffer(buffer, None) };
                    return Err(BackendError::Internal(format!(
                        "staging vkAllocateMemory: {error:?}"
                    )));
                }
            };
            // SAFETY: allocation satisfies requirements.
            if let Err(error) = unsafe { self.device.bind_buffer_memory(buffer, memory, 0) } {
                // SAFETY: resources are idle and owned here.
                unsafe {
                    self.device.free_memory(memory, None);
                    self.device.destroy_buffer(buffer, None);
                }
                return Err(BackendError::Internal(format!(
                    "staging vkBindBufferMemory: {error:?}"
                )));
            }
            Ok(VulkanBuffer {
                device: self.device.clone(),
                buffer,
                memory,
                logical_size: bytes,
                allocation_size: requirements.size,
            })
        }

        fn write_staging(&self, staging: &VulkanBuffer, data: &[u8]) -> Result<(), BackendError> {
            if data.len() as u64 != staging.logical_size {
                return Err(BackendError::Internal(format!(
                    "staging write length {} does not match buffer size {}",
                    data.len(),
                    staging.logical_size
                )));
            }
            // SAFETY: staging allocation is HOST_VISIBLE and mapped range is valid.
            let pointer = unsafe {
                self.device.map_memory(
                    staging.memory,
                    0,
                    staging.logical_size,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|error| BackendError::Internal(format!("vkMapMemory(upload): {error:?}")))?;
            // SAFETY: mapped region is at least data.len bytes and non-overlapping.
            unsafe {
                ptr::copy_nonoverlapping(data.as_ptr(), pointer.cast::<u8>(), data.len());
                self.device.unmap_memory(staging.memory);
            }
            Ok(())
        }

        fn read_staging(&self, staging: &VulkanBuffer) -> Result<Vec<u8>, BackendError> {
            let length = usize::try_from(staging.logical_size).map_err(|_| {
                BackendError::Unsupported("staging readback length does not fit usize".to_owned())
            })?;
            // SAFETY: staging memory is HOST_VISIBLE/HOST_COHERENT and queue work is complete.
            let pointer = unsafe {
                self.device.map_memory(
                    staging.memory,
                    0,
                    staging.logical_size,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|error| BackendError::Internal(format!("vkMapMemory(readback): {error:?}")))?;
            // SAFETY: mapped region remains valid through copy.
            let bytes = unsafe { slice::from_raw_parts(pointer.cast::<u8>(), length) }.to_vec();
            // SAFETY: mapping is released exactly once.
            unsafe { self.device.unmap_memory(staging.memory) };
            Ok(bytes)
        }

        fn copy_staging_to_device(
            &self,
            staging: &VulkanBuffer,
            shared: &VulkanBuffer,
        ) -> Result<(), BackendError> {
            self.submit_one_time(|command_buffer| {
                let host_barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.queue_family_index)
                    .buffer(staging.buffer)
                    .offset(0)
                    .size(staging.logical_size);
                // SAFETY: command buffer is recording and barriers reference live buffers.
                unsafe {
                    self.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::HOST,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[host_barrier],
                        &[],
                    )
                };
                let region = vk::BufferCopy::default().size(staging.logical_size);
                // SAFETY: buffers are sized equally for this copy.
                unsafe {
                    self.device.cmd_copy_buffer(
                        command_buffer,
                        staging.buffer,
                        shared.buffer,
                        &[region],
                    )
                };
            })
        }

        fn copy_device_to_staging(
            &self,
            shared: &VulkanBuffer,
            staging: &VulkanBuffer,
        ) -> Result<(), BackendError> {
            self.submit_one_time(|command_buffer| {
                let region = vk::BufferCopy::default().size(shared.logical_size);
                // SAFETY: buffers are live and correctly sized.
                unsafe {
                    self.device.cmd_copy_buffer(
                        command_buffer,
                        shared.buffer,
                        staging.buffer,
                        &[region],
                    )
                };
                let host_barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.queue_family_index)
                    .buffer(staging.buffer)
                    .offset(0)
                    .size(staging.logical_size);
                // SAFETY: establishes visibility to host after transfer.
                unsafe {
                    self.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[host_barrier],
                        &[],
                    )
                };
            })
        }

        fn export_kmt_handle(&self, shared: &VulkanBuffer) -> Result<vk::HANDLE, BackendError> {
            let info = vk::MemoryGetWin32HandleInfoKHR::default()
                .memory(shared.memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT);
            // SAFETY: allocation was created with the same export type.
            unsafe { self.external_memory_win32.get_memory_win32_handle(&info) }.map_err(|error| {
                BackendError::Internal(format!("vkGetMemoryWin32HandleKHR: {error:?}"))
            })
        }

        fn release_to_external(&self, shared: &VulkanBuffer) -> Result<(), BackendError> {
            self.submit_one_time(|command_buffer| {
                let barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                    .buffer(shared.buffer)
                    .offset(0)
                    .size(shared.logical_size);
                // SAFETY: releases queue-family ownership to external consumer.
                unsafe {
                    self.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[barrier],
                        &[],
                    )
                };
            })
        }

        fn acquire_from_external(&self, shared: &VulkanBuffer) -> Result<(), BackendError> {
            self.submit_one_time(|command_buffer| {
                let barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                    .dst_queue_family_index(self.queue_family_index)
                    .buffer(shared.buffer)
                    .offset(0)
                    .size(shared.logical_size);
                // SAFETY: reacquires queue-family ownership after synchronous external callback.
                unsafe {
                    self.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[barrier],
                        &[],
                    )
                };
            })
        }

        fn submit_one_time(
            &self,
            record: impl FnOnce(vk::CommandBuffer),
        ) -> Result<(), BackendError> {
            let allocation_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: command pool belongs to device.
            let command_buffers = unsafe { self.device.allocate_command_buffers(&allocation_info) }
                .map_err(|error| {
                    BackendError::Internal(format!("vkAllocateCommandBuffers: {error:?}"))
                })?;
            let command_buffer = command_buffers[0];
            let result = (|| {
                let begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                // SAFETY: fresh command buffer is idle.
                unsafe { self.device.begin_command_buffer(command_buffer, &begin) }.map_err(
                    |error| BackendError::Internal(format!("vkBeginCommandBuffer: {error:?}")),
                )?;
                record(command_buffer);
                // SAFETY: recording is complete.
                unsafe { self.device.end_command_buffer(command_buffer) }.map_err(|error| {
                    BackendError::Internal(format!("vkEndCommandBuffer: {error:?}"))
                })?;
                let list = [command_buffer];
                let submits = [vk::SubmitInfo::default().command_buffers(&list)];
                // SAFETY: queue and command buffer belong to same device.
                unsafe {
                    self.device
                        .queue_submit(self.queue, &submits, vk::Fence::null())
                }
                .map_err(|error| BackendError::Internal(format!("vkQueueSubmit: {error:?}")))?;
                // SAFETY: host waits synchronously for completion.
                unsafe { self.device.queue_wait_idle(self.queue) }.map_err(|error| {
                    BackendError::Internal(format!("vkQueueWaitIdle: {error:?}"))
                })?;
                Ok(())
            })();
            // SAFETY: submission is idle on success; failed recording is not in flight.
            unsafe {
                self.device
                    .free_command_buffers(self.command_pool, &command_buffers)
            };
            result
        }
    }

    impl Drop for VulkanContext {
        fn drop(&mut self) {
            // SAFETY: session buffers drop before context; wait defensively.
            unsafe {
                let _ = self.device.device_wait_idle();
                self.device.destroy_command_pool(self.command_pool, None);
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }

    struct VulkanBuffer {
        device: Device,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        logical_size: u64,
        allocation_size: u64,
    }

    impl Drop for VulkanBuffer {
        fn drop(&mut self) {
            // SAFETY: buffer owns both resources and device outlives this guard.
            unsafe {
                self.device.destroy_buffer(self.buffer, None);
                self.device.free_memory(self.memory, None);
            }
        }
    }

    fn select_vulkan_device(
        instance: &Instance,
    ) -> Result<(vk::PhysicalDevice, u32, String), BackendError> {
        // SAFETY: instance is valid.
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|error| BackendError::Probe(format!("enumerate Vulkan devices: {error:?}")))?;
        let mut candidates = Vec::new();
        for physical_device in devices {
            // SAFETY: physical device came from instance.
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            // SAFETY: Vulkan device_name is NUL terminated.
            let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: physical device came from instance.
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let Some((queue_index, _)) = queue_families.iter().enumerate().find(|(_, family)| {
                family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            }) else {
                continue;
            };
            // SAFETY: physical device came from instance.
            let extensions =
                unsafe { instance.enumerate_device_extension_properties(physical_device) }
                    .map_err(|error| {
                        BackendError::Probe(format!("enumerate Vulkan extensions: {error:?}"))
                    })?;
            let supports_win32 = extensions.iter().any(|extension| {
                // SAFETY: Vulkan extension_name is NUL terminated.
                (unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) })
                    == khr::external_memory_win32::NAME
            });
            if !supports_win32 {
                continue;
            }
            let rank = (
                if properties.vendor_id == AMD_PCI_VENDOR_ID {
                    0_u8
                } else {
                    1_u8
                },
                if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    0_u8
                } else {
                    1_u8
                },
                properties.device_id,
            );
            candidates.push((rank, physical_device, queue_index as u32, name));
        }
        candidates.sort_by_key(|candidate| candidate.0);
        candidates
            .into_iter()
            .next()
            .map(|(_, device, queue, name)| (device, queue, name))
            .ok_or_else(|| {
                BackendError::Unavailable(
                    "no compute-capable Vulkan device with VK_KHR_external_memory_win32 was found"
                        .to_owned(),
                )
            })
    }

    fn shared_buffer_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
    }

    fn find_memory_type(
        properties: &vk::PhysicalDeviceMemoryProperties,
        memory_type_bits: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        (0..properties.memory_type_count).find(|index| {
            let supported = memory_type_bits & (1_u32 << *index) != 0;
            let flags = properties.memory_types[*index as usize].property_flags;
            supported && flags.contains(required)
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn spec_validation_rejects_zero_size() {
            let specs = [SharedBufferSpec {
                bytes: 0,
                initial_data: None,
                readback: false,
            }];
            assert!(validate_specs(&specs).is_err());
        }
    }
}
