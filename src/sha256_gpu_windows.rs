use std::{
    cmp::min,
    sync::{Arc, Mutex},
};

use crate::types::{ChecksumPatternMatch, ChecksumPatternSpec};
use anyhow::Result;
use enum_map::{Enum, EnumMap, enum_map};
use log::info;
use once_cell::sync::OnceCell;

// ============================================================
// Pipeline selection
// ============================================================

#[derive(Clone, Copy, Enum)]
pub enum ComputePipelineVersionWindows {
    Sha256SingleWindows,
    Sha256DoubleWindows,
}

// ============================================================
// Uniforms (must match WGSL layout exactly)
// ============================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SearchConfig {
    input_len_bytes: u32,
    message_len_bytes: u32,
    compare_len_bytes: u32,
}

// ============================================================
// GPU container
// ============================================================

struct GPU {
    device: wgpu::Device,
    queue: wgpu::Queue,

    pipelines: EnumMap<ComputePipelineVersionWindows, wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,

    buffers: Mutex<Option<GpuBuffers>>,
}

struct GpuBuffers {
    input_upload_buffer: wgpu::Buffer,
    input_upload_capacity: u64,

    input_storage_buffer: wgpu::Buffer,
    input_storage_capacity: u64,

    search_config_buffer: wgpu::Buffer, // fixed-size (small struct)

    match_offsets_buffer: wgpu::Buffer,
    match_offsets_capacity: u64,

    match_count_buffer: wgpu::Buffer, // fixed-size (one u32)

    readback_offsets_buffer: wgpu::Buffer,
    readback_offsets_capacity: u64,

    readback_count_buffer: wgpu::Buffer, // fixed-size (one u32)

    bind_group: wgpu::BindGroup,
}

// ============================================================
// GPU initialization
// ============================================================

impl GPU {
    async fn init() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter.request_device(&Default::default()).await?;

        let shader = device.create_shader_module(wgpu::include_wgsl!("sha256_gpu_windows.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sliding SHA256 Bind Group Layout"),
            entries: &[
                // input_bytes
                layout_entry(0, true),
                // search_config
                layout_entry_uniform(1),
                // match_offsets
                layout_entry(2, false),
                // match_count
                layout_entry(3, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sliding SHA256x Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            ..Default::default()
        });

        let pipelines = enum_map! {
            ComputePipelineVersionWindows::Sha256SingleWindows => create_pipeline(
                &device, &shader, &pipeline_layout, false
            ),
            ComputePipelineVersionWindows::Sha256DoubleWindows => create_pipeline(
                &device, &shader, &pipeline_layout, true
            ),
        };

        Ok(Self {
            device,
            queue,
            pipelines,
            bind_group_layout,
            buffers: Mutex::new(None),
        })
    }

    fn pipeline(&self, v: ComputePipelineVersionWindows) -> &wgpu::ComputePipeline {
        &self.pipelines[v]
    }
}

static GPU_INSTANCE: OnceCell<GPU> = OnceCell::new();

fn gpu() -> Result<&'static GPU> {
    GPU_INSTANCE.get_or_try_init(|| pollster::block_on(GPU::init()))
}

// ============================================================
// Helpers
// ============================================================

// limits().max_compute_workgroups_per_dimension = 65_536

fn get_match_offsets_buffer_size(total_offsets: u32) -> anyhow::Result<u64> {
    // One u32 per potential output offset. Could be much lower.
    Ok(min(
        (total_offsets as u64) * 4,
        gpu()?.device.limits().max_storage_buffer_binding_size as u64,
    ))
}

fn layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn layout_entry_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn ensure_buffer_capacity(
    device: &wgpu::Device,
    existing: Option<wgpu::Buffer>,
    existing_capacity: u64,
    required_capacity: u64,
    usage: wgpu::BufferUsages,
    label: &str,
) -> (wgpu::Buffer, u64) {
    if let Some(buffer) = existing {
        if existing_capacity >= required_capacity {
            return (buffer, existing_capacity);
        }
    }

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: required_capacity,
        usage,
        mapped_at_creation: false,
    });

    (buffer, required_capacity)
}

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    sha256d: bool,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(if sha256d {
            "Sliding SHA256d"
        } else {
            "Sliding SHA256"
        }),
        layout: Some(layout),
        module: shader,
        entry_point: Some("sliding_sha256d"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[
                (
                    "CONFIG_WORKGROUP_SIZE",
                    device.limits().max_compute_workgroup_size_x as f64,
                ),
                ("CONFIG_ENABLE_SHA256D", if sha256d { 1.0 } else { 0.0 }),
                ("CONFIG_STRIDE_X", {
                    let wg_size = device.limits().max_compute_workgroup_size_x;
                    let workgroups_x = device.limits().max_compute_workgroups_per_dimension;
                    let stride_x = workgroups_x * wg_size;
                    stride_x as f64
                }),
            ],
            ..Default::default()
        },
        cache: None,
    })
}

// ============================================================
// Public API
// ============================================================

pub fn search_sha256_gpu_windows(
    input_data: &[u8],
    input_patterns: &[ChecksumPatternSpec],
    input_data_absolute_offset: usize,
    algo: ComputePipelineVersionWindows,
) -> anyhow::Result<Vec<ChecksumPatternMatch>> {
    debug_assert!(input_data.len() as u32 <= get_max_input_len_bytes()?);

    let gpu = gpu()?;

    let longest_pattern_length_bytes = input_patterns
        .iter()
        .map(|p| p.total_length())
        .max()
        .unwrap_or(0);

    let total_offsets =
        (input_data.len() as u32).saturating_sub(longest_pattern_length_bytes as u32) + 1;
    let wg_size = gpu.device.limits().max_compute_workgroup_size_x;

    // Choose X as large as possible
    let workgroups_x = gpu.device.limits().max_compute_workgroups_per_dimension;
    let threads_per_row = workgroups_x * wg_size;

    // Number of rows needed.
    let workgroups_y = ((total_offsets + threads_per_row - 1) / threads_per_row)
        // Clamp Y to device limit.
        .min(gpu.device.limits().max_compute_workgroups_per_dimension);

    // Required buffer sizes.
    let match_offsets_buffer_size = get_match_offsets_buffer_size(total_offsets)?;
    let packed_data_length_bytes = (get_packed_data_words_length(input_data.len()) * 4) as u64;

    let mut guard = gpu.buffers.lock().unwrap();
    let buffers = get_or_resize_buffers(
        &gpu,
        &mut guard,
        packed_data_length_bytes,
        match_offsets_buffer_size,
    );

    let mut all_results = Vec::new();

    for (checksum_pattern_idx, &pattern) in input_patterns.iter().enumerate() {
        // Edge case: data too small for this pattern.
        // Rare edge-case in the very last chunk, and/or if the total length is tiny.
        if input_data.len() < pattern.total_length() {
            continue;
        }

        let search_config_val = SearchConfig {
            input_len_bytes: input_data.len() as u32,
            message_len_bytes: pattern.chunk_len as u32,
            compare_len_bytes: pattern.checksum_len as u32,
        };

        write_inputs(
            &gpu,
            buffers,
            {
                // Don't need to re-write this on subsequent iterations.
                if checksum_pattern_idx == 0 {
                    Some(&input_data)
                } else {
                    None
                }
            },
            &search_config_val,
        )?;

        let local_result = dispatch_and_readback(
            &gpu,
            buffers,
            algo,
            (workgroups_x, workgroups_y),
            packed_data_length_bytes,
            checksum_pattern_idx == 0, // Disable copy when on subsequent pattern iterations.
        )?;

        // Materialize matches (rare path).
        for &relative_offset in &local_result {
            let absolute_offset = input_data_absolute_offset + relative_offset as usize;

            let chunk_and_checksum = &input_data
                [(relative_offset as usize)..(relative_offset as usize) + pattern.total_length()];

            let chunk_data = &chunk_and_checksum[..pattern.chunk_len];
            let checksum_data = &chunk_and_checksum[pattern.chunk_len..pattern.total_length()];

            debug_assert_eq!(chunk_data.len(), pattern.chunk_len);
            debug_assert_eq!(checksum_data.len(), pattern.checksum_len);

            info!(
                "✅ Match! Offset: {}=0x{:x}, Chunk Length: {}, Chunk: {:x?}",
                absolute_offset, absolute_offset, pattern.chunk_len, chunk_data
            );

            all_results.push(ChecksumPatternMatch {
                chunk_len: pattern.chunk_len,
                checksum_len: pattern.checksum_len,
                chunk_start_offset: absolute_offset as u64,
                chunk_data: chunk_data.to_vec(),
                checksum_data: checksum_data.to_vec(),
            });
        }
    }

    Ok(all_results)
}

/// Pack input bytes into u32 words (big endian).
///
/// Note: Does not do the SHA256 packing here (no 0x80, padding, length).
/// This function assumes packed_destination may not be zeroed.
/// This function doesn't write past the end of the last word filled by `input_data`,
/// even if the buffer is longer than the `input_data` (after word conversion).
fn pack_input_data<D>(input_data: &[u8], mut packed_destination: D)
where
    D: AsMut<[u32]>,
{
    let packed_destination = packed_destination.as_mut();
    let required_len_words = get_packed_data_words_length(input_data.len());
    if packed_destination.len() < required_len_words {
        panic!("packed_destination is too small");
    }

    let full_words = input_data.len() / 4;
    let tail_bytes = input_data.len() % 4;

    // Pack full words (4 bytes → 1 u32, big endian).
    let input_ptr = input_data.as_ptr();
    for word_idx in 0..full_words {
        // Unsafe: No bounds-checking needed because the upper bound is logically based on the input length.
        unsafe {
            let p = input_ptr.add(word_idx * 4) as *const u32;
            packed_destination[word_idx] = u32::from_be(p.read_unaligned());
        }
    }

    // Pack trailing partial word, if any.
    if tail_bytes != 0 {
        let base = full_words * 4;
        let mut tmp = [0u8; 4];
        tmp[..tail_bytes].copy_from_slice(&input_data[base..base + tail_bytes]);
        packed_destination[full_words] = u32::from_be_bytes(tmp);
    }
}

/// Get the number of u32 words required to store the input data.
fn get_packed_data_words_length(input_data_length_bytes: usize) -> usize {
    (input_data_length_bytes + 3) / 4
}

fn get_or_resize_buffers<'a>(
    gpu: &'a GPU,
    guard: &'a mut Option<GpuBuffers>,
    required_input_bytes: u64,
    match_offsets_buffer_size: u64,
) -> &'a mut GpuBuffers {
    let buffers = guard.get_or_insert_with(|| {
        let input_upload_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input_upload"),
            size: required_input_bytes,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let input_storage_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input"),
            size: required_input_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let search_config_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("search_config"),
            size: std::mem::size_of::<SearchConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let match_offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("match_offsets"),
            // One u32 per potential output offset. Could be much lower.
            size: match_offsets_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let match_count = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("match_count"),
            size: 4, // One u32.
            usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST // Must reset the point at each start.
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_offsets"),
            size: match_offsets_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_count = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_count"),
            size: 4, // Single integer.
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &gpu.bind_group_layout,
            entries: &[
                entry(0, &input_storage_buffer),
                entry(1, &search_config_buffer),
                entry(2, &match_offsets),
                entry(3, &match_count),
            ],
            label: Some("sliding-bind-group"),
        });

        GpuBuffers {
            input_upload_buffer,
            input_upload_capacity: required_input_bytes,
            input_storage_buffer,
            input_storage_capacity: required_input_bytes,
            search_config_buffer,
            match_offsets_buffer: match_offsets,
            match_offsets_capacity: match_offsets_buffer_size,
            match_count_buffer: match_count,
            readback_offsets_buffer: readback_offsets,
            readback_offsets_capacity: match_offsets_buffer_size,
            readback_count_buffer: readback_count,
            bind_group,
        }
    });

    let (new_input_upload_buffer, new_input_upload_capacity) = ensure_buffer_capacity(
        &gpu.device,
        Some(buffers.input_upload_buffer.clone()),
        buffers.input_upload_capacity,
        required_input_bytes,
        wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        "input_upload",
    );

    let (new_input_storage_buffer, new_input_storage_capacity) = ensure_buffer_capacity(
        &gpu.device,
        Some(buffers.input_storage_buffer.clone()),
        buffers.input_storage_capacity,
        required_input_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "input",
    );

    let (new_match_offsets_buffer, new_match_offsets_capacity) = ensure_buffer_capacity(
        &gpu.device,
        Some(buffers.match_offsets_buffer.clone()),
        buffers.match_offsets_capacity,
        match_offsets_buffer_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "match_offsets",
    );

    let (new_readback_offsets_buffer, new_readback_offsets_capacity) = ensure_buffer_capacity(
        &gpu.device,
        Some(buffers.readback_offsets_buffer.clone()),
        buffers.readback_offsets_capacity,
        match_offsets_buffer_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "readback_offsets",
    );
    let buffers_changed = new_input_upload_capacity != buffers.input_upload_capacity
        || new_input_storage_capacity != buffers.input_storage_capacity
        || new_match_offsets_capacity != buffers.match_offsets_capacity
        || new_readback_offsets_capacity != buffers.readback_offsets_capacity;

    if buffers_changed {
        buffers.bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &gpu.bind_group_layout,
            entries: &[
                entry(0, &new_input_storage_buffer),
                entry(1, &buffers.search_config_buffer),
                entry(2, &new_match_offsets_buffer),
                entry(3, &buffers.match_count_buffer),
            ],
            label: Some("sliding-bind-group"),
        });

        buffers.input_upload_buffer = new_input_upload_buffer;
        buffers.input_upload_capacity = new_input_upload_capacity;

        buffers.input_storage_buffer = new_input_storage_buffer;
        buffers.input_storage_capacity = new_input_storage_capacity;

        buffers.match_offsets_buffer = new_match_offsets_buffer;
        buffers.match_offsets_capacity = new_match_offsets_capacity;

        buffers.readback_offsets_buffer = new_readback_offsets_buffer;
        buffers.readback_offsets_capacity = new_readback_offsets_capacity;
    }
    buffers
}

fn write_inputs(
    gpu: &GPU,
    buffers: &mut GpuBuffers,
    input_data: Option<&[u8]>,
    search_config_val: &SearchConfig,
) -> anyhow::Result<()> {
    // Write padded bytes directly from `input_data` into upload buffer.
    if let Some(input_data) = input_data {
        let packed_data_length_bytes = get_packed_data_words_length(input_data.len()) as u64 * 4;

        let upload_buffer_slice = buffers
            .input_upload_buffer
            .slice(0..packed_data_length_bytes);

        let upload_map_done = Arc::new(Mutex::new(None));
        let upload_map_done_clone = upload_map_done.clone();
        upload_buffer_slice.map_async(wgpu::MapMode::Write, move |result| {
            *upload_map_done_clone.lock().unwrap() = Some(result);
        });

        // VERY IMPORTANT: Drive the mapping to completion.
        gpu.device.poll(wgpu::PollType::wait_indefinitely())?;

        // Check mapping result.
        match upload_map_done.lock().unwrap().take().unwrap() {
            Ok(()) => {}
            Err(e) => return Err(e.into()),
        }

        {
            let mut temp_buffer_u8 = upload_buffer_slice.get_mapped_range_mut();

            debug_assert!(
                // Must use .as_mut() or it panics.
                temp_buffer_u8.as_mut().len() % 4 == 0,
                "Buffer length is not a multiple of 4: {}",
                temp_buffer_u8.as_mut().len()
            );

            let temp_buffer_u32: &mut [u32] = bytemuck::cast_slice_mut(&mut temp_buffer_u8);

            pack_input_data(input_data, temp_buffer_u32);
        }
        buffers.input_upload_buffer.unmap();
    }

    gpu.queue.write_buffer(
        &buffers.search_config_buffer,
        0,
        bytemuck::bytes_of(search_config_val),
    );

    // Clear the output writer index.
    // Don't need to clear/rewrite the actual result buffer, as its length is virtually limited by this counter.
    gpu.queue
        .write_buffer(&buffers.match_count_buffer, 0, bytemuck::bytes_of(&0u32));

    Ok(())
}

fn dispatch_and_readback(
    gpu: &GPU,
    buffers: &mut GpuBuffers,
    algo: ComputePipelineVersionWindows,
    (workgroups_x, workgroups_y): (u32, u32),
    packed_data_length_bytes: u64,
    enable_input_buffer_copy: bool,
) -> anyhow::Result<Vec<u32>> {
    let mut encoder = gpu.device.create_command_encoder(&Default::default());

    // Copy in the input upload buffer to the input storage buffer.
    if enable_input_buffer_copy {
        encoder.copy_buffer_to_buffer(
            &buffers.input_upload_buffer,
            0,
            &buffers.input_storage_buffer,
            0,
            packed_data_length_bytes,
        );
    }

    // Dispatch.
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(gpu.pipeline(algo));
        pass.set_bind_group(0, &buffers.bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    // First, read the match_count buffer.
    encoder.copy_buffer_to_buffer(
        &buffers.match_count_buffer,
        0,
        &buffers.readback_count_buffer,
        0,
        4,
    );

    gpu.queue.submit(Some(encoder.finish()));

    buffers
        .readback_count_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});

    // Long stall (1-5 seconds) here, waiting on this poll for completion.
    // This is where 99% of the runtime occurs.
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;

    let result_count = {
        let mapped = buffers.readback_count_buffer.slice(..).get_mapped_range();
        let count = *bytemuck::from_bytes::<u32>(&mapped) as usize;
        drop(mapped);
        buffers.readback_count_buffer.unmap();
        count
    };

    // Hot path optimization: If no matches found, then no need to read the other buffer.
    // This is the nominal path, as matches are very rare.
    if result_count == 0 {
        return Ok(Vec::new());
    } else if (result_count as u64) >= buffers.match_offsets_capacity {
        // Technically could be fine with a >, but doing a >= to flag issues nearing the limit.
        anyhow::bail!(
            "Result count exceeds match offsets capacity: {} > {}",
            result_count,
            buffers.match_offsets_capacity
        );
    }

    // Read the relevant slice of the match_offsets_buffer.
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let readback_buffer_len_bytes = (result_count as u64) * 4;
    encoder.copy_buffer_to_buffer(
        &buffers.match_offsets_buffer,
        0,
        &buffers.readback_offsets_buffer,
        0,
        readback_buffer_len_bytes,
    );

    gpu.queue.submit(Some(encoder.finish()));

    buffers
        .readback_offsets_buffer
        .slice(0..readback_buffer_len_bytes)
        .map_async(wgpu::MapMode::Read, |_| {});

    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;

    let result = {
        let mapped = buffers
            .readback_offsets_buffer
            .slice(0..readback_buffer_len_bytes)
            .get_mapped_range();

        let offsets: &[u32] = bytemuck::cast_slice(&mapped);
        let v = offsets.to_vec();

        drop(mapped);
        buffers.readback_offsets_buffer.unmap();
        v
    };

    if false {
        println!(
            "result_count={}, result.len()={}, first chunk of result: {:?}",
            result_count,
            result.len(),
            &result[..std::cmp::min(64, result.len())],
        );
    }

    let result = result[..result_count].to_vec();
    Ok(result)
}

/// Max length of the input data to `search_sha256_gpu_windows()`.
pub fn get_max_input_len_bytes() -> anyhow::Result<u32> {
    let limits = gpu()?.device.limits();

    let threads_per_wg = limits.max_compute_workgroup_size_x as u64;
    let max_wg = limits.max_compute_workgroups_per_dimension as u64;

    let total_threads = threads_per_wg * max_wg * max_wg;

    Ok(total_threads
        .min(u32::MAX as u64)
        .min(limits.max_buffer_size)
        .min(limits.max_storage_buffer_binding_size as u64) as u32)
}

fn entry(binding: u32, buffer: &'_ wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
