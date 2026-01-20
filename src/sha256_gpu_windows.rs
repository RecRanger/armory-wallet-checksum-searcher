use std::sync::Mutex;

use anyhow::Result;
use enum_map::{Enum, EnumMap, enum_map};
use once_cell::sync::OnceCell;
use wgpu::util::DeviceExt;

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
    input_buffer: wgpu::Buffer,
    input_capacity: u64,

    match_offsets: wgpu::Buffer,
    match_offsets_capacity: u64,

    match_count: wgpu::Buffer,

    config_buffer: wgpu::Buffer,

    readback_offsets: wgpu::Buffer,
    readback_count: wgpu::Buffer,

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
                // match_offsets
                layout_entry(1, false),
                // match_count
                layout_entry(2, false),
                // config
                layout_entry_uniform(3),
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
    message_len_bytes: u32,
    compare_len_bytes: u32,
    algo: ComputePipelineVersionWindows,
) -> Result<Vec<u32>> {
    debug_assert!(message_len_bytes > 0);
    debug_assert!(compare_len_bytes <= 32);

    let gpu = gpu()?;

    let total_offsets = input_data.len() as u32 - message_len_bytes - compare_len_bytes + 1;
    let wg_size = gpu.device.limits().max_compute_workgroup_size_x;
    let num_workgroups = (total_offsets + wg_size - 1) / wg_size;

    // Pack input bytes into u32 words (big endian).
    let mut packed = vec![0u32; (input_data.len() + 3) / 4];
    for (i, b) in input_data.iter().enumerate() {
        packed[i / 4] |= (*b as u32) << (24 - 8 * (i & 3));
    }

    let mut guard = gpu.buffers.lock().unwrap();

    let buffers = guard.get_or_insert_with(|| {
        let input_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let match_offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("match_offsets"),
            size: total_offsets as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let match_count = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("match_count"),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let config = SearchConfig {
            input_len_bytes: input_data.len() as u32,
            message_len_bytes,
            compare_len_bytes,
        };

        let config_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("config"),
                contents: bytemuck::bytes_of(&config),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let readback_offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_offsets"),
            size: total_offsets as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_count = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_count"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &gpu.bind_group_layout,
            entries: &[
                entry(0, &input_buffer),
                entry(1, &match_offsets),
                entry(2, &match_count),
                entry(3, &config_buffer),
            ],
            label: Some("sliding-bind-group"),
        });

        GpuBuffers {
            input_buffer,
            input_capacity: packed.len() as u64 * 4,
            match_offsets,
            match_offsets_capacity: total_offsets as u64 * 4,
            match_count,
            config_buffer,
            readback_offsets,
            readback_count,
            bind_group,
        }
    });

    let mut encoder = gpu.device.create_command_encoder(&Default::default());

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(gpu.pipeline(algo));
        pass.set_bind_group(0, &buffers.bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &buffers.match_offsets,
        0,
        &buffers.readback_offsets,
        0,
        buffers.match_offsets_capacity,
    );
    encoder.copy_buffer_to_buffer(&buffers.match_count, 0, &buffers.readback_count, 0, 4);

    gpu.queue.submit(Some(encoder.finish()));

    buffers
        .readback_count
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});
    buffers
        .readback_offsets
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});

    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;

    let result_count = {
        let count_bytes = buffers.readback_count.slice(..).get_mapped_range();
        let count_bytes_u32_array: &u32 = bytemuck::from_bytes::<u32>(&count_bytes);
        *count_bytes_u32_array as usize
    };

    let offsets_bytes = buffers.readback_offsets.slice(..).get_mapped_range();
    let offsets_all: &[u32] = bytemuck::cast_slice(&offsets_bytes);

    // Clone here is very cheap (generally, there are zero values in the slice).
    // Required in order to be able to unmap the readback buffer.
    let result = offsets_all[..result_count].to_vec().clone();

    drop(offsets_bytes);
    buffers.readback_offsets.unmap();
    buffers.readback_count.unmap();

    Ok(result)
}

fn entry(binding: u32, buffer: &'_ wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
