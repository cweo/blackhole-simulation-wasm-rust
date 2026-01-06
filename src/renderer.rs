use std::sync::Arc;
use wgpu::*;

#[cfg(not(target_arch = "wasm32"))]
use winit::window::Window;

use crate::camera::{Camera, CameraUniform};
use crate::SimParams;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BlackHoleUniform {
    spin: f32,
    mass: f32,
    disk_inner: f32,
    disk_outer: f32,
    time: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,  // Extra padding for 16-byte alignment of vec3 in WGSL
    _padding5: f32,
    _padding6: f32,
    _padding7: f32,
}

impl BlackHoleUniform {
    fn new(spin: f32, time: f32) -> Self {
        let a = spin;
        let z1 = 1.0 + (1.0 - a * a).powf(1.0 / 3.0) * ((1.0 + a).powf(1.0 / 3.0) + (1.0 - a).powf(1.0 / 3.0));
        let z2 = (3.0 * a * a + z1 * z1).sqrt();
        let isco = 3.0 + z2 - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt();
        
        Self {
            spin,
            mass: 1.0,
            disk_inner: isco,
            disk_outer: 20.0,
            time,
            _padding1: 0.0,
            _padding2: 0.0,
            _padding3: 0.0,
            _padding4: 0.0,
            _padding5: 0.0,
            _padding6: 0.0,
            _padding7: 0.0,
        }
    }
}

pub struct Renderer {
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    compute_pipeline: ComputePipeline,
    compute_bind_group_layout: BindGroupLayout,
    compute_bind_group: BindGroup,
    render_pipeline: RenderPipeline,
    render_bind_group_layout: BindGroupLayout,
    render_bind_group: BindGroup,
    render_texture: Texture,
    render_texture_view: TextureView,
    camera_buffer: Buffer,
    blackhole_buffer: Buffer,
    width: u32,
    height: u32,
    render_width: u32,
    render_height: u32,
    #[cfg(not(target_arch = "wasm32"))]
    start_time: std::time::Instant,
    #[cfg(target_arch = "wasm32")]
    start_time: f64,
}

impl Renderer {
    #[cfg(target_arch = "wasm32")]
    pub async fn new_from_canvas(canvas: &web_sys::HtmlCanvasElement, width: u32, height: u32) -> Self {
        use wasm_bindgen::JsCast;
        
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let surface_target = SurfaceTarget::Canvas(canvas.clone());
        let surface = instance.create_surface(surface_target).unwrap();

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find adapter");

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    label: None,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        Self::init_common(surface, device, queue, &adapter, width, height)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn new(window: Arc<Window>, width: u32, height: u32) -> Self {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    label: None,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        Self::init_common(surface, device, queue, &adapter, width, height)
    }

    fn init_common(surface: Surface<'static>, device: Device, queue: Queue, adapter: &Adapter, width: u32, height: u32) -> Self {
        let surface_caps = surface.get_capabilities(adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let (render_texture, render_texture_view) = Self::create_render_texture(&device, width, height);

        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let blackhole_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("BlackHole Buffer"),
            size: std::mem::size_of::<BlackHoleUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/blackhole.wgsl").into()),
        });

        let compute_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_bind_group = Self::create_compute_bind_group(
            &device, &compute_bind_group_layout, &render_texture_view, &camera_buffer, &blackhole_buffer,
        );

        let compute_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let render_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
        });

        let render_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let render_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&render_texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            compute_pipeline,
            compute_bind_group_layout,
            compute_bind_group,
            render_pipeline,
            render_bind_group_layout,
            render_bind_group,
            render_texture,
            render_texture_view,
            camera_buffer,
            blackhole_buffer,
            width,
            height,
            render_width: width,
            render_height: height,
            #[cfg(not(target_arch = "wasm32"))]
            start_time: std::time::Instant::now(),
            #[cfg(target_arch = "wasm32")]
            start_time: web_sys::window().unwrap().performance().unwrap().now() / 1000.0,
        }
    }

    fn create_render_texture(device: &Device, width: u32, height: u32) -> (Texture, TextureView) {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("Render Texture"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_compute_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        texture_view: &TextureView,
        camera_buffer: &Buffer,
        blackhole_buffer: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(texture_view) },
                BindGroupEntry { binding: 1, resource: camera_buffer.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: blackhole_buffer.as_entire_binding() },
            ],
        })
    }

    pub fn resize(&mut self, width: u32, height: u32, resolution_scale: f32) {
        if width == 0 || height == 0 { return; }

        self.width = width;
        self.height = height;
        self.render_width = ((width as f32) * resolution_scale) as u32;
        self.render_height = ((height as f32) * resolution_scale) as u32;

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);

        let (render_texture, render_texture_view) = Self::create_render_texture(&self.device, self.render_width, self.render_height);
        self.render_texture = render_texture;
        self.render_texture_view = render_texture_view;

        self.compute_bind_group = Self::create_compute_bind_group(
            &self.device, &self.compute_bind_group_layout, &self.render_texture_view, &self.camera_buffer, &self.blackhole_buffer,
        );

        let sampler = self.device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        self.render_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &self.render_bind_group_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&self.render_texture_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&sampler) },
            ],
        });
    }

    pub fn render(&mut self, camera: &Camera, params: &SimParams) -> Result<(), SurfaceError> {
        let camera_uniform = camera.uniform_data(self.render_width, self.render_height);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        #[cfg(not(target_arch = "wasm32"))]
        let time = self.start_time.elapsed().as_secs_f32();
        #[cfg(target_arch = "wasm32")]
        let time = (web_sys::window().unwrap().performance().unwrap().now() / 1000.0 - self.start_time) as f32;

        let blackhole_uniform = BlackHoleUniform::new(params.spin, time);
        self.queue.write_buffer(&self.blackhole_buffer, 0, bytemuck::cast_slice(&[blackhole_uniform]));

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            
            let workgroup_x = (self.render_width + 7) / 8;
            let workgroup_y = (self.render_height + 7) / 8;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
