//! GPU-accelerated offline high-quality black hole renderer
//! Uses wgpu compute shaders for massively parallel ray tracing

use glam::{Mat4, Vec3};
use image::{ImageBuffer, Rgba};
use wgpu::*;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy)]
pub enum Quality {
    Preview,
    High,
    Ultra,
    Insane,
}

impl Quality {
    fn max_steps(&self) -> u32 {
        match self {
            Quality::Preview => 500,
            Quality::High => 2000,
            Quality::Ultra => 5000,
            Quality::Insane => 10000,
        }
    }
    
    fn samples(&self) -> u32 {
        match self {
            Quality::Preview => 1,
            Quality::High => 4,
            Quality::Ultra => 16,
            Quality::Insane => 64,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    position: [f32; 4],
    inv_view_proj: [f32; 16],
    resolution: [f32; 2],
    fov: f32,
    _padding: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BlackHoleUniform {
    spin: f32,
    mass: f32,
    disk_inner: f32,
    disk_outer: f32,
    time: f32,
    max_steps: f32,
    sample_index: f32,
    total_samples: f32,
    _padding: [f32; 4],
}

fn isco_radius(spin: f32) -> f32 {
    let a = spin;
    let z1 = 1.0 + (1.0 - a * a).powf(1.0 / 3.0) * ((1.0 + a).powf(1.0 / 3.0) + (1.0 - a).powf(1.0 / 3.0));
    let z2 = (3.0 * a * a + z1 * z1).sqrt();
    3.0 + z2 - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt()
}

fn horizon_radius(spin: f32) -> f32 {
    1.0 + (1.0 - spin * spin).max(0.0).sqrt()
}

// Gargantua-style thin disk shader with proper gravitational lensing
const SHADER_SOURCE: &str = r#"
struct CameraUniform {
    position: vec4<f32>,
    inv_view_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    fov: f32,
    _padding: f32,
}

struct BlackHoleUniform {
    spin: f32,
    mass: f32,
    disk_inner: f32,
    disk_outer: f32,
    time: f32,
    max_steps: f32,
    sample_index: f32,
    total_samples: f32,
    _padding: vec4<f32>,
}

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var<uniform> camera: CameraUniform;
@group(0) @binding(2) var<uniform> blackhole: BlackHoleUniform;

const PI: f32 = 3.14159265359;
const ESCAPE_RADIUS: f32 = 200.0;

// High quality hash functions
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash3(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash2(p: vec2<f32>) -> vec2<f32> {
    let k = vec2<f32>(0.3183099, 0.3678794);
    var pp = p * k + k.yx;
    return fract(16.0 * k * fract(pp.x * pp.y * (pp.x + pp.y)));
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0); // Quintic interpolation
    return mix(mix(hash(i), hash(i + vec2<f32>(1.0, 0.0)), u.x),
               mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

fn fbm(p: vec2<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pp = p;
    for (var i = 0; i < octaves; i++) {
        value += amplitude * noise(pp);
        pp *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// Schwarzschild metric - much more numerically stable
fn schwarzschild_r(pos: vec3<f32>) -> f32 {
    return length(pos);
}

fn horizon_radius(a: f32) -> f32 {
    return 2.0;  // Schwarzschild radius = 2M (using M=1)
}

// Photon acceleration in Schwarzschild spacetime
// Using the effective potential approach for null geodesics
fn schwarzschild_acceleration(pos: vec3<f32>, vel: vec3<f32>) -> vec3<f32> {
    let r = length(pos);
    let r2 = r * r;
    let r3 = r2 * r;
    
    // Angular momentum
    let L = cross(pos, vel);
    let L2 = dot(L, L);
    
    // Schwarzschild geodesic equation for light
    // d²r/dλ² = -M/r² + L²(r-3M)/r⁴  (for M=1)
    // In 3D: a = -1.5 * L² / r⁵ * pos
    let accel = -1.5 * L2 / (r3 * r2) * pos;
    
    return accel;
}

fn adaptive_step(r: f32, r_horizon: f32) -> f32 {
    let dist = r - r_horizon;
    if (dist < 0.3) { return 0.008; }
    if (dist < 0.6) { return 0.015; }
    if (dist < 1.5) { return 0.025; }
    if (dist < 3.0) { return 0.04; }
    if (dist < 8.0) { return 0.06; }
    if (r < 30.0) { return 0.1; }
    return min(0.2, r * 0.01);
}

fn rk4_step(pos: vec3<f32>, vel: vec3<f32>, dt: f32) -> array<vec3<f32>, 2> {
    let k1_v = schwarzschild_acceleration(pos, vel);
    let k1_p = vel;
    let k2_v = schwarzschild_acceleration(pos + k1_p * dt * 0.5, vel + k1_v * dt * 0.5);
    let k2_p = vel + k1_v * dt * 0.5;
    let k3_v = schwarzschild_acceleration(pos + k2_p * dt * 0.5, vel + k2_v * dt * 0.5);
    let k3_p = vel + k2_v * dt * 0.5;
    let k4_v = schwarzschild_acceleration(pos + k3_p * dt, vel + k3_v * dt);
    let k4_p = vel + k3_v * dt;
    return array<vec3<f32>, 2>(
        pos + (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) * dt / 6.0,
        vel + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * dt / 6.0
    );
}

// Physically-based blackbody color
fn blackbody_color(temp: f32) -> vec3<f32> {
    let t = clamp(temp, 1000.0, 40000.0) / 100.0;
    var color: vec3<f32>;
    
    // Red
    if (t <= 66.0) {
        color.r = 1.0;
    } else {
        color.r = clamp(1.29294 * pow(t - 60.0, -0.1332), 0.0, 1.0);
    }
    
    // Green
    if (t <= 66.0) {
        color.g = clamp(0.39008 * log(t) - 0.63184, 0.0, 1.0);
    } else {
        color.g = clamp(1.12989 * pow(t - 60.0, -0.0755), 0.0, 1.0);
    }
    
    // Blue
    if (t >= 66.0) {
        color.b = 1.0;
    } else if (t <= 19.0) {
        color.b = 0.0;
    } else {
        color.b = clamp(0.54320 * log(t - 10.0) - 1.19625, 0.0, 1.0);
    }
    
    return color;
}

// Thin accretion disk sampling - Gargantua style
fn sample_thin_disk(hit_r: f32, hit_phi: f32, ray_dir: vec3<f32>, is_backside: bool) -> vec4<f32> {
    let r_safe = max(hit_r, blackhole.disk_inner);
    
    // Keplerian orbital velocity
    let v_kepler = sqrt(1.0 / r_safe);
    let disk_vel = v_kepler * vec3<f32>(-sin(hit_phi), cos(hit_phi), 0.0);
    
    // Relativistic Doppler + beaming
    let v2 = min(dot(disk_vel, disk_vel), 0.9);
    let gamma = 1.0 / sqrt(1.0 - v2);
    let cos_angle = dot(normalize(disk_vel), normalize(ray_dir));
    let doppler = gamma * (1.0 - sqrt(v2) * cos_angle);
    let grav_redshift = sqrt(max(1.0 - 2.0 / r_safe, 0.05));
    let freq_shift = clamp(doppler * grav_redshift, 0.3, 3.0);
    
    // Temperature profile - hot inner (white), cool outer (orange/red)
    let r_ratio = blackhole.disk_inner / r_safe;
    let base_temp = 6000.0;
    let local_temp = base_temp * pow(r_ratio, 0.65);
    let observed_temp = clamp(local_temp / freq_shift, 2000.0, 15000.0);
    
    var color = blackbody_color(observed_temp);
    
    // Doppler beaming
    let beam_factor = pow(1.0 / freq_shift, 2.5);
    
    // Radial brightness profile
    let r_norm = (hit_r - blackhole.disk_inner) / (blackhole.disk_outer - blackhole.disk_inner);
    let radial_profile = pow(r_ratio, 2.0) * exp(-r_norm * 1.2);
    
    // Subtle turbulence
    let turb = 0.92 + 0.08 * fbm(vec2<f32>(hit_phi * 5.0, log(r_safe) * 3.0), 3);
    
    var intensity = beam_factor * radial_profile * turb;
    
    // Photon ring - bright thin ring at ISCO  
    let ring_dist = abs(hit_r - blackhole.disk_inner * 1.01);
    intensity += exp(-ring_dist * ring_dist * 50.0) * 1.5;
    
    // Backside dimmer
    if (is_backside) {
        intensity *= 0.7;
    }
    
    // Edge fades
    let inner_fade = smoothstep(blackhole.disk_inner * 0.995, blackhole.disk_inner * 1.01, hit_r);
    let outer_fade = 1.0 - smoothstep(blackhole.disk_outer * 0.5, blackhole.disk_outer * 0.95, hit_r);
    
    // Less aggressive brightness multiplier
    color *= intensity * inner_fade * outer_fade * 2.0;
    
    return vec4<f32>(color, inner_fade * outer_fade);
}

// 3D noise for volumetric effects
fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    
    let n = i.x + i.y * 157.0 + i.z * 113.0;
    let a = hash(vec2<f32>(n, 0.0));
    let b = hash(vec2<f32>(n + 1.0, 0.0));
    let c = hash(vec2<f32>(n + 157.0, 0.0));
    let d = hash(vec2<f32>(n + 158.0, 0.0));
    let e = hash(vec2<f32>(n + 113.0, 0.0));
    let ff = hash(vec2<f32>(n + 114.0, 0.0));
    let g = hash(vec2<f32>(n + 270.0, 0.0));
    let h = hash(vec2<f32>(n + 271.0, 0.0));
    
    return mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
               mix(mix(e, ff, u.x), mix(g, h, u.x), u.y), u.z);
}

fn fbm3d(p: vec3<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pos = p;
    for (var i = 0; i < octaves; i++) {
        value += amplitude * noise3d(pos);
        pos *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// Sample dust cloud at a 3D position
fn sample_dust_cloud(pos: vec3<f32>) -> vec4<f32> {
    let r = length(pos);
    
    // Dust exists in a torus around the disk
    let disk_height = abs(pos.z);
    let disk_r = length(pos.xy);
    
    // Dust only in specific regions - sparse wisps
    let height_falloff = exp(-disk_height * disk_height / 8.0);
    let radial_falloff = smoothstep(15.0, 25.0, disk_r) * (1.0 - smoothstep(45.0, 60.0, disk_r));
    
    if (height_falloff < 0.1 || radial_falloff < 0.1) {
        return vec4<f32>(0.0);
    }
    
    // Sparse turbulent dust
    let noise_pos = pos * 0.08;
    let density = fbm3d(noise_pos, 3);
    // Only show dust where noise is high (sparse clouds)
    let sparse = smoothstep(0.5, 0.7, density);
    let dust_density = sparse * height_falloff * radial_falloff;
    
    // Dust is illuminated by the disk - subtle warm glow
    let illumination = 1.0 / (1.0 + disk_r * 0.1);
    let dust_color = vec3<f32>(1.0, 0.5, 0.2) * illumination * 0.02;
    
    return vec4<f32>(dust_color, dust_density * 0.005);
}

// Sample hot gas wisps near the black hole
fn sample_hot_gas(pos: vec3<f32>) -> vec3<f32> {
    let r = length(pos);
    
    // Hot gas close to the black hole
    if (r < 5.0 || r > 12.0) {
        return vec3<f32>(0.0);
    }
    
    let gas_noise = fbm3d(pos * 0.4, 3);
    let proximity = exp(-(r - 7.0) * (r - 7.0) / 10.0);
    
    // Spiral pattern
    let phi = atan2(pos.y, pos.x);
    let spiral = sin(phi * 4.0 + r * 0.8) * 0.5 + 0.5;
    
    // Only show where noise is high (sparse wisps)
    let sparse = smoothstep(0.55, 0.7, gas_noise);
    let gas_density = sparse * proximity * spiral * 0.3;
    
    // Hot blue-white gas - very subtle
    let gas_temp = 10000.0 + gas_noise * 5000.0;
    return blackbody_color(gas_temp) * gas_density * 0.015;
}

// Smooth star field with proper anti-aliasing
fn sample_stars(dir: vec3<f32>) -> vec3<f32> {
    let d = normalize(dir);
    
    // Convert to spherical coordinates
    let theta = atan2(d.z, d.x);
    let phi = asin(clamp(d.y, -1.0, 1.0));
    
    var stars = vec3<f32>(0.0);
    
    // Multiple star layers with different densities
    for (var layer = 0; layer < 4; layer++) {
        let scale = 80.0 + f32(layer) * 40.0;
        let uv = vec2<f32>(theta, phi) * scale;
        let cell = floor(uv);
        let frac_uv = fract(uv);
        
        // Check neighboring cells for smooth stars
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor = cell + vec2<f32>(f32(dx), f32(dy));
                let h = hash(neighbor + f32(layer) * 127.0);
                
                if (h > 0.97) {
                    // Star position within cell
                    let star_pos = hash2(neighbor * 1.73 + f32(layer) * 31.0);
                    let offset = vec2<f32>(f32(dx), f32(dy)) + star_pos - frac_uv;
                    let dist = length(offset);
                    
                    // Smooth circular star with gaussian falloff
                    let star_size = 0.02 + (h - 0.97) * 0.3;
                    let brightness = exp(-dist * dist / (star_size * star_size * 2.0));
                    
                    // Star color based on temperature
                    let star_temp = 3000.0 + hash(neighbor * 2.37) * 25000.0;
                    let star_color = blackbody_color(star_temp);
                    
                    let star_intensity = brightness * (0.3 + (h - 0.97) * 10.0);
                    stars += star_color * star_intensity * (1.0 - f32(layer) * 0.15);
                }
            }
        }
    }
    
    // Rich colorful nebula background
    let nebula_uv = vec2<f32>(theta * 1.5, phi * 2.0);
    let nebula1 = fbm(nebula_uv * 0.8, 6);
    let nebula2 = fbm(nebula_uv * 1.5 + vec2<f32>(5.0, 3.0), 5);
    let nebula3 = fbm(nebula_uv * 2.5 + vec2<f32>(10.0, 7.0), 4);
    
    // Galactic plane concentration
    let galactic_plane = exp(-phi * phi * 4.0);
    
    // Multiple nebula colors - subtle
    let red_nebula = vec3<f32>(0.8, 0.2, 0.1) * nebula1 * nebula1;
    let blue_nebula = vec3<f32>(0.1, 0.3, 0.9) * nebula2 * (1.0 - galactic_plane * 0.5);
    let gold_nebula = vec3<f32>(1.0, 0.7, 0.2) * nebula3 * galactic_plane;
    
    let combined_nebula = (red_nebula + blue_nebula * 0.5 + gold_nebula) * 0.04;
    stars += combined_nebula;
    
    // Dark dust lanes in the galactic plane
    let dust_lane = fbm(nebula_uv * 3.0, 4) * galactic_plane;
    stars *= 1.0 - dust_lane * 0.4;
    
    return stars;
}

fn trace_ray(origin: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    var pos = origin;
    var vel = normalize(direction);
    var color = vec3<f32>(0.0);
    var alpha = 0.0;
    
    let r_horizon = 2.0;  // Schwarzschild radius
    let r_photon = 3.0;   // Photon sphere at 3M
    let max_steps = i32(blackhole.max_steps);
    
    var total_disk_hits = 0;
    var prev_z_sign = sign(pos.z);
    var min_r = 1000.0;  // Track minimum radius reached
    var accumulated_gas = vec3<f32>(0.0);
    var accumulated_dust = vec3<f32>(0.0);
    var dust_opacity = 0.0;
    
    for (var i = 0; i < max_steps; i++) {
        if (alpha > 0.99) { break; }
        
        let r = length(pos);
        min_r = min(min_r, r);
        
        // Fell into black hole - anything inside photon sphere is captured
        if (r < 3.0) {
            alpha = 1.0;
            break;
        }
        
        // Escaped to infinity
        if (r > ESCAPE_RADIUS) {
            let bg = sample_stars(vel);
            color = mix(color, bg, 1.0 - alpha);
            break;
        }
        
        // Sample volumetric effects every few steps for performance
        if (i % 4 == 0 && r < 80.0) {
            // Hot gas near black hole
            let gas = sample_hot_gas(pos);
            accumulated_gas += gas * (1.0 - alpha);
            
            // Dust clouds
            let dust = sample_dust_cloud(pos);
            accumulated_dust += dust.rgb * (1.0 - dust_opacity);
            dust_opacity += dust.a * (1.0 - dust_opacity);
        }
        
        let dt = adaptive_step(r, r_horizon);
        let old_pos = pos;
        
        let result = rk4_step(pos, vel, dt);
        pos = result[0];
        vel = normalize(result[1]);
        
        // Check disk plane crossing (z = 0)
        let curr_z_sign = sign(pos.z);
        let crossed_disk = prev_z_sign != curr_z_sign && prev_z_sign != 0.0;
        prev_z_sign = curr_z_sign;
        
        if (crossed_disk && total_disk_hits < 3) {
            // Interpolate to find exact crossing point
            let t = -old_pos.z / (pos.z - old_pos.z + 0.00001);
            let hit_pos = mix(old_pos, pos, clamp(t, 0.0, 1.0));
            let hit_r = length(hit_pos.xy);
            
            if (hit_r >= blackhole.disk_inner && hit_r <= blackhole.disk_outer) {
                let hit_phi = atan2(hit_pos.y, hit_pos.x);
                
                // Determine if this is front or back of disk (lensed image)
                let is_backside = total_disk_hits > 0;
                
                let disk_color = sample_thin_disk(hit_r, hit_phi, vel, is_backside);
                let opacity = disk_color.a * 0.9;
                
                color += disk_color.rgb * (1.0 - alpha);
                alpha += opacity * (1.0 - alpha);
                
                total_disk_hits++;
            }
        }
    }
    
    // Add volumetric contributions - very subtle
    color += accumulated_gas * 0.5;
    color += accumulated_dust * clamp(dust_opacity, 0.0, 0.05);
    
    return color;
}

fn generate_ray(pixel: vec2<f32>) -> array<vec3<f32>, 2> {
    let ndc = (pixel / camera.resolution) * 2.0 - 1.0;
    let clip_near = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    let clip_far = vec4<f32>(ndc.x, -ndc.y, 1.0, 1.0);
    var world_near = camera.inv_view_proj * clip_near;
    var world_far = camera.inv_view_proj * clip_far;
    world_near /= world_near.w;
    world_far /= world_far.w;
    return array<vec3<f32>, 2>(camera.position.xyz, normalize(world_far.xyz - world_near.xyz));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) { return; }
    
    // Stratified sampling for AA
    let sample_idx = blackhole.sample_index;
    let grid_size = sqrt(blackhole.total_samples);
    let sx = (sample_idx % grid_size) / grid_size;
    let sy = floor(sample_idx / grid_size) / grid_size;
    
    // Jittered sample position
    let jitter = hash2(vec2<f32>(f32(global_id.x) + sample_idx * 100.0, f32(global_id.y) + sample_idx * 137.0));
    let offset = vec2<f32>(sx + jitter.x / grid_size, sy + jitter.y / grid_size);
    
    let pixel = vec2<f32>(f32(global_id.x), f32(global_id.y)) + offset;
    
    let ray = generate_ray(pixel);
    let color = trace_ray(ray[0], ray[1]);
    
    // Accumulate samples
    let prev = textureLoad(output_texture, vec2<i32>(global_id.xy));
    let weight = 1.0 / (sample_idx + 1.0);
    let accumulated = mix(prev.rgb, color, weight);
    
    textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(accumulated, 1.0));
}

// Tone mapping pass
@compute @workgroup_size(8, 8)
fn tonemap(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) { return; }
    
    var color = textureLoad(output_texture, vec2<i32>(global_id.xy)).rgb;
    
    // Slight bloom simulation
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let bloom = max(luminance - 1.0, 0.0) * 0.15;
    color += bloom;
    
    // ACES filmic tone mapping
    let x = color;
    var mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    
    // Gamma correction
    mapped = pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    
    textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(mapped, 1.0));
}
"#;

async fn run_gpu_render(
    width: u32,
    height: u32,
    spin: f32,
    camera_distance: f32,
    camera_theta: f32,
    camera_phi: f32,
    fov: f32,
    quality: Quality,
    output_path: &str,
) {
    println!("Initializing GPU...");
    
    // Try all backends - in WSL2, DX12 or GL may work better than Vulkan
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        dx12_shader_compiler: Dx12Compiler::Fxc,
        ..Default::default()
    });
    
    // List available adapters
    println!("Available GPUs:");
    for adapter in instance.enumerate_adapters(Backends::all()) {
        let info = adapter.get_info();
        println!("  - {} ({:?})", info.name, info.backend);
    }
    println!();
    
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find GPU adapter. Make sure NVIDIA drivers are installed.");
    
    let info = adapter.get_info();
    println!("Selected GPU: {} ({:?})", info.name, info.backend);
    
    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: Some("GPU Renderer"),
                required_features: Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                required_limits: Limits::default(),
                memory_hints: MemoryHints::Performance,
            },
            None,
        )
        .await
        .expect("Failed to create device");
    
    // Create render texture (RGBA32Float for accumulation)
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Render Texture"),
        size: Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&TextureViewDescriptor::default());
    
    // Camera setup
    let aspect = width as f32 / height as f32;
    let cam_x = camera_distance * camera_phi.cos() * camera_theta.sin();
    let cam_y = camera_distance * camera_phi.sin();
    let cam_z = camera_distance * camera_phi.cos() * camera_theta.cos();
    let origin = Vec3::new(cam_x, cam_y, cam_z);
    let view = Mat4::look_at_rh(origin, Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(fov, aspect, 0.1, 1000.0);
    let inv_vp = (proj * view).inverse();
    
    let camera_uniform = CameraUniform {
        position: [origin.x, origin.y, origin.z, 0.0],
        inv_view_proj: inv_vp.to_cols_array(),
        resolution: [width as f32, height as f32],
        fov,
        _padding: 0.0,
    };
    
    let camera_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[camera_uniform]),
        usage: BufferUsages::UNIFORM,
    });
    
    let blackhole_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("BlackHole Buffer"),
        size: std::mem::size_of::<BlackHoleUniform>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    // Create shader
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("GPU Render Shader"),
        source: ShaderSource::Wgsl(SHADER_SOURCE.into()),
    });
    
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rgba32Float,
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
    
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&texture_view) },
            BindGroupEntry { binding: 1, resource: camera_buffer.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: blackhole_buffer.as_entire_binding() },
        ],
    });
    
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let render_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    
    let tonemap_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Tonemap Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("tonemap"),
        compilation_options: Default::default(),
        cache: None,
    });
    
    let disk_inner = 6.0;  // ISCO for Schwarzschild = 6M
    let disk_outer = 30.0; // Outer disk radius
    let samples = quality.samples();
    let max_steps = quality.max_steps();
    
    println!("Rendering {} samples at {}x{}...", samples, width, height);
    println!("Max steps per ray: {}", max_steps);
    
    let start = std::time::Instant::now();
    
    // Render each sample
    for sample in 0..samples {
        let blackhole_uniform = BlackHoleUniform {
            spin,
            mass: 1.0,
            disk_inner,
            disk_outer,
            time: 0.0,
            max_steps: max_steps as f32,
            sample_index: sample as f32,
            total_samples: samples as f32,
            _padding: [0.0; 4],
        };
        
        queue.write_buffer(&blackhole_buffer, 0, bytemuck::cast_slice(&[blackhole_uniform]));
        
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Render Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&render_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
        }
        
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(Maintain::Wait);
        
        print!("\rSample {}/{}", sample + 1, samples);
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }
    println!();
    
    // Apply tone mapping
    {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Tonemap Encoder"),
        });
        
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Tonemap Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&tonemap_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
        }
        
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(Maintain::Wait);
    }
    
    let render_time = start.elapsed();
    println!("Render time: {:.2}s", render_time.as_secs_f32());
    
    // Read back pixels
    println!("Saving image...");
    
    let bytes_per_pixel = 16; // RGBA32Float = 4 * 4 bytes
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;
    
    let output_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Output Buffer"),
        size: (padded_bytes_per_row * height) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Copy Encoder"),
    });
    
    encoder.copy_texture_to_buffer(
        ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        ImageCopyBuffer {
            buffer: &output_buffer,
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        Extent3d { width, height, depth_or_array_layers: 1 },
    );
    
    queue.submit(std::iter::once(encoder.finish()));
    
    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(Maintain::Wait);
    rx.recv().unwrap().unwrap();
    
    let data = buffer_slice.get_mapped_range();
    
    // Convert to 8-bit RGBA
    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            let offset = (y * padded_bytes_per_row / 4 + x * 4) as usize;
            let float_data: &[f32] = bytemuck::cast_slice(&data);
            
            let r = (float_data[offset].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (float_data[offset + 1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (float_data[offset + 2].clamp(0.0, 1.0) * 255.0) as u8;
            
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }
    
    drop(data);
    output_buffer.unmap();
    
    img.save(output_path).expect("Failed to save image");
    println!("Saved to: {}", output_path);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    let mut width = 3840u32;
    let mut height = 2160u32;
    let mut spin = 0.9f32;
    let mut quality = Quality::Ultra;
    let mut camera_distance = 30.0f32;
    let mut camera_theta = 0.0f32;
    let mut camera_phi = 1.31f32;  // ~75 degrees
    let fov = 35.0_f32.to_radians();
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-w" | "--width" => {
                width = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3840);
                i += 1;
            }
            "-h" | "--height" => {
                height = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2160);
                i += 1;
            }
            "-s" | "--spin" => {
                spin = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0.9);
                i += 1;
            }
            "-q" | "--quality" => {
                quality = match args.get(i + 1).map(|s| s.as_str()) {
                    Some("preview") => Quality::Preview,
                    Some("high") => Quality::High,
                    Some("ultra") => Quality::Ultra,
                    Some("insane") => Quality::Insane,
                    _ => Quality::Ultra,
                };
                i += 1;
            }
            "-d" | "--distance" => {
                camera_distance = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(25.0);
                i += 1;
            }
            "-t" | "--theta" => {
                camera_theta = args.get(i + 1).and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0).to_radians();
                i += 1;
            }
            "-p" | "--phi" => {
                camera_phi = args.get(i + 1).and_then(|s| s.parse::<f32>().ok()).unwrap_or(20.0).to_radians();
                i += 1;
            }
            "--help" => {
                println!("Kerr Black Hole GPU Renderer");
                println!();
                println!("Usage: gpu_render [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -w, --width <WIDTH>      Output width (default: 3840)");
                println!("  -h, --height <HEIGHT>    Output height (default: 2160)");
                println!("  -s, --spin <SPIN>        Black hole spin 0-0.99 (default: 0.9)");
                println!("  -q, --quality <QUALITY>  preview|high|ultra|insane (default: ultra)");
                println!("  -d, --distance <DIST>    Camera distance (default: 25)");
                println!("  -t, --theta <DEG>        Camera horizontal angle (default: 0)");
                println!("  -p, --phi <DEG>          Camera vertical angle (default: 20)");
                println!();
                println!("Quality presets (GPU-accelerated):");
                println!("  preview: 500 steps, 1 sample   (~1 sec at 4K)");
                println!("  high:    2000 steps, 4 samples (~10 sec at 4K)");
                println!("  ultra:   5000 steps, 16 samples (~1 min at 4K)");
                println!("  insane:  10000 steps, 64 samples (~5 min at 4K)");
                return;
            }
            _ => {}
        }
        i += 1;
    }
    
    println!("Kerr Black Hole GPU Renderer");
    println!("============================");
    println!("Resolution: {}x{}", width, height);
    println!("Spin: {:.2}", spin);
    println!("Quality: {} steps, {} samples", quality.max_steps(), quality.samples());
    println!("ISCO: {:.3}M", isco_radius(spin));
    println!("Event Horizon: {:.3}M", horizon_radius(spin));
    println!();
    
    let output_path = format!(
        "blackhole_gpu_{}x{}_spin{:.2}_q{}.png",
        width, height, spin,
        match quality {
            Quality::Preview => "preview",
            Quality::High => "high",
            Quality::Ultra => "ultra",
            Quality::Insane => "insane",
        }
    );
    
    pollster::block_on(run_gpu_render(
        width, height, spin,
        camera_distance, camera_theta, camera_phi, fov,
        quality, &output_path
    ));
}
