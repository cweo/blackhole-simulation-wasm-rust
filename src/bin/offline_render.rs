//! Offline high-quality black hole renderer
//! Produces publication-quality images with extreme ray tracing settings

use std::f32::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use glam::{Mat4, Vec3, Vec4};
use image::{ImageBuffer, Rgb};
use rayon::prelude::*;

// Quality presets
#[derive(Clone, Copy)]
pub enum Quality {
    Preview,    // Fast preview
    High,       // Good quality
    Ultra,      // Very high quality
    Insane,     // Maximum quality (slow)
}

impl Quality {
    fn max_steps(&self) -> i32 {
        // Fixed at 550 to match WGSL (use --steps to override)
        550
    }
    
    fn samples_per_pixel(&self) -> u32 {
        match self {
            Quality::Preview => 1,
            Quality::High => 4,
            Quality::Ultra => 16,
            Quality::Insane => 64,
        }
    }
    
    fn escape_radius(&self) -> f32 {
        // Match WGSL constant
        100.0
    }
}

#[derive(Clone, Copy)]
struct RenderParams {
    width: u32,
    height: u32,
    spin: f32,
    camera_distance: f32,
    camera_theta: f32,
    camera_phi: f32,
    fov: f32,
    quality: Quality,
    max_steps: i32,
    disk_inner: f32,
    disk_outer: f32,
    time: f32,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            width: 3840,
            height: 2160,
            spin: 0.9,
            camera_distance: 20.0,                      // Match web: 20.0
            camera_theta: std::f32::consts::PI,         // Match web: PI (180 degrees)
            camera_phi: -79.0_f32.to_radians(),         // Match web: -79 degrees
            fov: 60.0_f32.to_radians(),
            quality: Quality::Ultra,
            max_steps: 550,                             // Match WGSL MAX_STEPS
            disk_inner: 0.0, // Will be calculated
            disk_outer: 20.0,
            time: 0.0,
        }
    }
}

// Simple hash functions
fn hash(p: [f32; 2]) -> f32 {
    let p3 = [
        (p[0] * 0.1031).fract(),
        (p[1] * 0.1031).fract(),
        (p[0] * 0.1031).fract(),
    ];
    let dot = p3[0] * (p3[1] + 33.33) + p3[1] * (p3[2] + 33.33) + p3[2] * (p3[0] + 33.33);
    ((p3[0] + p3[1]) * (p3[2] + dot)).fract()
}

fn hash2(p: [f32; 2]) -> [f32; 2] {
    let k = [0.3183099, 0.3678794];
    let pp = [p[0] * k[0] + k[1], p[1] * k[1] + k[0]];
    [
        (16.0 * k[0] * (pp[0] * pp[1] * (pp[0] + pp[1])).fract()).fract(),
        (16.0 * k[1] * (pp[0] * pp[1] * (pp[0] + pp[1])).fract()).fract(),
    ]
}

fn noise(p: [f32; 2]) -> f32 {
    let i = [p[0].floor(), p[1].floor()];
    let f = [p[0].fract(), p[1].fract()];
    let u = [
        f[0] * f[0] * (3.0 - 2.0 * f[0]),
        f[1] * f[1] * (3.0 - 2.0 * f[1]),
    ];
    
    let a = hash(i);
    let b = hash([i[0] + 1.0, i[1]]);
    let c = hash([i[0], i[1] + 1.0]);
    let d = hash([i[0] + 1.0, i[1] + 1.0]);
    
    a + (b - a) * u[0] + (c - a) * u[1] + (a - b - c + d) * u[0] * u[1]
}

fn fbm(mut p: [f32; 2], octaves: i32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 0.5;
    for _ in 0..octaves {
        value += amplitude * noise(p);
        p = [p[0] * 2.0, p[1] * 2.0];
        amplitude *= 0.5;
    }
    value
}

fn voronoi(p: [f32; 2]) -> f32 {
    let n = [p[0].floor(), p[1].floor()];
    let f = [p[0].fract(), p[1].fract()];
    let mut md = 8.0_f32;
    
    for j in -1..=1 {
        for i in -1..=1 {
            let g = [i as f32, j as f32];
            let o = hash2([n[0] + g[0], n[1] + g[1]]);
            let r = [g[0] + o[0] - f[0], g[1] + o[1] - f[1]];
            let d = r[0] * r[0] + r[1] * r[1];
            md = md.min(d);
        }
    }
    md.sqrt()
}

// Kerr metric functions
fn kerr_r(pos: Vec3, a: f32) -> f32 {
    let rho2 = pos.dot(pos);
    let a2 = a * a;
    let b = rho2 - a2;
    let c = -a2 * pos.z * pos.z;
    let disc = (b * b - 4.0 * c).max(0.0).sqrt();
    ((b + disc) * 0.5).max(0.001).sqrt()
}

fn horizon_radius(a: f32) -> f32 {
    1.0 + (1.0 - a * a).max(0.0).sqrt()
}

fn isco_radius(spin: f32) -> f32 {
    let a = spin;
    let z1 = 1.0 + (1.0 - a * a).powf(1.0 / 3.0) * ((1.0 + a).powf(1.0 / 3.0) + (1.0 - a).powf(1.0 / 3.0));
    let z2 = (3.0 * a * a + z1 * z1).sqrt();
    3.0 + z2 - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt()
}

// Match WGSL photon_sphere_radius
fn photon_sphere_radius(a: f32) -> f32 {
    1.5 * (1.0 + (1.0 - a * a * 0.5).sqrt())
}

fn frame_drag_omega(r: f32, z: f32, a: f32) -> f32 {
    let r2 = r * r;
    let a2 = a * a;
    let cos_theta2 = z * z / r2.max(0.001);
    let sigma = r2 + a2 * cos_theta2;
    let delta = r2 - 2.0 * r + a2;
    let big_a = (r2 + a2) * (r2 + a2) - a2 * delta * (1.0 - cos_theta2);
    2.0 * a * r / big_a.max(0.001)
}

fn kerr_acceleration(pos: Vec3, vel: Vec3, a: f32) -> Vec3 {
    let r = kerr_r(pos, a);
    let r2 = r * r;
    let a2 = a * a;
    let l = pos.cross(vel);
    let h2 = l.dot(l);
    let r5 = r2 * r2 * r;
    let mut accel = -1.5 * h2 / r5.max(0.001) * pos;
    let omega = frame_drag_omega(r, pos.z, a);
    accel += omega * a * Vec3::new(-vel.y, vel.x, 0.0) * 2.0;
    let sigma = r2 + a2 * (pos.z * pos.z) / r2.max(0.001);
    accel.z += a2 * pos.z / (sigma * r2).max(0.001) * (vel.x * vel.x + vel.y * vel.y);
    accel
}

fn adaptive_step(r: f32, r_horizon: f32) -> f32 {
    // Match WGSL adaptive_step function
    let dist = r - r_horizon;
    if dist < 0.3 { return 0.015; }
    if dist < 1.0 { return 0.03; }
    if dist < 3.0 { return 0.05; }
    if r < 15.0 { return 0.07; }
    (0.12_f32).min(r * 0.015)
}

fn rk4_step(pos: Vec3, vel: Vec3, a: f32, dt: f32) -> (Vec3, Vec3) {
    let k1_v = kerr_acceleration(pos, vel, a);
    let k1_p = vel;
    let k2_v = kerr_acceleration(pos + k1_p * dt * 0.5, vel + k1_v * dt * 0.5, a);
    let k2_p = vel + k1_v * dt * 0.5;
    let k3_v = kerr_acceleration(pos + k2_p * dt * 0.5, vel + k2_v * dt * 0.5, a);
    let k3_p = vel + k2_v * dt * 0.5;
    let k4_v = kerr_acceleration(pos + k3_p * dt, vel + k3_v * dt, a);
    let k4_p = vel + k3_v * dt;
    (
        pos + (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) * dt / 6.0,
        vel + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * dt / 6.0,
    )
}

fn blackbody_color(temp: f32) -> Vec3 {
    let t = temp.clamp(800.0, 40000.0);
    let r = if t < 6600.0 { 1.0 } else { (1.29 * ((t - 6000.0) / 1000.0).powf(-0.1332)).clamp(0.0, 1.0) };
    let g = if t < 6600.0 {
        (0.39 * (t / 1000.0).ln() - 0.19).clamp(0.0, 1.0)
    } else {
        (1.29 * ((t - 6000.0) / 1000.0).powf(-0.0755)).clamp(0.0, 1.0)
    };
    let b = if t < 1900.0 {
        0.0
    } else if t < 6600.0 {
        (0.39 * (t / 1000.0 - 1.0).ln() - 0.35).clamp(0.0, 1.0)
    } else {
        1.0
    };
    Vec3::new(r, g, b)
}

fn hot_spots(r: f32, phi: f32, time: f32, disk_inner: f32) -> f32 {
    let mut spots = 0.0;
    // Match WGSL: 5 spots with different spacing
    for i in 0..5 {
        let spot_r = disk_inner * (1.2 + i as f32 * 0.4);
        let orbit_speed = 1.0 / (spot_r.sqrt() * spot_r);
        let spot_phi = i as f32 * 1.256 + time * orbit_speed;
        let dr = r - spot_r;
        let dphi = (phi - spot_phi).sin().atan2((phi - spot_phi).cos());
        let dist2 = dr * dr + (dphi * spot_r) * (dphi * spot_r);
        let spot_size = 0.3 + hash([i as f32, 0.0]) * 0.4;
        spots += (-dist2 / (spot_size * spot_size)).exp() * (0.5 + hash([i as f32, 1.0]) * 0.5);
    }
    spots
}

fn particle_density(r: f32, phi: f32, time: f32) -> f32 {
    // Match WGSL particle_density
    let orbit_phase = phi - time * 0.2 / r.max(1.0).sqrt();
    let p = [orbit_phase * 3.0, r.ln() * 4.0];
    let v = voronoi([p[0] * 3.0, p[1] * 3.0]);
    // Create clumpy structure using smoothstep
    let clumps = smoothstep(0.3, 0.0, v);
    // Add smaller scale detail
    let fine_detail = fbm([phi * 20.0, r * 2.0 + time * 0.1], 6);
    clumps * 0.5 + fine_detail * 0.3 + 0.4
}

// Match WGSL dust_lanes
fn dust_lanes(r: f32, phi: f32) -> f32 {
    let spiral_arm = (phi * 2.0 + r.ln() * 3.0).sin() * 0.5 + 0.5;
    let dust = fbm([phi * 5.0, r * 0.5], 6) * spiral_arm;
    1.0 - dust * 0.3
}

fn sample_disk(hit_r: f32, hit_phi: f32, ray_dir: Vec3, depth: f32, params: &RenderParams) -> Vec4 {
    // Match WGSL sample_disk function
    let r_safe = hit_r.max(params.disk_inner);
    let v_kepler = (1.0 / r_safe).sqrt();
    let disk_vel = v_kepler * Vec3::new(-hit_phi.sin(), hit_phi.cos(), 0.0);
    
    let v2 = disk_vel.dot(disk_vel).min(0.95);
    let gamma = 1.0 / (1.0 - v2).sqrt();
    let photon_dir = ray_dir.normalize();
    let cos_angle = disk_vel.normalize().dot(photon_dir);
    let doppler = gamma * (1.0 - v2.sqrt() * cos_angle);
    let grav_redshift = (1.0 - 2.0 / r_safe).max(0.01).sqrt();
    let freq_shift = doppler * grav_redshift;
    
    // Temperature with local variations (match WGSL)
    let r_ratio = params.disk_inner / r_safe;
    let base_temp = 5500.0;
    let temp_variation = 1.0 + 0.3 * fbm([hit_phi * 10.0, hit_r + params.time * 0.2], 6);
    let local_temp = base_temp * r_ratio.powf(0.75) * temp_variation;
    let observed_temp = local_temp / freq_shift;
    
    let mut color = blackbody_color(observed_temp);
    
    // Relativistic beaming
    let beam_factor = (1.0 / freq_shift.max(0.1)).powf(3.5);
    
    // Radial profile (match WGSL)
    let r_norm = (hit_r - params.disk_inner) / (params.disk_outer - params.disk_inner);
    let radial_profile = (-r_norm * 1.5).exp() * (1.0 - (-r_norm * 10.0).exp());
    
    // Particle density and clumping
    let particles = particle_density(hit_r, hit_phi, params.time);
    
    // Hot spots
    let spots = hot_spots(hit_r, hit_phi, params.time, params.disk_inner);
    
    // Dust absorption
    let dust = dust_lanes(hit_r, hit_phi);
    
    // Spiral density waves (match WGSL)
    let spiral_phase = hit_phi * 2.0 + r_safe.ln() * 4.0 - params.time * 0.15;
    let spiral_density = 0.8 + 0.2 * spiral_phase.sin();
    let spiral_phase2 = hit_phi * 3.0 - r_safe.ln() * 2.0 + params.time * 0.1;
    let spiral_density2 = 0.9 + 0.1 * spiral_phase2.sin();
    
    // Combine all effects
    let mut intensity = beam_factor * radial_profile * particles * dust * spiral_density * spiral_density2;
    
    // Add hot spot glow (match WGSL)
    let spot_color = blackbody_color(observed_temp * 1.5);
    color = color.lerp(spot_color, spots * 0.5);
    intensity += spots * 2.0;
    
    // Inner edge glow (matter plunging into BH) - match WGSL
    let inner_glow = (-(hit_r - params.disk_inner) * 2.0).exp() * 3.0;
    color += Vec3::new(1.0, 0.7, 0.3) * inner_glow * beam_factor;
    
    // Edge softening (match WGSL)
    let inner_fade = smoothstep(params.disk_inner * 0.95, params.disk_inner * 1.15, hit_r);
    let outer_fade = 1.0 - smoothstep(params.disk_outer * 0.7, params.disk_outer, hit_r);
    
    // Depth factor for volumetric feel
    let volume_factor = 0.7 + 0.3 * depth;
    
    color *= intensity * inner_fade * outer_fade * volume_factor * 2.0;
    
    Vec4::new(color.x, color.y, color.z, inner_fade * outer_fade * depth)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn sample_stars(dir: Vec3) -> Vec3 {
    // Match WGSL sample_stars function
    let d = dir.normalize();
    let theta = d.z.atan2(d.x);
    let phi = d.y.clamp(-1.0, 1.0).asin();
    
    let mut stars = Vec3::ZERO;
    
    // Multiple star layers with varying densities (match WGSL)
    for layer in 0..4 {
        let scale = 40.0 + layer as f32 * 25.0;
        let grid = [(theta * scale).floor(), (phi * scale).floor()];
        let cell_center = [(grid[0] + 0.5) / scale, (grid[1] + 0.5) / scale];
        let h = hash([grid[0] + layer as f32 * 137.0, grid[1] + layer as f32 * 137.0]);
        
        if h > 0.982 {
            // Star position within cell
            let star_offset = hash2([grid[0], grid[1]]);
            let star_offset = [star_offset[0] - 0.5, star_offset[1] - 0.5];
            let star_pos = [cell_center[0] + star_offset[0] * 0.8 / scale, 
                           cell_center[1] + star_offset[1] * 0.8 / scale];
            let dist = ((theta - star_pos[0] * scale / scale).powi(2) + 
                       (phi - star_pos[1] * scale / scale).powi(2)).sqrt();
            
            let brightness = ((h - 0.982) / 0.018).powf(0.4) * (1.0 - layer as f32 * 0.15);
            let star_temp = 2500.0 + h * 25000.0;
            let star_glow = (-dist * dist * scale * 2.0).exp();
            stars += blackbody_color(star_temp) * brightness * star_glow * 0.5;
        }
    }
    
    // Nebula clouds (match WGSL)
    let nebula_coord = [theta * 2.0, phi * 3.0];
    let nebula_density = fbm(nebula_coord, 6) * fbm([nebula_coord[0] * 2.0 + 5.0, nebula_coord[1] * 2.0], 5);
    let nebula_color = Vec3::new(0.1, 0.05, 0.2).lerp(
        Vec3::new(0.2, 0.1, 0.15),
        fbm([nebula_coord[0] * 0.5, nebula_coord[1] * 0.5], 4)
    );
    stars += nebula_color * nebula_density * 0.1;
    
    // Milky way band (match WGSL)
    let galactic = (-phi * phi * 8.0).exp();
    let milky = fbm([theta * 3.0, phi * 10.0], 6) * galactic;
    stars += Vec3::new(0.15, 0.12, 0.1) * milky * 0.3;
    
    stars
}

// Photon ring glow near the photon sphere (match WGSL)
fn photon_ring_glow(r: f32, r_photon: f32) -> Vec3 {
    let dist = (r - r_photon).abs();
    let glow = (-dist * 3.0).exp() * 0.3;
    Vec3::new(1.0, 0.9, 0.7) * glow
}

// Hawking radiation visualization (match WGSL)
fn hawking_radiation(pos: Vec3, r: f32, r_horizon: f32, time: f32) -> Vec3 {
    // Only emit from just outside event horizon
    let emission_zone = smoothstep(r_horizon * 0.95, r_horizon * 1.05, r) 
                       * (1.0 - smoothstep(r_horizon * 1.05, r_horizon * 2.5, r));
    
    if emission_zone < 0.01 {
        return Vec3::ZERO;
    }
    
    // Convert position to spherical for particle placement
    let theta = pos.y.atan2(pos.x);
    let phi = (pos.z / r.max(0.001)).clamp(-1.0, 1.0).asin();
    
    let mut hawking = Vec3::ZERO;
    
    // Create multiple particle streams at different angles
    for i in 0..12 {
        let stream_theta = i as f32 * PI / 6.0;
        let stream_phi = (i as f32 * 2.3).sin() * 0.8;
        
        // Particles move outward over time
        let particle_speed = 0.3 + hash([i as f32, 0.0]) * 0.4;
        let particle_phase = (time * particle_speed + hash([i as f32, 1.0])).fract();
        
        // Particle radial position
        let particle_r = r_horizon * (1.02 + particle_phase * 1.5);
        
        // Distance from this ray position to particle
        let dr = (r - particle_r).abs();
        let dtheta = (theta - stream_theta).sin().atan2((theta - stream_theta).cos()).abs();
        let dphi = (phi - stream_phi).abs();
        
        let angular_dist = (dtheta * dtheta + dphi * dphi).sqrt() * particle_r;
        let total_dist = (angular_dist * angular_dist + dr * dr).sqrt();
        
        // Particle glow
        let particle_size = 0.15 + hash([i as f32, 2.0]) * 0.1;
        let particle_glow = (-total_dist * total_dist / (particle_size * particle_size)).exp();
        
        // Particles fade as they move away
        let fade = 1.0 - particle_phase;
        
        let visual_temp = 8000.0 + hash([i as f32, 3.0]) * 12000.0;
        let mut particle_color = blackbody_color(visual_temp);
        
        // Some particles are matter, some antimatter (cool blue tint)
        if hash([i as f32, 4.0]) > 0.5 {
            particle_color = particle_color.lerp(Vec3::new(0.6, 0.8, 1.0), 0.3);
        }
        
        hawking += particle_color * particle_glow * fade * 1.5;
    }
    
    // Add continuous faint glow representing the quantum foam
    let quantum_foam = fbm([theta * 10.0 + time * 2.0, phi * 10.0], 6) * emission_zone;
    hawking += Vec3::new(0.5, 0.6, 0.9) * quantum_foam * 0.15;
    
    hawking * emission_zone
}

// Check disk crossing with thickness (match WGSL)
fn check_disk_crossing(old_pos: Vec3, new_pos: Vec3, disk_height: f32, disk_inner: f32, disk_outer: f32) -> Vec4 {
    // Check if we're within disk thickness
    let avg_z = (old_pos.z + new_pos.z) * 0.5;
    let r = ((old_pos.x + new_pos.x) * 0.5).hypot((old_pos.y + new_pos.y) * 0.5);
    
    // Disk thickness varies with radius (flared disk)
    let local_height = disk_height * (1.0 + (r - disk_inner) * 0.05);
    
    if avg_z.abs() < local_height && r >= disk_inner && r <= disk_outer {
        let hit_pos = (old_pos + new_pos) * 0.5;
        let hit_r = hit_pos.x.hypot(hit_pos.y);
        let hit_phi = hit_pos.y.atan2(hit_pos.x);
        let depth = 1.0 - avg_z.abs() / local_height;
        return Vec4::new(1.0, hit_r, hit_phi, depth);
    }
    
    // Also check z-plane crossing for thin disk component
    if old_pos.z * new_pos.z < 0.0 {
        let t = -old_pos.z / (new_pos.z - old_pos.z + 0.00001);
        let hit_pos = old_pos.lerp(new_pos, t);
        let hit_r = hit_pos.x.hypot(hit_pos.y);
        if hit_r >= disk_inner && hit_r <= disk_outer {
            return Vec4::new(1.0, hit_r, hit_pos.y.atan2(hit_pos.x), 1.0);
        }
    }
    Vec4::new(-1.0, 0.0, 0.0, 0.0)
}

fn trace_ray(origin: Vec3, direction: Vec3, params: &RenderParams) -> Vec3 {
    let mut pos = origin;
    let mut vel = direction.normalize();
    let mut color = Vec3::ZERO;
    let mut accumulated_alpha = 0.0;
    
    let a = params.spin;
    let r_horizon = horizon_radius(a);
    let r_photon = photon_sphere_radius(a);
    let escape_radius = params.quality.escape_radius();
    let max_steps = params.max_steps;
    let disk_height = 0.15;  // Match WGSL
    
    let mut in_disk_count = 0;
    
    for _ in 0..max_steps {
        if accumulated_alpha > 0.995 { break; }
        
        let r = kerr_r(pos, a);
        
        // Fell into black hole
        if r < r_horizon * 1.01 {
            color = color.lerp(Vec3::ZERO, 1.0 - accumulated_alpha);
            accumulated_alpha = 1.0;
            break;
        }
        
        // Hawking radiation near horizon
        if r > r_horizon * 0.95 && r < r_horizon * 2.5 {
            let hawking = hawking_radiation(pos, r, r_horizon, params.time);
            color += hawking * (1.0 - accumulated_alpha);
        }
        
        // Photon ring glow
        if r > r_horizon * 1.05 && r < r_photon * 1.5 {
            let ring_glow = photon_ring_glow(r, r_photon);
            color += ring_glow * (1.0 - accumulated_alpha) * 0.1;
        }
        
        // Escaped to infinity
        if r > escape_radius {
            let bg = sample_stars(vel);
            color = color.lerp(bg, 1.0 - accumulated_alpha);
            accumulated_alpha = 1.0;
            break;
        }
        
        let dt = adaptive_step(r, r_horizon);
        let old_pos = pos;
        
        let (new_pos, new_vel) = rk4_step(pos, vel, a, dt);
        pos = new_pos;
        vel = new_vel.normalize();
        
        // Check disk with thickness (match WGSL)
        let disk_hit = check_disk_crossing(old_pos, pos, disk_height, params.disk_inner, params.disk_outer);
        if disk_hit.x > 0.0 && in_disk_count < 3 {
            let disk_color = sample_disk(disk_hit.y, disk_hit.z, vel, disk_hit.w, params);
            let blend = disk_color.w * (1.0 - accumulated_alpha);
            color += Vec3::new(disk_color.x, disk_color.y, disk_color.z) * blend;
            accumulated_alpha += blend * 0.85;
            in_disk_count += 1;
        }
    }
    
    color
}

fn generate_ray(x: f32, y: f32, params: &RenderParams) -> (Vec3, Vec3) {
    let aspect = params.width as f32 / params.height as f32;
    
    // Camera position
    let cam_x = params.camera_distance * params.camera_phi.cos() * params.camera_theta.sin();
    let cam_y = params.camera_distance * params.camera_phi.sin();
    let cam_z = params.camera_distance * params.camera_phi.cos() * params.camera_theta.cos();
    let origin = Vec3::new(cam_x, cam_y, cam_z);
    
    // View matrix
    let view = Mat4::look_at_rh(origin, Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(params.fov, aspect, 0.1, 1000.0);
    let inv_vp = (proj * view).inverse();
    
    // NDC coordinates
    let ndc_x = (x / params.width as f32) * 2.0 - 1.0;
    let ndc_y = 1.0 - (y / params.height as f32) * 2.0;
    
    let clip_near = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let clip_far = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    
    let world_near = inv_vp * clip_near;
    let world_far = inv_vp * clip_far;
    
    let near = world_near.truncate() / world_near.w;
    let far = world_far.truncate() / world_far.w;
    
    let direction = (far - near).normalize();
    
    (origin, direction)
}

fn aces_tonemap(x: Vec3) -> Vec3 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    (x * (a * x + Vec3::splat(b))) / (x * (c * x + Vec3::splat(d)) + Vec3::splat(e))
}

fn render_pixel(x: u32, y: u32, params: &RenderParams) -> [u8; 3] {
    let samples = params.quality.samples_per_pixel();
    let mut color = Vec3::ZERO;
    
    for sy in 0..samples.isqrt() {
        for sx in 0..samples.isqrt() {
            let jitter_x = if samples > 1 {
                (sx as f32 + hash([x as f32 + sx as f32, y as f32 + sy as f32])) / samples.isqrt() as f32
            } else {
                0.5
            };
            let jitter_y = if samples > 1 {
                (sy as f32 + hash([x as f32 + sx as f32 + 100.0, y as f32 + sy as f32])) / samples.isqrt() as f32
            } else {
                0.5
            };
            
            let px = x as f32 + jitter_x;
            let py = y as f32 + jitter_y;
            
            let (origin, direction) = generate_ray(px, py, params);
            color += trace_ray(origin, direction, params);
        }
    }
    
    color /= samples as f32;
    
    // Tone mapping
    let mapped = aces_tonemap(color);
    
    // Gamma correction
    let r = (mapped.x.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
    let g = (mapped.y.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
    let b = (mapped.z.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
    
    [r, g, b]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    let mut params = RenderParams::default();
    
    // Parse arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-w" | "--width" => {
                params.width = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3840);
                i += 1;
            }
            "-h" | "--height" => {
                params.height = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2160);
                i += 1;
            }
            "-s" | "--spin" => {
                params.spin = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0.9);
                i += 1;
            }
            "-q" | "--quality" => {
                params.quality = match args.get(i + 1).map(|s| s.as_str()) {
                    Some("preview") => Quality::Preview,
                    Some("high") => Quality::High,
                    Some("ultra") => Quality::Ultra,
                    Some("insane") => Quality::Insane,
                    _ => Quality::Ultra,
                };
                i += 1;
            }
            "-d" | "--distance" => {
                params.camera_distance = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(20.0);
                i += 1;
            }
            "-t" | "--theta" => {
                params.camera_theta = args.get(i + 1).and_then(|s| s.parse::<f32>().ok()).unwrap_or(180.0).to_radians();
                i += 1;
            }
            "-p" | "--phi" => {
                params.camera_phi = args.get(i + 1).and_then(|s| s.parse::<f32>().ok()).unwrap_or(-79.0).to_radians();
                i += 1;
            }
            "-n" | "--steps" => {
                params.max_steps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(550);
                i += 1;
            }
            "--help" => {
                println!("Kerr Black Hole Offline Renderer");
                println!();
                println!("Usage: offline_render [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -w, --width <WIDTH>      Output width (default: 3840)");
                println!("  -h, --height <HEIGHT>    Output height (default: 2160)");
                println!("  -s, --spin <SPIN>        Black hole spin 0-0.99 (default: 0.9)");
                println!("  -q, --quality <QUALITY>  preview|high|ultra|insane (default: ultra)");
                println!("  -d, --distance <DIST>    Camera distance (default: 20)");
                println!("  -t, --theta <DEG>        Camera horizontal angle in degrees (default: 180)");
                println!("  -p, --phi <DEG>          Camera vertical angle in degrees (default: -79)");
                println!("  -n, --steps <STEPS>      Max ray tracing steps (default: 550)");
                println!();
                println!("Quality presets affect samples/pixel:");
                println!("  preview: 1 sample/pixel");
                println!("  high:    4 samples/pixel");
                println!("  ultra:   16 samples/pixel");
                println!("  insane:  64 samples/pixel");
                return;
            }
            _ => {}
        }
        i += 1;
    }
    
    // Calculate ISCO
    params.disk_inner = isco_radius(params.spin);
    
    println!("Kerr Black Hole Offline Renderer");
    println!("================================");
    println!("Resolution: {}x{}", params.width, params.height);
    println!("Spin: {:.2}", params.spin);
    println!("Max Steps: {}", params.max_steps);
    println!("Samples/pixel: {}", params.quality.samples_per_pixel());
    println!("ISCO: {:.3}M", params.disk_inner);
    println!("Event Horizon: {:.3}M", horizon_radius(params.spin));
    println!();
    
    let total_pixels = params.width * params.height;
    let progress = Arc::new(AtomicUsize::new(0));
    
    println!("Rendering...");
    
    let start = std::time::Instant::now();
    
    // Parallel rendering with rayon
    let pixels: Vec<(u32, u32, [u8; 3])> = (0..params.height)
        .into_par_iter()
        .flat_map(|y| {
            let progress = Arc::clone(&progress);
            (0..params.width).into_par_iter().map(move |x| {
                let color = render_pixel(x, y, &params);
                
                let prog = progress.fetch_add(1, Ordering::Relaxed);
                if prog % 10000 == 0 {
                    let pct = (prog as f32 / total_pixels as f32 * 100.0) as u32;
                    eprint!("\rProgress: {}%  ", pct);
                }
                
                (x, y, color)
            })
        })
        .collect();
    
    eprintln!("\rProgress: 100%  ");
    
    // Create image
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(params.width, params.height);
    
    for (x, y, color) in pixels {
        img.put_pixel(x, y, Rgb(color));
    }
    
    let elapsed = start.elapsed();
    println!("Render time: {:.1}s", elapsed.as_secs_f32());
    
    // Save
    let filename = format!(
        "blackhole_{}x{}_spin{:.2}_q{}.png",
        params.width,
        params.height,
        params.spin,
        match params.quality {
            Quality::Preview => "preview",
            Quality::High => "high",
            Quality::Ultra => "ultra",
            Quality::Insane => "insane",
        }
    );
    
    img.save(&filename).expect("Failed to save image");
    println!("Saved to: {}", filename);
}
