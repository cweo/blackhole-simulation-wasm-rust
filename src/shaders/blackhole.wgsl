// Kerr Black Hole Ray Tracer - Compute Shader
// Photorealistic visualization with gravitational lensing, accretion disk, and particle effects

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
    _padding: vec3<f32>,
}

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> camera: CameraUniform;
@group(0) @binding(2) var<uniform> blackhole: BlackHoleUniform;

const PI: f32 = 3.14159265359;
const MAX_STEPS: i32 = 500;
const ESCAPE_RADIUS: f32 = 100.0;

// 3D hash
fn hash3(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// 2D hash
fn hash(p: vec2<f32>) -> f32 {
    let p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    let p3d = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3d.x + p3d.y) * p3d.z);
}

fn hash2(p: vec2<f32>) -> vec2<f32> {
    let k = vec2<f32>(0.3183099, 0.3678794);
    let pp = p * k + k.yx;
    return fract(16.0 * k * fract(pp.x * pp.y * (pp.x + pp.y)));
}

// Smooth noise
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2<f32>(1.0, 0.0)), u.x),
               mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// 3D noise for volumetric effects
fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash3(i), hash3(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
                   mix(hash3(i + vec3<f32>(0.0, 1.0, 0.0)), hash3(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y),
               mix(mix(hash3(i + vec3<f32>(0.0, 0.0, 1.0)), hash3(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
                   mix(hash3(i + vec3<f32>(0.0, 1.0, 1.0)), hash3(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}

// High quality FBM with more octaves
fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pp = p;
    for (var i = 0; i < 6; i++) {
        value += amplitude * noise(pp);
        pp *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// FBM for 3D
fn fbm3(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pp = p;
    for (var i = 0; i < 5; i++) {
        value += amplitude * noise3(pp);
        pp *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// Voronoi for clumpy particle structures
fn voronoi(p: vec2<f32>) -> vec2<f32> {
    let n = floor(p);
    let f = fract(p);
    var md = 8.0;
    var mr = vec2<f32>(0.0);
    
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let g = vec2<f32>(f32(i), f32(j));
            let o = hash2(n + g);
            let r = g + o - f;
            let d = dot(r, r);
            if (d < md) {
                md = d;
                mr = r;
            }
        }
    }
    return vec2<f32>(sqrt(md), md);
}

// Compute Kerr r-coordinate
fn kerr_r(pos: vec3<f32>, a: f32) -> f32 {
    let rho2 = dot(pos, pos);
    let a2 = a * a;
    let b = rho2 - a2;
    let c = -a2 * pos.z * pos.z;
    let disc = sqrt(max(b * b - 4.0 * c, 0.0));
    return sqrt(max((b + disc) * 0.5, 0.001));
}

// Event horizon radius
fn horizon_radius(a: f32) -> f32 {
    return 1.0 + sqrt(max(1.0 - a * a, 0.0));
}

// Photon sphere radius (approximate for Kerr)
fn photon_sphere_radius(a: f32) -> f32 {
    return 1.5 * (1.0 + sqrt(1.0 - a * a * 0.5));
}

// Frame dragging
fn frame_drag_omega(r: f32, z: f32, a: f32) -> f32 {
    let r2 = r * r;
    let a2 = a * a;
    let cos_theta2 = z * z / max(r2, 0.001);
    let sigma = r2 + a2 * cos_theta2;
    let delta = r2 - 2.0 * r + a2;
    let A = (r2 + a2) * (r2 + a2) - a2 * delta * (1.0 - cos_theta2);
    return 2.0 * a * r / max(A, 0.001);
}

// Kerr geodesic acceleration
fn kerr_acceleration(pos: vec3<f32>, vel: vec3<f32>, a: f32) -> vec3<f32> {
    let r = kerr_r(pos, a);
    let r2 = r * r;
    let a2 = a * a;
    let L = cross(pos, vel);
    let h2 = dot(L, L);
    let r5 = r2 * r2 * r;
    var accel = -1.5 * h2 / max(r5, 0.001) * pos;
    let omega = frame_drag_omega(r, pos.z, a);
    accel += omega * a * vec3<f32>(-vel.y, vel.x, 0.0) * 2.0;
    let sigma = r2 + a2 * (pos.z * pos.z) / max(r2, 0.001);
    accel.z += a2 * pos.z / max(sigma * r2, 0.001) * dot(vel.xy, vel.xy);
    return accel;
}

// Adaptive step size
fn adaptive_step(r: f32, r_horizon: f32) -> f32 {
    let dist = r - r_horizon;
    if (dist < 0.3) { return 0.015; }
    if (dist < 1.0) { return 0.03; }
    if (dist < 3.0) { return 0.05; }
    if (r < 15.0) { return 0.07; }
    return min(0.12, r * 0.015);
}

// RK4 integration
fn rk4_step(pos: vec3<f32>, vel: vec3<f32>, a: f32, dt: f32) -> array<vec3<f32>, 2> {
    let k1_v = kerr_acceleration(pos, vel, a);
    let k1_p = vel;
    let k2_v = kerr_acceleration(pos + k1_p * dt * 0.5, vel + k1_v * dt * 0.5, a);
    let k2_p = vel + k1_v * dt * 0.5;
    let k3_v = kerr_acceleration(pos + k2_p * dt * 0.5, vel + k2_v * dt * 0.5, a);
    let k3_p = vel + k2_v * dt * 0.5;
    let k4_v = kerr_acceleration(pos + k3_p * dt, vel + k3_v * dt, a);
    let k4_p = vel + k3_v * dt;
    return array<vec3<f32>, 2>(
        pos + (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) * dt / 6.0,
        vel + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * dt / 6.0
    );
}

// Check disk crossing with thickness
fn check_disk_crossing(old_pos: vec3<f32>, new_pos: vec3<f32>, disk_height: f32) -> vec4<f32> {
    // Check if we're within disk thickness
    let avg_z = (old_pos.z + new_pos.z) * 0.5;
    let r = length(vec2<f32>(old_pos.x + new_pos.x, old_pos.y + new_pos.y) * 0.5);
    
    // Disk thickness varies with radius (flared disk)
    let local_height = disk_height * (1.0 + (r - blackhole.disk_inner) * 0.05);
    
    if (abs(avg_z) < local_height && r >= blackhole.disk_inner && r <= blackhole.disk_outer) {
        let hit_pos = (old_pos + new_pos) * 0.5;
        let hit_r = length(hit_pos.xy);
        let hit_phi = atan2(hit_pos.y, hit_pos.x);
        let depth = 1.0 - abs(avg_z) / local_height;
        return vec4<f32>(1.0, hit_r, hit_phi, depth);
    }
    
    // Also check z-plane crossing for thin disk component
    if (old_pos.z * new_pos.z < 0.0) {
        let t = -old_pos.z / (new_pos.z - old_pos.z + 0.00001);
        let hit_pos = mix(old_pos, new_pos, t);
        let hit_r = length(hit_pos.xy);
        if (hit_r >= blackhole.disk_inner && hit_r <= blackhole.disk_outer) {
            return vec4<f32>(1.0, hit_r, atan2(hit_pos.y, hit_pos.x), 1.0);
        }
    }
    return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
}

// Enhanced blackbody with Interstellar-like colors
fn blackbody_color(temp: f32) -> vec3<f32> {
    let t = clamp(temp, 800.0, 40000.0);
    var color: vec3<f32>;
    
    if (t < 6600.0) {
        color.r = 1.0;
        color.g = clamp(0.39 * log(t / 1000.0) - 0.19, 0.0, 1.0);
        if (t > 1900.0) {
            color.b = clamp(0.39 * log(t / 1000.0 - 1.0) - 0.35, 0.0, 1.0);
        } else {
            color.b = 0.0;
        }
    } else {
        color.r = clamp(1.29 * pow((t - 6000.0) / 1000.0, -0.1332), 0.0, 1.0);
        color.g = clamp(1.29 * pow((t - 6000.0) / 1000.0, -0.0755), 0.0, 1.0);
        color.b = 1.0;
    }
    return color;
}

// Hot spot particles in the disk
fn hot_spots(r: f32, phi: f32, time: f32) -> f32 {
    var spots = 0.0;
    
    // Multiple orbiting hot spots at different radii
    for (var i = 0; i < 5; i++) {
        let spot_r = blackhole.disk_inner * (1.2 + f32(i) * 0.4);
        let orbit_speed = 1.0 / (sqrt(spot_r) * spot_r);
        let spot_phi = f32(i) * 1.256 + time * orbit_speed;
        
        let dr = r - spot_r;
        let dphi = phi - spot_phi;
        let dphi_wrapped = atan2(sin(dphi), cos(dphi));
        
        let dist2 = dr * dr + (dphi_wrapped * spot_r) * (dphi_wrapped * spot_r);
        let spot_size = 0.3 + hash(vec2<f32>(f32(i), 0.0)) * 0.4;
        spots += exp(-dist2 / (spot_size * spot_size)) * (0.5 + hash(vec2<f32>(f32(i), 1.0)) * 0.5);
    }
    
    return spots;
}

// Clumpy particle density using Voronoi
fn particle_density(r: f32, phi: f32, time: f32) -> f32 {
    let orbit_phase = phi - time * 0.2 / sqrt(max(r, 1.0));
    let p = vec2<f32>(orbit_phase * 3.0, log(r) * 4.0);
    let v = voronoi(p * 3.0);
    
    // Create clumpy structure
    let clumps = smoothstep(0.3, 0.0, v.x);
    
    // Add smaller scale detail
    let fine_detail = fbm(vec2<f32>(phi * 20.0, r * 2.0) + time * 0.1);
    
    return clumps * 0.5 + fine_detail * 0.3 + 0.4;
}

// Dust lanes (darker absorption regions)
fn dust_lanes(r: f32, phi: f32) -> f32 {
    let spiral_arm = sin(phi * 2.0 + log(r) * 3.0) * 0.5 + 0.5;
    let dust = fbm(vec2<f32>(phi * 5.0, r * 0.5)) * spiral_arm;
    return 1.0 - dust * 0.3;
}

// Sample accretion disk with particles and enhanced detail
fn sample_disk(hit_r: f32, hit_phi: f32, ray_dir: vec3<f32>, depth: f32, a: f32) -> vec4<f32> {
    let r_safe = max(hit_r, blackhole.disk_inner);
    let v_kepler = sqrt(1.0 / r_safe);
    let disk_vel = v_kepler * vec3<f32>(-sin(hit_phi), cos(hit_phi), 0.0);
    
    let v2 = min(dot(disk_vel, disk_vel), 0.95);
    let gamma = 1.0 / sqrt(1.0 - v2);
    let photon_dir = normalize(ray_dir);
    let cos_angle = dot(normalize(disk_vel), photon_dir);
    let doppler = gamma * (1.0 - sqrt(v2) * cos_angle);
    let grav_redshift = sqrt(max(1.0 - 2.0 / r_safe, 0.01));
    let freq_shift = doppler * grav_redshift;
    
    // Temperature with local variations
    let r_ratio = blackhole.disk_inner / r_safe;
    let base_temp = 5500.0;
    let temp_variation = 1.0 + 0.3 * fbm(vec2<f32>(hit_phi * 10.0, hit_r + blackhole.time * 0.2));
    let local_temp = base_temp * pow(r_ratio, 0.75) * temp_variation;
    let observed_temp = local_temp / freq_shift;
    
    var color = blackbody_color(observed_temp);
    
    // Relativistic beaming
    let beam_factor = pow(1.0 / max(freq_shift, 0.1), 3.5);
    
    // Radial profile
    let r_norm = (hit_r - blackhole.disk_inner) / (blackhole.disk_outer - blackhole.disk_inner);
    let radial_profile = exp(-r_norm * 1.5) * (1.0 - exp(-r_norm * 10.0));
    
    // Particle density and clumping
    let particles = particle_density(hit_r, hit_phi, blackhole.time);
    
    // Hot spots (bright orbiting blobs)
    let spots = hot_spots(hit_r, hit_phi, blackhole.time);
    
    // Dust absorption
    let dust = dust_lanes(hit_r, hit_phi);
    
    // Spiral density waves
    let spiral_phase = hit_phi * 2.0 + log(r_safe) * 4.0 - blackhole.time * 0.15;
    let spiral_density = 0.8 + 0.2 * sin(spiral_phase);
    let spiral_phase2 = hit_phi * 3.0 - log(r_safe) * 2.0 + blackhole.time * 0.1;
    let spiral_density2 = 0.9 + 0.1 * sin(spiral_phase2);
    
    // Combine all effects
    var intensity = beam_factor * radial_profile * particles * dust * spiral_density * spiral_density2;
    
    // Add hot spot glow
    let spot_color = blackbody_color(observed_temp * 1.5);
    color = mix(color, spot_color, spots * 0.5);
    intensity += spots * 2.0;
    
    // Inner edge glow (matter plunging into BH)
    let inner_glow = exp(-(hit_r - blackhole.disk_inner) * 2.0) * 3.0;
    color += vec3<f32>(1.0, 0.7, 0.3) * inner_glow * beam_factor;
    
    // Edge softening
    let inner_fade = smoothstep(blackhole.disk_inner * 0.95, blackhole.disk_inner * 1.15, hit_r);
    let outer_fade = 1.0 - smoothstep(blackhole.disk_outer * 0.7, blackhole.disk_outer, hit_r);
    
    // Depth factor for volumetric feel
    let volume_factor = 0.7 + 0.3 * depth;
    
    color *= intensity * inner_fade * outer_fade * volume_factor * 2.0;
    
    return vec4<f32>(color, inner_fade * outer_fade * depth);
}

// Photon ring glow near the photon sphere
fn photon_ring_glow(r: f32, r_photon: f32) -> vec3<f32> {
    let dist = abs(r - r_photon);
    let glow = exp(-dist * 3.0) * 0.3;
    return vec3<f32>(1.0, 0.9, 0.7) * glow;
}

// Hawking radiation - particles emerging from near the event horizon
// Visualized as discrete particle-like emissions
fn hawking_radiation(pos: vec3<f32>, r: f32, r_horizon: f32, time: f32) -> vec3<f32> {
    // Only emit from just outside event horizon
    let emission_zone = smoothstep(r_horizon * 0.95, r_horizon * 1.05, r) 
                       * (1.0 - smoothstep(r_horizon * 1.05, r_horizon * 2.5, r));
    
    if (emission_zone < 0.01) {
        return vec3<f32>(0.0);
    }
    
    // Convert position to spherical for particle placement
    let theta = atan2(pos.y, pos.x);
    let phi = asin(clamp(pos.z / max(r, 0.001), -1.0, 1.0));
    
    var hawking = vec3<f32>(0.0);
    
    // Create multiple particle streams at different angles
    for (var i = 0; i < 12; i++) {
        // Each particle stream has a base angle
        let stream_theta = f32(i) * PI / 6.0;
        let stream_phi = sin(f32(i) * 2.3) * 0.8;
        
        // Particles move outward over time
        let particle_speed = 0.3 + hash(vec2<f32>(f32(i), 0.0)) * 0.4;
        let particle_phase = fract(time * particle_speed + hash(vec2<f32>(f32(i), 1.0)));
        
        // Particle radial position (emerging from horizon)
        let particle_r = r_horizon * (1.02 + particle_phase * 1.5);
        
        // Distance from this ray position to particle
        let dr = abs(r - particle_r);
        let dtheta = abs(atan2(sin(theta - stream_theta), cos(theta - stream_theta)));
        let dphi = abs(phi - stream_phi);
        
        // Particle intensity with distance falloff
        let angular_dist = sqrt(dtheta * dtheta + dphi * dphi) * particle_r;
        let radial_dist = dr;
        let total_dist = sqrt(angular_dist * angular_dist + radial_dist * radial_dist);
        
        // Particle glow - small bright points
        let particle_size = 0.15 + hash(vec2<f32>(f32(i), 2.0)) * 0.1;
        let particle_glow = exp(-total_dist * total_dist / (particle_size * particle_size));
        
        // Particles fade as they move away
        let fade = 1.0 - particle_phase;
        
        // Hawking radiation is thermal - mix of particle colors
        // Virtual particle pairs: one escapes (we see it), one falls in
        let temp = 6.2e-8 / r_horizon; // Hawking temperature (scaled for visibility)
        let visual_temp = 8000.0 + hash(vec2<f32>(f32(i), 3.0)) * 12000.0; // Artistic license
        let particle_color = blackbody_color(visual_temp);
        
        // Some particles are matter (warm), some antimatter (cool blue tint)
        var final_color = particle_color;
        if (hash(vec2<f32>(f32(i), 4.0)) > 0.5) {
            final_color = mix(particle_color, vec3<f32>(0.6, 0.8, 1.0), 0.3);
        }
        
        hawking += final_color * particle_glow * fade * 1.5;
    }
    
    // Add continuous faint glow representing the quantum foam
    let quantum_foam = fbm(vec2<f32>(theta * 10.0 + time * 2.0, phi * 10.0)) * emission_zone;
    hawking += vec3<f32>(0.5, 0.6, 0.9) * quantum_foam * 0.15;
    
    return hawking * emission_zone;
}

// Enhanced starfield with nebula
fn sample_stars(dir: vec3<f32>) -> vec3<f32> {
    let d = normalize(dir);
    let theta = atan2(d.z, d.x);
    let phi = asin(clamp(d.y, -1.0, 1.0));
    
    var stars = vec3<f32>(0.0);
    
    // Multiple star layers with varying densities
    for (var layer = 0; layer < 4; layer++) {
        let scale = 40.0 + f32(layer) * 25.0;
        let grid = floor(vec2<f32>(theta, phi) * scale);
        let cell_center = (grid + 0.5) / scale;
        let h = hash(grid + f32(layer) * 137.0);
        
        if (h > 0.982) {
            // Star position within cell
            let star_offset = hash2(grid) - 0.5;
            let star_pos = cell_center + star_offset * 0.8 / scale;
            let dist = length(vec2<f32>(theta, phi) - star_pos * scale / scale);
            
            let brightness = pow((h - 0.982) / 0.018, 0.4) * (1.0 - f32(layer) * 0.15);
            let star_temp = 2500.0 + h * 25000.0;
            let star_glow = exp(-dist * dist * scale * 2.0);
            stars += blackbody_color(star_temp) * brightness * star_glow * 0.5;
        }
    }
    
    // Nebula clouds
    let nebula_coord = vec2<f32>(theta * 2.0, phi * 3.0);
    let nebula_density = fbm(nebula_coord) * fbm(nebula_coord * 2.0 + 5.0);
    let nebula_color = mix(
        vec3<f32>(0.1, 0.05, 0.2),
        vec3<f32>(0.2, 0.1, 0.15),
        fbm(nebula_coord * 0.5)
    );
    stars += nebula_color * nebula_density * 0.1;
    
    // Milky way band
    let galactic_plane = exp(-phi * phi * 8.0);
    let milky_way = fbm(vec2<f32>(theta * 3.0, phi * 10.0)) * galactic_plane;
    stars += vec3<f32>(0.15, 0.12, 0.1) * milky_way * 0.3;
    
    return stars;
}

// Main ray tracing with volumetric disk
fn trace_ray(origin: vec3<f32>, direction: vec3<f32>) -> vec4<f32> {
    var pos = origin;
    var vel = normalize(direction);
    var color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;
    
    let a = blackhole.spin * blackhole.mass;
    let r_horizon = horizon_radius(blackhole.spin);
    let r_photon = photon_sphere_radius(blackhole.spin);
    let disk_height = 0.15;
    
    var in_disk_count = 0;
    
    for (var i = 0; i < MAX_STEPS; i++) {
        if (accumulated_alpha > 0.995) { break; }
        
        let r = kerr_r(pos, a);
        
        // Fell into black hole
        if (r < r_horizon * 1.01) {
            color = mix(color, vec3<f32>(0.0), 1.0 - accumulated_alpha);
            accumulated_alpha = 1.0;
            break;
        }
        
        // Hawking radiation near horizon
        if (r > r_horizon * 0.95 && r < r_horizon * 2.5) {
            let hawking = hawking_radiation(pos, r, r_horizon, blackhole.time);
            color += hawking * (1.0 - accumulated_alpha);
        }
        
        // Photon ring glow
        if (r > r_horizon * 1.05 && r < r_photon * 1.5) {
            let ring_glow = photon_ring_glow(r, r_photon);
            color += ring_glow * (1.0 - accumulated_alpha) * 0.1;
        }
        
        // Escaped to infinity
        if (r > ESCAPE_RADIUS) {
            let bg = sample_stars(vel);
            color = mix(color, bg, 1.0 - accumulated_alpha);
            accumulated_alpha = 1.0;
            break;
        }
        
        let dt = adaptive_step(r, r_horizon);
        let old_pos = pos;
        
        let result = rk4_step(pos, vel, a, dt);
        pos = result[0];
        vel = normalize(result[1]);
        
        // Check disk with thickness
        let disk_hit = check_disk_crossing(old_pos, pos, disk_height);
        if (disk_hit.x > 0.0 && in_disk_count < 3) {
            let disk_color = sample_disk(disk_hit.y, disk_hit.z, vel, disk_hit.w, a);
            let blend = disk_color.a * (1.0 - accumulated_alpha);
            color += disk_color.rgb * blend;
            accumulated_alpha += blend * 0.85;
            in_disk_count++;
        }
    }
    
    return vec4<f32>(color, 1.0);
}

// Generate camera ray
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
    
    let pixel = vec2<f32>(f32(global_id.x) + 0.5, f32(global_id.y) + 0.5);
    
    let ray = generate_ray(pixel);
    var color = trace_ray(ray[0], ray[1]);
    
    // ACES filmic tone mapping
    let x = color.rgb;
    var mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    
    // Slight bloom simulation
    let luminance = dot(mapped, vec3<f32>(0.299, 0.587, 0.114));
    if (luminance > 0.8) {
        mapped += (mapped - 0.8) * 0.2;
    }
    
    // Film grain for realism
    let grain = (hash(pixel + blackhole.time * 100.0) - 0.5) * 0.015;
    mapped += grain;
    
    // Gamma correction
    let final_color = pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    
    textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(final_color, 1.0));
}
