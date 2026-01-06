// Kerr Black Hole Ray Tracer - Compute Shader
// Real-time visualization with gravitational lensing and accretion disk

struct CameraUniform {
    position: vec4<f32>,
    inv_view_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    fov: f32,
    _padding: f32,
}

struct BlackHoleUniform {
    spin: f32,        // a/M, range [0, 0.99]
    mass: f32,        // M (normalized to 1)
    disk_inner: f32,  // ISCO radius
    disk_outer: f32,  // outer disk radius
    time: f32,        // animation time
    _padding: vec3<f32>,
}

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> camera: CameraUniform;
@group(0) @binding(2) var<uniform> blackhole: BlackHoleUniform;

const PI: f32 = 3.14159265359;
const MAX_STEPS: i32 = 256;
const ESCAPE_RADIUS: f32 = 50.0;

// Compute Kerr r-coordinate from Cartesian position
// Solves: x²+y²/(r²+a²) + z²/r² = 1
fn kerr_r(pos: vec3<f32>, a: f32) -> f32 {
    let rho2 = dot(pos.xy, pos.xy) + pos.z * pos.z;
    let a2 = a * a;
    
    // Solve r⁴ - (ρ² - a²)r² - a²z² = 0
    let b = rho2 - a2;
    let c = -a2 * pos.z * pos.z;
    let disc = sqrt(max(b * b - 4.0 * c, 0.0));
    
    return sqrt(max((b + disc) * 0.5, 0.001));
}

// Compute event horizon radius for Kerr black hole
fn horizon_radius(a: f32) -> f32 {
    return 1.0 + sqrt(max(1.0 - a * a, 0.0));
}

// Frame dragging angular velocity
fn frame_drag_omega(r: f32, z: f32, a: f32) -> f32 {
    let r2 = r * r;
    let a2 = a * a;
    let sigma = r2 + a2 * (z * z) / max(r2, 0.001);
    let denom = sigma * (r2 + a2) + 2.0 * r * a2 * (1.0 - (z * z) / max(r2, 0.001));
    return 2.0 * a * r / max(denom, 0.001);
}

// Kerr geodesic acceleration
// Combines Schwarzschild-like radial term with frame dragging
fn kerr_acceleration(pos: vec3<f32>, vel: vec3<f32>, a: f32) -> vec3<f32> {
    let r = kerr_r(pos, a);
    let r2 = r * r;
    let r5 = r2 * r2 * r;
    let a2 = a * a;
    
    // Angular momentum (conserved in Schwarzschild limit)
    let L = cross(pos, vel);
    let h2 = dot(L, L);
    
    // Schwarzschild-like radial acceleration
    let radial_factor = -1.5 * h2 / max(r5, 0.001);
    var accel = radial_factor * pos;
    
    // Frame dragging correction
    let omega = frame_drag_omega(r, pos.z, a);
    let frame_drag = omega * vec3<f32>(-vel.y, vel.x, 0.0);
    accel += frame_drag * a * 2.0;
    
    // Additional Kerr correction for polar motion
    let sigma = r2 + a2 * (pos.z * pos.z) / max(r2, 0.001);
    let polar_correction = a2 * pos.z / max(sigma * r2, 0.001);
    accel.z += polar_correction * dot(vel.xy, vel.xy);
    
    return accel;
}

// Adaptive step size - smaller near horizon, larger far away
fn adaptive_step(r: f32, r_horizon: f32) -> f32 {
    let base_step = 0.1;
    let horizon_dist = r - r_horizon;
    
    // Very small steps near horizon
    if (horizon_dist < 1.0) {
        return base_step * horizon_dist * 0.5;
    }
    
    // Medium steps in interesting region
    if (r < 10.0) {
        return base_step * 0.5;
    }
    
    // Larger steps far away
    return base_step * min(r * 0.1, 2.0);
}

// RK4 integration step
fn rk4_step(pos: vec3<f32>, vel: vec3<f32>, a: f32, dt: f32) -> array<vec3<f32>, 2> {
    let k1_v = kerr_acceleration(pos, vel, a);
    let k1_p = vel;
    
    let k2_v = kerr_acceleration(pos + k1_p * dt * 0.5, vel + k1_v * dt * 0.5, a);
    let k2_p = vel + k1_v * dt * 0.5;
    
    let k3_v = kerr_acceleration(pos + k2_p * dt * 0.5, vel + k2_v * dt * 0.5, a);
    let k3_p = vel + k2_v * dt * 0.5;
    
    let k4_v = kerr_acceleration(pos + k3_p * dt, vel + k3_v * dt, a);
    let k4_p = vel + k3_v * dt;
    
    let new_pos = pos + (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) * dt / 6.0;
    let new_vel = vel + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * dt / 6.0;
    
    return array<vec3<f32>, 2>(new_pos, new_vel);
}

// Check disk crossing (z-plane intersection)
fn check_disk_crossing(old_pos: vec3<f32>, new_pos: vec3<f32>) -> vec3<f32> {
    // Check if ray crossed z=0 plane
    if (old_pos.z * new_pos.z > 0.0) {
        return vec3<f32>(-1.0, 0.0, 0.0); // No crossing
    }
    
    // Interpolate to find crossing point
    let t = -old_pos.z / (new_pos.z - old_pos.z + 0.0001);
    let hit_pos = mix(old_pos, new_pos, t);
    
    let r = length(hit_pos.xy);
    if (r >= blackhole.disk_inner && r <= blackhole.disk_outer) {
        return vec3<f32>(1.0, r, atan2(hit_pos.y, hit_pos.x));
    }
    
    return vec3<f32>(-1.0, 0.0, 0.0);
}

// Blackbody color approximation
fn blackbody_color(temp: f32) -> vec3<f32> {
    let t = clamp(temp / 1000.0, 1.0, 40.0);
    
    var color: vec3<f32>;
    if (t < 6.6) {
        color.r = 1.0;
        color.g = clamp(0.39 * log(t) - 0.63, 0.0, 1.0);
        if (t > 1.9) {
            color.b = clamp(0.39 * log(t - 1.0) - 1.08, 0.0, 1.0);
        } else {
            color.b = 0.0;
        }
    } else {
        color.r = clamp(1.29 * pow(t - 0.6, -0.13), 0.0, 1.0);
        color.g = clamp(1.29 * pow(t - 0.6, -0.08), 0.0, 1.0);
        color.b = 1.0;
    }
    
    return color;
}

// Sample accretion disk with Doppler and gravitational redshift
fn sample_disk(hit_r: f32, hit_phi: f32, ray_dir: vec3<f32>, a: f32) -> vec4<f32> {
    let hit_pos = vec3<f32>(hit_r * cos(hit_phi), hit_r * sin(hit_phi), 0.0);
    
    // Keplerian orbital velocity (prograde)
    let v_orbit = 1.0 / (sqrt(hit_r) + a);
    
    // Disk velocity direction (circular orbit)
    let disk_vel = v_orbit * vec3<f32>(-sin(hit_phi), cos(hit_phi), 0.0);
    
    // Lorentz factor
    let v2 = dot(disk_vel, disk_vel);
    let gamma = 1.0 / sqrt(max(1.0 - v2, 0.01));
    
    // Doppler factor
    let photon_dir = normalize(ray_dir);
    let doppler = gamma * (1.0 + dot(disk_vel, photon_dir));
    
    // Gravitational redshift
    let grav_z = 1.0 / sqrt(max(1.0 - 1.0 / hit_r, 0.01));
    
    // Total redshift factor
    let total_z = doppler * grav_z;
    
    // Temperature profile: T ∝ R^(-3/4)
    let base_temp = 8000.0;
    let temp = base_temp * pow(blackhole.disk_inner / hit_r, 0.75) / total_z;
    
    // Blackbody color from temperature
    let color = blackbody_color(temp);
    
    // Intensity: I ∝ T^4 (Stefan-Boltzmann), modified for visual appeal
    let intensity = pow(temp / base_temp, 2.0);
    
    // Radial brightness profile - brighter near ISCO
    let radial_brightness = 1.0 / (1.0 + (hit_r - blackhole.disk_inner) * 0.1);
    
    // Smooth taper at ISCO
    let isco_taper = smoothstep(blackhole.disk_inner * 0.9, blackhole.disk_inner * 1.2, hit_r);
    
    // Turbulence/noise pattern for texture
    let noise = 0.7 + 0.3 * sin(hit_phi * 20.0 + blackhole.time * 2.0) 
                    * sin(hit_r * 3.0 + blackhole.time);
    
    let final_intensity = intensity * radial_brightness * isco_taper * noise;
    
    return vec4<f32>(color * final_intensity * 2.0, min(isco_taper, 1.0));
}

// Procedural starfield
fn sample_stars(dir: vec3<f32>) -> vec3<f32> {
    // Simple hash-based stars
    let d = normalize(dir);
    let theta = atan2(d.z, d.x);
    let phi = asin(d.y);
    
    // Grid-based star positions
    let grid = floor(vec2<f32>(theta * 20.0, phi * 20.0));
    let hash = fract(sin(dot(grid, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    
    // Star probability and brightness
    if (hash > 0.97) {
        let brightness = pow(hash - 0.97, 0.5) * 30.0;
        let star_color = mix(vec3<f32>(1.0, 0.9, 0.8), vec3<f32>(0.8, 0.9, 1.0), hash);
        return star_color * brightness;
    }
    
    // Background nebula glow
    let nebula = 0.02 * vec3<f32>(0.1, 0.05, 0.15) * (1.0 + sin(theta * 3.0) * cos(phi * 2.0));
    
    return nebula;
}

// Main ray tracing function
fn trace_ray(origin: vec3<f32>, direction: vec3<f32>) -> vec4<f32> {
    var pos = origin;
    var vel = normalize(direction);
    var color = vec3<f32>(0.0);
    var alpha = 0.0;
    
    let a = blackhole.spin * blackhole.mass;
    let r_horizon = horizon_radius(blackhole.spin);
    
    for (var i = 0; i < MAX_STEPS; i++) {
        if (alpha > 0.99) {
            break;
        }
        
        let r = kerr_r(pos, a);
        
        // Check event horizon
        if (r < r_horizon * 1.01) {
            // Ray fell into black hole - black
            color = mix(color, vec3<f32>(0.0), 1.0 - alpha);
            alpha = 1.0;
            break;
        }
        
        // Check escape
        if (r > ESCAPE_RADIUS) {
            // Sample starfield
            let stars = sample_stars(vel);
            color = mix(color, stars, 1.0 - alpha);
            alpha = 1.0;
            break;
        }
        
        // Adaptive step size
        let dt = adaptive_step(r, r_horizon);
        
        // RK4 integration
        let old_pos = pos;
        let result = rk4_step(pos, vel, a, dt);
        pos = result[0];
        vel = normalize(result[1]);
        
        // Check disk intersection
        let disk_hit = check_disk_crossing(old_pos, pos);
        if (disk_hit.x > 0.0) {
            let disk_color = sample_disk(disk_hit.y, disk_hit.z, vel, a);
            
            // Blend disk color
            let blend = disk_color.a * (1.0 - alpha);
            color = color + disk_color.rgb * blend;
            alpha = alpha + blend * 0.7; // Semi-transparent disk
        }
    }
    
    return vec4<f32>(color, 1.0);
}

// Generate ray from pixel coordinates
fn generate_ray(pixel: vec2<f32>) -> array<vec3<f32>, 2> {
    // Normalized device coordinates
    let ndc = (pixel / camera.resolution) * 2.0 - 1.0;
    
    // Clip space positions
    let clip_near = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    let clip_far = vec4<f32>(ndc.x, -ndc.y, 1.0, 1.0);
    
    // Transform to world space
    var world_near = camera.inv_view_proj * clip_near;
    var world_far = camera.inv_view_proj * clip_far;
    
    world_near /= world_near.w;
    world_far /= world_far.w;
    
    let origin = camera.position.xyz;
    let direction = normalize(world_far.xyz - world_near.xyz);
    
    return array<vec3<f32>, 2>(origin, direction);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output_texture);
    
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }
    
    let pixel = vec2<f32>(f32(global_id.x) + 0.5, f32(global_id.y) + 0.5);
    
    // Generate ray
    let ray = generate_ray(pixel);
    let origin = ray[0];
    let direction = ray[1];
    
    // Trace ray
    var color = trace_ray(origin, direction);
    
    // Simple tone mapping
    color = vec4<f32>(color.rgb / (color.rgb + vec3<f32>(1.0)), color.a);
    
    // Gamma correction
    color = vec4<f32>(pow(color.rgb, vec3<f32>(1.0 / 2.2)), color.a);
    
    textureStore(output_texture, vec2<i32>(global_id.xy), color);
}
