# Kerr Black Hole Simulation

A photorealistic, real-time black hole visualization built with Rust, WebAssembly, and WebGPU. Inspired by the Gargantua black hole from *Interstellar*.

![Black Hole Simulation](image.png)

## Features

- **Kerr (rotating) black hole** - Accurate spacetime geometry with frame dragging
- **Real-time ray tracing** - GPU compute shader with RK4 geodesic integration
- **Relativistic effects**:
  - Gravitational lensing and light bending
  - Doppler beaming (approaching side brighter)
  - Gravitational redshift
  - Photon sphere and Einstein ring
- **Photorealistic accretion disk**:
  - Blackbody temperature gradient
  - Orbiting hot spots
  - Voronoi-based particle clumping
  - Spiral density waves
  - Dust lanes
  - Volumetric thickness
- **Cinematic post-processing**:
  - ACES filmic tone mapping
  - Film grain
  - Bloom effect
- **Interactive controls**:
  - Mouse drag to orbit camera
  - Scroll to zoom
  - Adjustable spin parameter (0 to 0.99)
  - Resolution toggle (720p/1080p)

## Tech Stack

- **Rust** - Systems programming language
- **wgpu** - Cross-platform WebGPU implementation
- **WGSL** - WebGPU Shading Language for compute shaders
- **wasm-bindgen** - Rust/JavaScript interop
- **wasm-pack** - WASM build tooling

## Building

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack
cargo install wasm-pack

# Add WASM target
rustup target add wasm32-unknown-unknown
```

### Build & Run

```bash
# Build WASM module
wasm-pack build --target web --release

# Serve locally (Python)
python3 -m http.server 8080

# Or use any static file server
npx serve .
```

Then open http://localhost:8080 in a WebGPU-compatible browser (Chrome 113+, Edge 113+, Firefox Nightly).

## Physics

The simulation implements Kerr black hole geodesics using:

- **Boyer-Lindquist coordinates** for the rotating black hole metric
- **RK4 integration** for photon trajectories with adaptive step sizing
- **ISCO calculation** - Innermost Stable Circular Orbit depends on spin
- **Frame dragging** - Space itself rotates near the black hole

Key equations:
- Event horizon: $r_H = M + \sqrt{M^2 - a^2}$
- Photon sphere: $r_{ph} \approx 1.5(1 + \sqrt{1 - a^2/2})$
- ISCO: $r_{ISCO} = 3 + Z_2 - \sqrt{(3-Z_1)(3+Z_1+2Z_2)}$

## Controls

| Control | Action |
|---------|--------|
| Left mouse drag | Rotate camera |
| Scroll wheel | Zoom in/out |
| Spin slider | Adjust black hole spin (a/M) |
| Quality toggle | Switch 720p/1080p |

## Browser Requirements

Requires WebGPU support:
- Chrome 113+ (recommended)
- Edge 113+
- Firefox Nightly (with `dom.webgpu.enabled`)
- Safari 18+ (macOS Sequoia)

## Offline High-Quality Renderers

Two offline renderers are available for publication-quality images:

### GPU Renderer (Recommended)

Uses wgpu compute shaders for massively parallel rendering. **10-100x faster** than CPU.

```bash
# Build the GPU renderer
cargo build --release --bin gpu_render

# Run with default settings (4K, ultra quality)
./target/release/gpu_render

# Custom settings
./target/release/gpu_render -w 7680 -h 4320 -q insane -s 0.95

# See all options
./target/release/gpu_render --help
```

### CPU Renderer (Fallback)

Uses Rayon for CPU parallelization. Useful if GPU is unavailable.

```bash
# Build the CPU renderer
cargo build --release --bin offline_render

# Run with default settings
./target/release/offline_render

# See all options
./target/release/offline_render --help
```

### Quality Presets

| Preset | Ray Steps | Samples/Pixel | GPU Time (4K) | CPU Time (4K) |
|--------|-----------|---------------|---------------|---------------|
| preview | 500 | 1 | ~1 sec | ~30 sec |
| high | 2,000 | 4 | ~10 sec | ~5 min |
| ultra | 5,000 | 16 | ~1 min | ~30 min |
| insane | 10,000 | 64 | ~5 min | ~3 hours |

### Command Line Options

Both renderers share the same CLI interface:

```
-w, --width <WIDTH>      Output width (default: 3840)
-h, --height <HEIGHT>    Output height (default: 2160)
-s, --spin <SPIN>        Black hole spin 0-0.99 (default: 0.9)
-q, --quality <QUALITY>  preview|high|ultra|insane (default: ultra)
-d, --distance <DIST>    Camera distance (default: 25)
-t, --theta <DEG>        Camera horizontal angle (default: 0)
-p, --phi <DEG>          Camera vertical angle (default: 20)
```

Example for an 8K insane quality render with GPU:
```bash
./target/release/gpu_render -w 7680 -h 4320 -q insane -s 0.98 -p 25
```

## License

MIT