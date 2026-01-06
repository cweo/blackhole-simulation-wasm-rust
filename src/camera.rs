use glam::{Mat4, Vec3};

#[derive(Clone)]
pub struct Camera {
    pub distance: f32,
    pub theta: f32,  // horizontal angle
    pub phi: f32,    // vertical angle
    pub target: Vec3,
    pub fov: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            distance: 20.0,
            theta: std::f32::consts::PI,  // 180 degrees
            phi: -79.0_f32.to_radians(),  // -79 degrees
            target: Vec3::ZERO,
            fov: 60.0_f32.to_radians(),
        }
    }

    pub fn position(&self) -> Vec3 {
        let x = self.distance * self.phi.cos() * self.theta.sin();
        let y = self.distance * self.phi.sin();
        let z = self.distance * self.phi.cos() * self.theta.cos();
        self.target + Vec3::new(x, y, z)
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.theta += dx;
        self.phi = (self.phi + dy).clamp(-1.5, 1.5);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta).clamp(5.0, 100.0);
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position(), self.target, Vec3::Y)
    }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov, aspect, 0.1, 1000.0)
    }

    /// Get camera uniform data for GPU
    pub fn uniform_data(&self, width: u32, height: u32) -> CameraUniform {
        let aspect = width as f32 / height as f32;
        let pos = self.position();
        let view = self.view_matrix();
        let proj = self.projection_matrix(aspect);
        let inv_view_proj = (proj * view).inverse();

        CameraUniform {
            position: [pos.x, pos.y, pos.z, 0.0],
            inv_view_proj: inv_view_proj.to_cols_array(),
            resolution: [width as f32, height as f32],
            fov: self.fov,
            _padding: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub position: [f32; 4],
    pub inv_view_proj: [f32; 16],
    pub resolution: [f32; 2],
    pub fov: f32,
    pub _padding: f32,
}
