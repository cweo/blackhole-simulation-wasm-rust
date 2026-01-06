mod renderer;
mod camera;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
use std::cell::RefCell;
use std::rc::Rc;
use wgpu::SurfaceError;

use crate::camera::Camera;
use crate::renderer::Renderer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[derive(Clone, Copy)]
pub struct SimParams {
    pub spin: f32,
    pub resolution_scale: f32,
}

impl Default for SimParams {
    fn default() -> Self {
        Self { spin: 0.9, resolution_scale: 1.0 }
    }
}

#[cfg(target_arch = "wasm32")]
struct AppState {
    renderer: Option<Renderer>,
    camera: Camera,
    params: SimParams,
    mouse_pressed: bool,
    last_mouse_x: f32,
    last_mouse_y: f32,
    canvas_width: u32,
    canvas_height: u32,
}

#[cfg(target_arch = "wasm32")]
impl AppState {
    fn new() -> Self {
        Self {
            renderer: None,
            camera: Camera::new(),
            params: SimParams::default(),
            mouse_pressed: false,
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
            canvas_width: 1280,
            canvas_height: 720,
        }
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static APP_STATE: RefCell<AppState> = RefCell::new(AppState::new());
}

#[cfg(target_arch = "wasm32")]
fn read_ui_params() -> SimParams {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    
    let spin = document
        .get_element_by_id("spin-slider")
        .and_then(|el| el.dyn_into::<web_sys::HtmlInputElement>().ok())
        .map(|input| input.value().parse::<f32>().unwrap_or(0.9))
        .unwrap_or(0.9);

    let resolution_scale = document
        .get_element_by_id("resolution-toggle")
        .and_then(|el| el.dyn_into::<web_sys::HtmlInputElement>().ok())
        .map(|input| if input.checked() { 1.0 } else { 0.667 })
        .unwrap_or(1.0);

    SimParams { spin, resolution_scale }
}

#[cfg(target_arch = "wasm32")]
fn read_camera_overrides() -> (Option<f32>, Option<f32>, Option<f32>) {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    
    let distance = document
        .get_element_by_id("distance-slider")
        .and_then(|el| el.dyn_into::<web_sys::HtmlInputElement>().ok())
        .map(|input| input.value().parse::<f32>().unwrap_or(20.0));

    let theta = document
        .get_element_by_id("theta-slider")
        .and_then(|el| el.dyn_into::<web_sys::HtmlInputElement>().ok())
        .map(|input| input.value().parse::<f32>().unwrap_or(180.0).to_radians());

    let phi = document
        .get_element_by_id("phi-slider")
        .and_then(|el| el.dyn_into::<web_sys::HtmlInputElement>().ok())
        .map(|input| input.value().parse::<f32>().unwrap_or(-79.0).to_radians());

    (distance, theta, phi)
}

#[cfg(target_arch = "wasm32")]
fn render_frame() {
    APP_STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.params = read_ui_params();
        
        // Apply camera overrides from sliders
        let (distance, theta, phi) = read_camera_overrides();
        if let Some(d) = distance {
            state.camera.distance = d;
        }
        if let Some(t) = theta {
            state.camera.theta = t;
        }
        if let Some(p) = phi {
            state.camera.phi = p;
        }
        
        // Copy values we need before borrowing renderer mutably
        let camera = state.camera.clone();
        let params = state.params;
        let canvas_width = state.canvas_width;
        let canvas_height = state.canvas_height;
        
        if let Some(renderer) = &mut state.renderer {
            match renderer.render(&camera, &params) {
                Ok(_) => {}
                Err(SurfaceError::Lost) => {
                    renderer.resize(canvas_width, canvas_height, params.resolution_scale);
                }
                Err(e) => {
                    log::error!("Render error: {:?}", e);
                }
            }
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}

#[cfg(target_arch = "wasm32")]
fn start_render_loop() {
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();
    
    *g.borrow_mut() = Some(Closure::new(move || {
        render_frame();
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    
    request_animation_frame(g.borrow().as_ref().unwrap());
}

#[cfg(target_arch = "wasm32")]
fn setup_event_listeners(canvas: &web_sys::HtmlCanvasElement) {
    {
        let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
            if event.button() == 0 {
                APP_STATE.with(|state| {
                    let mut state = state.borrow_mut();
                    state.mouse_pressed = true;
                    state.last_mouse_x = event.client_x() as f32;
                    state.last_mouse_y = event.client_y() as f32;
                });
            }
        });
        canvas.add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref()).unwrap();
        closure.forget();
    }
    
    {
        let closure = Closure::<dyn FnMut(_)>::new(move |_event: web_sys::MouseEvent| {
            APP_STATE.with(|state| {
                state.borrow_mut().mouse_pressed = false;
            });
        });
        web_sys::window().unwrap().add_event_listener_with_callback("mouseup", closure.as_ref().unchecked_ref()).unwrap();
        closure.forget();
    }
    
    {
        let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
            APP_STATE.with(|state| {
                let mut state = state.borrow_mut();
                if state.mouse_pressed {
                    let x = event.client_x() as f32;
                    let y = event.client_y() as f32;
                    let dx = x - state.last_mouse_x;
                    let dy = y - state.last_mouse_y;
                    state.camera.rotate(dx * 0.005, dy * 0.005);
                    state.last_mouse_x = x;
                    state.last_mouse_y = y;
                }
            });
        });
        canvas.add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref()).unwrap();
        closure.forget();
    }
    
    {
        let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::WheelEvent| {
            event.prevent_default();
            APP_STATE.with(|state| {
                let delta = -event.delta_y() as f32 * 0.01;
                state.borrow_mut().camera.zoom(delta);
            });
        });
        canvas.add_event_listener_with_callback("wheel", closure.as_ref().unchecked_ref()).unwrap();
        closure.forget();
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub async fn run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
    
    log::info!("Starting Kerr Black Hole Simulation...");
    
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document
        .get_element_by_id("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    
    let container = document.get_element_by_id("container").unwrap();
    let width = container.client_width() as u32;
    let height = container.client_height() as u32;
    canvas.set_width(width);
    canvas.set_height(height);
    
    log::info!("Canvas size: {}x{}", width, height);
    
    let renderer = Renderer::new_from_canvas(&canvas, width, height).await;
    
    APP_STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.renderer = Some(renderer);
        state.canvas_width = width;
        state.canvas_height = height;
    });
    
    if let Some(loading) = document.get_element_by_id("loading") {
        loading.set_attribute("style", "display: none").unwrap();
    }
    
    setup_event_listeners(&canvas);
    start_render_loop();
    
    log::info!("Render loop started!");
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run() {
    use winit::{
        application::ApplicationHandler,
        dpi::LogicalSize,
        event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
        event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
        window::{Window, WindowId},
    };

    struct App {
        window: Option<Arc<Window>>,
        renderer: Option<Renderer>,
        camera: Camera,
        params: SimParams,
        mouse_pressed: bool,
        last_mouse_pos: Option<(f64, f64)>,
    }

    impl App {
        fn new() -> Self {
            Self {
                window: None,
                renderer: None,
                camera: Camera::new(),
                params: SimParams::default(),
                mouse_pressed: false,
                last_mouse_pos: None,
            }
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() { return; }

            let window_attrs = Window::default_attributes()
                .with_title("Kerr Black Hole Simulation")
                .with_inner_size(LogicalSize::new(1280, 720));

            let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
            let size = window.inner_size();
            let renderer = pollster::block_on(Renderer::new(window.clone(), size.width, size.height));

            self.window = Some(window);
            self.renderer = Some(renderer);
        }

        fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    if let Some(renderer) = &mut self.renderer {
                        renderer.resize(size.width, size.height, self.params.resolution_scale);
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        self.mouse_pressed = state == ElementState::Pressed;
                        if !self.mouse_pressed { self.last_mouse_pos = None; }
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if self.mouse_pressed {
                        if let Some((lx, ly)) = self.last_mouse_pos {
                            let dx = (position.x - lx) as f32;
                            let dy = (position.y - ly) as f32;
                            self.camera.rotate(dx * 0.005, dy * 0.005);
                        }
                        self.last_mouse_pos = Some((position.x, position.y));
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                    };
                    self.camera.zoom(scroll * 0.5);
                }
                WindowEvent::RedrawRequested => {
                    if let Some(renderer) = &mut self.renderer {
                        let _ = renderer.render(&self.camera, &self.params);
                    }
                }
                _ => {}
            }
        }

        fn about_to_wait(&mut self, _: &ActiveEventLoop) {
            if let Some(window) = &self.window { window.request_redraw(); }
        }
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    let _ = event_loop.run_app(&mut app);
}
