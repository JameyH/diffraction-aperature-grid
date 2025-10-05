import taichi as ti
import taichi.math as tm
import sys
import os
import subprocess
from physics import j1
from color import wavelength_to_rgb_func

ti.init(default_fp=ti.f64)

pi = tm.pi

# Resolution for each image (mask and simulation)
w = 600
h = 600

# --- Fields for diffraction simulation (right half) ---
# Using 3-channel (RGB) images.
pixels_screen = ti.Vector.field(3, dtype=ti.u8, shape=(w, h))
intensity_field = ti.field(dtype=ti.f64, shape=(w, h))

# Physical simulation parameters (in SI units)
W = ti.field(dtype=ti.f64, shape=())         # Aperture diameter (meters)
z = ti.field(dtype=ti.f64, shape=())         # Distance to screen (meters)
screen_size = ti.field(dtype=ti.f64, shape=()) # Physical screen size (meters)
d = ti.field(dtype=ti.f64, shape=())         # Center-to-center separation (meters)
lam_field = ti.field(dtype=ti.f64, shape=())   # Wavelength (meters)

reference_intensity = ti.field(dtype=ti.f64, shape=())

# Grid size configuration
grid_size = ti.field(dtype=ti.i32, shape=())  # 3 for 3x3, 4 for 4x4

# Global base color (normalized RGB vector) for the diffraction image.
base_color = ti.Vector.field(3, dtype=ti.f64, shape=())

# --- macOS focus helper ---
def _bring_app_to_front_macos():
    if sys.platform == "darwin":
        pid = os.getpid()
        script = f'tell application "System Events" to set frontmost of (first process whose unix id is {pid}) to true'
        try:
            subprocess.run(["osascript", "-e", script], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

# (Bessel functions moved to physics.py)

# --- Diffraction Simulation for 3x3 Grid of Apertures (right half) ---
@ti.kernel
def compute_intensity():
    for i, j in intensity_field:
        amp = 0.0  # Declare amp
        # Convert pixel indices to physical screen coordinates (meters)
        x_coord = (i - w / 2.0) * (screen_size[None] / w)
        y_coord = (j - h / 2.0) * (screen_size[None] / h)
        r_m = ti.sqrt(x_coord * x_coord + y_coord * y_coord)

        if ti.abs(r_m) < 1e-12:
            # At the center, each aperture gives π/2 amplitude;
            # For N x N apertures in phase, total amplitude is N*N*(π/2)
            num_apertures = grid_size[None] * grid_size[None]
            amp = (pi / 2.0) * num_apertures
        else:
            # Single-aperture Airy envelope:
            x_val = pi * W[None] * r_m / (lam_field[None] * z[None])
            envelope = ti.abs(lam_field[None] * z[None] * j1(x_val) / (W[None] * r_m))
            real_sum = 0.0
            imag_sum = 0.0

            # Dynamic grid based on grid_size
            if grid_size[None] == 3:
                # 3x3 grid: positions at (-d, 0, d) in both x and y
                for m in ti.static(range(-1, 2)):
                    for n in ti.static(range(-1, 2)):
                        phase = (2 * pi / lam_field[None]) * ((m * d[None] * x_coord + n * d[None] * y_coord) / z[None])
                        real_sum += ti.cos(phase)
                        imag_sum += ti.sin(phase)
            else:  # grid_size == 4
                # 4x4 grid: positions at (-1.5d, -0.5d, 0.5d, 1.5d) in both x and y
                for m in ti.static(range(-2, 2)):
                    for n in ti.static(range(-2, 2)):
                        m_pos = m * 0.5  # -1.5, -0.5, 0.5, 1.5
                        n_pos = n * 0.5  # -1.5, -0.5, 0.5, 1.5
                        phase = (2 * pi / lam_field[None]) * ((m_pos * d[None] * x_coord + n_pos * d[None] * y_coord) / z[None])
                        real_sum += ti.cos(phase)
                        imag_sum += ti.sin(phase)

            amp = envelope * ti.sqrt(real_sum * real_sum + imag_sum * imag_sum)
        intensity_field[i, j] = amp * amp

@ti.kernel
def normalize_intensity():
    for i, j in intensity_field:
        scaled_intensity = ti.sqrt(intensity_field[i, j] / reference_intensity[None])
        norm_val = ti.max(0.0, ti.min(1.0, scaled_intensity))
        col = base_color[None] * norm_val
        pixels_screen[i, j] = ti.Vector([ti.u8(col[0] * 255),
                                         ti.u8(col[1] * 255),
                                         ti.u8(col[2] * 255)])

@ti.kernel
def compute_reference_intensity():
    num_apertures = grid_size[None] * grid_size[None]
    reference_intensity[None] = (num_apertures * (pi / 2.0))**2

# --- Mask Drawing: Show 3x3 grid of apertures as circles (left half) ---
pixels_mask = ti.Vector.field(3, dtype=ti.u8, shape=(w, h))

@ti.kernel
def draw_mask():
    scale: ti.f64 = 1250.0  # pixels per mm
    aperture_mm = W[None] * 1e3
    separation_mm = d[None] * 1e3
    aperture_pixel_radius = aperture_mm * scale
    separation_pixel = separation_mm * scale
    cx = w // 2
    cy = h // 2

    # Clear the mask to black.
    for i, j in pixels_mask:
        pixels_mask[i, j] = ti.Vector([ti.u8(0), ti.u8(0), ti.u8(0)])

    # Draw circles for all apertures in a single loop
    for i, j in pixels_mask:
        # Check all aperture positions for both grid sizes
        if grid_size[None] == 3:
            # 3x3 grid: positions at (-d, 0, d) in both x and y
            for m in ti.static(range(-1, 2)):
                for n in ti.static(range(-1, 2)):
                    cx_i = cx + ti.cast(m * separation_pixel, ti.i32)
                    cy_i = cy + ti.cast(n * separation_pixel, ti.i32)
                    if ti.sqrt((i - cx_i)**2 + (j - cy_i)**2) < aperture_pixel_radius:
                        pixels_mask[i, j] = ti.Vector([ti.u8(255), ti.u8(255), ti.u8(255)])
        else:  # grid_size == 4
            # 4x4 grid: positions at (-1.5d, -0.5d, 0.5d, 1.5d) in both x and y
            for m in ti.static(range(-2, 2)):
                for n in ti.static(range(-2, 2)):
                    m_pos = m * 0.5  # -1.5, -0.5, 0.5, 1.5
                    n_pos = n * 0.5  # -1.5, -0.5, 0.5, 1.5
                    cx_i = cx + ti.cast(m_pos * separation_pixel, ti.i32)
                    cy_i = cy + ti.cast(n_pos * separation_pixel, ti.i32)
                    if ti.sqrt((i - cx_i)**2 + (j - cy_i)**2) < aperture_pixel_radius:
                        pixels_mask[i, j] = ti.Vector([ti.u8(255), ti.u8(255), ti.u8(255)])

# --- Combined Display: Left shows mask; right shows diffraction pattern ---
combined_pixels = ti.Vector.field(3, dtype=ti.u8, shape=(w * 2, h))

@ti.kernel
def combine_pixels():
    for i, j in combined_pixels:
        if i < w:
            combined_pixels[i, j] = pixels_mask[i, j]
        else:
            combined_pixels[i, j] = pixels_screen[i - w, j]

"""Wavelength-to-RGB conversion moved to color.py"""

# --- Kernel to update the base color using the ti.func ---
@ti.kernel
def update_base_color(wavelength: ti.f64):
    base_color[None] = wavelength_to_rgb_func(wavelength)

# --- GUI Setup ---
if __name__ == "__main__":
    gui = ti.GUI(f"{grid_size[None]}x{grid_size[None]} Aperture Grid: Mask & Diffraction", (w * 2, h))
    _bring_app_to_front_macos()

    # --- Initialize Simulation Parameters ---
    W[None] = 0.02e-3      # Aperture diameter (meters)
    z[None] = 10.0         # Distance to screen (meters)
    screen_size[None] = 2.0  # Physical screen size (meters)
    d[None] = 0.05e-3      # Center-to-center separation (meters)
    lam_field[None] = 500e-9  # Wavelength (meters)
    grid_size[None] = 3    # Start with 3x3 grid

    compute_reference_intensity()

    # --- Sliders for Adjusting Parameters ---
    z_slider = gui.slider("Distance (m)", 1.0, 20.0)
    W_slider = gui.slider("Aperture (mm)", 0.01, 0.10)
    d_slider = gui.slider("Separation (mm)", 0.0, 0.2)
    lam_slider = gui.slider("Wavelength (nm)", 400, 700)

    # Set initial slider positions.
    z_slider.value = z[None]
    W_slider.value = W[None] * 1e3
    d_slider.value = d[None] * 1e3
    lam_slider.value = lam_field[None] * 1e9

    # --- Buttons for Grid Selection ---
    grid_3x3_btn = gui.button('3x3 Grid')
    grid_4x4_btn = gui.button('4x4 Grid')

    # --- Main Loop ---
    while gui.running:
        z[None] = z_slider.value
        W[None] = W_slider.value * 1e-3
        d[None] = d_slider.value * 1e-3
        lam_field[None] = lam_slider.value * 1e-9

        # Update the base color using the Taichi function.
        update_base_color(lam_slider.value)

        # Handle button events for grid selection
        for e in gui.get_events(gui.PRESS):
            if e.key == grid_3x3_btn:
                grid_size[None] = 3
                compute_reference_intensity()
                gui.title = f"{grid_size[None]}x{grid_size[None]} Aperture Grid: Mask & Diffraction"
            elif e.key == grid_4x4_btn:
                grid_size[None] = 4
                compute_reference_intensity()
                gui.title = f"{grid_size[None]}x{grid_size[None]} Aperture Grid: Mask & Diffraction"

        compute_intensity()
        normalize_intensity()
        draw_mask()
        combine_pixels()

        gui.set_image(combined_pixels)
        gui.show()
        # Attempt to bring to front after first frame as some WMs ignore early requests
        if ti.static(False):
            _bring_app_to_front_macos()