import taichi as ti
import taichi.math as tm
import sys
import os
import subprocess
from physics import j1
from color import wavelength_to_rgb_func
from grids import (
    title_for_grid,
    OFFSETS_4,
    OFFSETS_5,
    SMALL_HEXAGONAL_POINTS,
    SMALL_TRIANGULAR_POINTS,
    LARGE_TRIANGULAR_POINTS,
    LARGE_HEXAGONAL_POINTS,
)

ti.init(default_fp=ti.f64)

pi = tm.pi

# Precomputed counts for grid options
SMALL_TRI_COUNT = len(SMALL_TRIANGULAR_POINTS)
SMALL_HEX_COUNT = len(SMALL_HEXAGONAL_POINTS)
LARGE_TRI_COUNT = len(LARGE_TRIANGULAR_POINTS)
LARGE_HEX_COUNT = len(LARGE_HEXAGONAL_POINTS)

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
        num_apertures = 0.0  # For center calculation
        real_sum = 0.0  # For interference calculation
        imag_sum = 0.0  # For interference calculation
        x_pos = 0.0  # For hexagonal pattern positions
        y_pos = 0.0  # For hexagonal pattern positions
        # Convert pixel indices to physical screen coordinates (meters)
        x_coord = (i - w / 2.0) * (screen_size[None] / w)
        y_coord = (j - h / 2.0) * (screen_size[None] / h)
        r_m = ti.sqrt(x_coord * x_coord + y_coord * y_coord)

        if ti.abs(r_m) < 1e-12:
            # At the center, each aperture gives π/2 amplitude;
            # For square grids, total amplitude is N*N*(π/2)
            # For hexagonal/triangular grids, use the actual number of apertures
            if grid_size[None] == 6:  # Small triangular
                num_apertures = ti.cast(SMALL_TRI_COUNT, ti.f64)
            elif grid_size[None] == 7:  # Small hexagonal
                num_apertures = ti.cast(SMALL_HEX_COUNT, ti.f64)
            elif grid_size[None] == 8:  # Large triangular
                num_apertures = ti.cast(LARGE_TRI_COUNT, ti.f64)
            elif grid_size[None] == 10:  # Large hexagonal
                num_apertures = ti.cast(LARGE_HEX_COUNT, ti.f64)
            else:
                num_apertures = grid_size[None] * grid_size[None]
            amp = (tm.pi / 2.0) * num_apertures
        else:
            # Single-aperture Airy envelope:
            x_val = tm.pi * W[None] * r_m / (lam_field[None] * z[None])
            envelope = ti.abs(lam_field[None] * z[None] * j1(x_val) / (W[None] * r_m))

            # Dynamic grid based on grid_size
            if grid_size[None] == 3:
                # 3x3 grid: positions at (-d, 0, d) in both x and y
                for m in ti.static(range(-1, 2)):
                    for n in ti.static(range(-1, 2)):
                        phase = (2 * tm.pi / lam_field[None]) * ((m * d[None] * x_coord + n * d[None] * y_coord) / z[None])
                        real_sum += ti.cos(phase)
                        imag_sum += ti.sin(phase)
            elif grid_size[None] == 4:
                # 4x4 grid: positions that expand symmetrically from center
                for m in ti.static(OFFSETS_4):
                    for n in ti.static(OFFSETS_4):
                        phase = (2 * tm.pi / lam_field[None]) * ((m * d[None] * x_coord + n * d[None] * y_coord) / z[None])
                        real_sum += ti.cos(phase)
                        imag_sum += ti.sin(phase)
            elif grid_size[None] == 5:
                # 5x5 grid: positions at (-2d, -d, 0, d, 2d) in both x and y
                for m in ti.static(OFFSETS_5):
                    for n in ti.static(OFFSETS_5):
                        phase = (2 * tm.pi / lam_field[None]) * ((m * d[None] * x_coord + n * d[None] * y_coord) / z[None])
                        real_sum += ti.cos(phase)
                        imag_sum += ti.sin(phase)
            elif grid_size[None] == 6:  # Small triangular pattern (constant-defined)
                # Use the exact coordinates provided for proper triangular lattice
                for x_pos, y_pos in ti.static(SMALL_TRIANGULAR_POINTS):
                    phase = (2 * tm.pi / lam_field[None]) * ((x_pos * d[None] * x_coord + y_pos * d[None] * y_coord) / z[None])
                    real_sum += ti.cos(phase)
                    imag_sum += ti.sin(phase)
            elif grid_size[None] == 7:  # Small hexagonal pattern (constant-defined)
                for x_pos, y_pos in ti.static(SMALL_HEXAGONAL_POINTS):
                    phase = (2 * tm.pi / lam_field[None]) * ((x_pos * d[None] * x_coord + y_pos * d[None] * y_coord) / z[None])
                    real_sum += ti.cos(phase)
                    imag_sum += ti.sin(phase)
            elif grid_size[None] == 8:  # Large triangular pattern (constant-defined)
                for x_pos, y_pos in ti.static(LARGE_TRIANGULAR_POINTS):
                    phase = (2 * tm.pi / lam_field[None]) * ((x_pos * d[None] * x_coord + y_pos * d[None] * y_coord) / z[None])
                    real_sum += ti.cos(phase)
                    imag_sum += ti.sin(phase)
            elif grid_size[None] == 10:  # Large hexagonal pattern (constant-defined)
                for x_pos, y_pos in ti.static(LARGE_HEXAGONAL_POINTS):
                    phase = (2 * tm.pi / lam_field[None]) * ((x_pos * d[None] * x_coord + y_pos * d[None] * y_coord) / z[None])
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
    num_apertures = 0.0  # Declare variable first
    if grid_size[None] == 6:  # Small triangular
        num_apertures = ti.cast(SMALL_TRI_COUNT, ti.f64)
    elif grid_size[None] == 7:  # Small hexagonal
        num_apertures = ti.cast(SMALL_HEX_COUNT, ti.f64)
    elif grid_size[None] == 8:  # Large triangular
        num_apertures = ti.cast(LARGE_TRI_COUNT, ti.f64)
    elif grid_size[None] == 10:  # Large hexagonal
        num_apertures = ti.cast(LARGE_HEX_COUNT, ti.f64)
    else:
        num_apertures = grid_size[None] * grid_size[None]
    reference_intensity[None] = (num_apertures * (tm.pi / 2.0))**2

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

    # Declare variables for hexagonal pattern
    offset_x = 0.0
    offset_y = 0.0

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
        elif grid_size[None] == 4:
            # 4x4 grid: positions that expand symmetrically from center
            # Use explicit positions to ensure perfect centering
            for m in ti.static([-1.5, -0.5, 0.5, 1.5]):
                for n in ti.static([-1.5, -0.5, 0.5, 1.5]):
                    cx_i = cx + ti.cast(m * separation_pixel, ti.i32)
                    cy_i = cy + ti.cast(n * separation_pixel, ti.i32)
                    if ti.sqrt((i - cx_i)**2 + (j - cy_i)**2) < aperture_pixel_radius:
                        pixels_mask[i, j] = ti.Vector([ti.u8(255), ti.u8(255), ti.u8(255)])
        elif grid_size[None] == 5:
            # 5x5 grid: positions at (-2d, -d, 0, d, 2d) in both x and y
            for m in ti.static([-2.0, -1.0, 0.0, 1.0, 2.0]):
                for n in ti.static([-2.0, -1.0, 0.0, 1.0, 2.0]):
                    cx_i = cx + ti.cast(m * separation_pixel, ti.i32)
                    cy_i = cy + ti.cast(n * separation_pixel, ti.i32)
                    if ti.sqrt((i - cx_i)**2 + (j - cy_i)**2) < aperture_pixel_radius:
                        pixels_mask[i, j] = ti.Vector([ti.u8(255), ti.u8(255), ti.u8(255)])
        elif grid_size[None] == 6:  # Small triangular pattern
            for x_pos, y_pos in ti.static(SMALL_TRIANGULAR_POINTS):
                cx_i = cx + ti.cast(x_pos * separation_pixel, ti.i32)
                cy_i = cy + ti.cast(y_pos * separation_pixel, ti.i32)
                if ti.sqrt((i - cx_i)**2 + (j - cy_i)**2) < aperture_pixel_radius:
                    pixels_mask[i, j] = ti.Vector([ti.u8(255), ti.u8(255), ti.u8(255)])
        elif grid_size[None] == 7:  # Small hexagonal pattern
            for x_pos, y_pos in ti.static(SMALL_HEXAGONAL_POINTS):
                cx_i = cx + ti.cast(x_pos * separation_pixel, ti.i32)
                cy_i = cy + ti.cast(y_pos * separation_pixel, ti.i32)
                if ti.sqrt((i - cx_i)**2 + (j - cy_i)**2) < aperture_pixel_radius:
                    pixels_mask[i, j] = ti.Vector([ti.u8(255), ti.u8(255), ti.u8(255)])
        elif grid_size[None] == 8:  # Large triangular pattern
            for x_pos, y_pos in ti.static(LARGE_TRIANGULAR_POINTS):
                cx_i = cx + ti.cast(x_pos * separation_pixel, ti.i32)
                cy_i = cy + ti.cast(y_pos * separation_pixel, ti.i32)
                if ti.sqrt((i - cx_i)**2 + (j - cy_i)**2) < aperture_pixel_radius:
                    pixels_mask[i, j] = ti.Vector([ti.u8(255), ti.u8(255), ti.u8(255)])
        elif grid_size[None] == 10:  # Large hexagonal pattern
            for x_pos, y_pos in ti.static(LARGE_HEXAGONAL_POINTS):
                cx_i = cx + ti.cast(x_pos * separation_pixel, ti.i32)
                cy_i = cy + ti.cast(y_pos * separation_pixel, ti.i32)
                if ti.sqrt((i - cx_i)**2 + (j - cy_i)**2) < aperture_pixel_radius:
                    pixels_mask[i, j] = ti.Vector([ti.u8(255), ti.u8(255), ti.u8(255)])

# --- Combined Display: Left shows mask; right shows diffraction pattern ---
combined_pixels = ti.Vector.field(3, dtype=ti.u8, shape=(w * 2, h))
combined_pixels_f32 = ti.Vector.field(3, dtype=ti.f32, shape=(w * 2, h))

@ti.kernel
def combine_pixels():
    for i, j in combined_pixels:
        if i < w:
            combined_pixels[i, j] = pixels_mask[i, j]
        else:
            combined_pixels[i, j] = pixels_screen[i - w, j]

@ti.kernel
def convert_combined_to_f32():
    for i, j in combined_pixels:
        c = combined_pixels[i, j]
        combined_pixels_f32[i, j] = ti.Vector([
            ti.cast(c[0], ti.f32) / 255.0,
            ti.cast(c[1], ti.f32) / 255.0,
            ti.cast(c[2], ti.f32) / 255.0,
        ])

"""Wavelength-to-RGB conversion moved to color.py"""

# --- Kernel to update the base color using the ti.func ---
@ti.kernel
def update_base_color(wavelength: ti.f64):
    base_color[None] = wavelength_to_rgb_func(wavelength)

# --- GUI Setup ---
# Helper to set window title across APIs
def _set_window_title(win_obj, title: str):
    try:
        win_obj.title = title
    except Exception:
        pass
if __name__ == "__main__":
    # Initialize grid selection before setting title
    grid_size[None] = 3    # Start with 3x3 grid
    initial_title = title_for_grid(grid_size[None])

    win = ti.ui.Window(initial_title, (w * 2, h))
    canvas = win.get_canvas()
    gui = win.get_gui()
    _bring_app_to_front_macos()

    # --- Initialize Simulation Parameters ---
    W[None] = 0.02e-3      # Aperture diameter (meters)
    z[None] = 10.0         # Distance to screen (meters)
    screen_size[None] = 2.0  # Physical screen size (meters)
    d[None] = 0.05e-3      # Center-to-center separation (meters)
    lam_field[None] = 500e-9  # Wavelength (meters)

    compute_reference_intensity()
    
    # --- Control Panel State (sliders return values directly) ---
    z_val = z[None]
    W_val = W[None] * 1e3
    d_val = d[None] * 1e3
    lam_val = lam_field[None] * 1e9

    # Control panel layout (top-left)
    panel_margin = 0.02
    panel_width = .17
    panel_height = .8
    panel_y = 0

    # --- Main Loop ---
    while win.running:
        # ---- Top Control Panel (top-left) ----
        with gui.sub_window("Controls", 0, panel_y, panel_width, panel_height):
            # Get current slider values (sliders return values directly)
            z_val = gui.slider_float("Distance (m)", z_val, 1.0, 40.0)
            W_val = gui.slider_float("Aperture (mm)", W_val, 0.01, 0.10)
            d_val = gui.slider_float("Separation (mm)", d_val, 0.00, 0.20)
            lam_val = gui.slider_float("Wavelength (nm)", lam_val, 400.0, 700.0)

            gui.text("Square Grids:")
            # Create buttons and handle their events (all in same GUI context)
            if gui.button("3x3"):
                grid_size[None] = 3
                compute_reference_intensity()
                _set_window_title(win, title_for_grid(3))
            if gui.button("4x4"):
                grid_size[None] = 4
                compute_reference_intensity()
                _set_window_title(win, title_for_grid(4))
            if gui.button("5x5"):
                grid_size[None] = 5
                compute_reference_intensity()
                _set_window_title(win, title_for_grid(5))

            gui.text("Hexagonal Grids:")
            if gui.button(f"Small Hex ({SMALL_HEX_COUNT})"):
                grid_size[None] = 7
                compute_reference_intensity()
                _set_window_title(win, title_for_grid(7))
            if gui.button(f"Large Hex ({LARGE_HEX_COUNT})"):
                grid_size[None] = 10
                compute_reference_intensity()
                _set_window_title(win, title_for_grid(10))

            gui.text("Triangular Grids:")
            if gui.button(f"Small Tri ({SMALL_TRI_COUNT})"):
                grid_size[None] = 6
                compute_reference_intensity()
                _set_window_title(win, title_for_grid(6))
            if gui.button(f"Large Tri ({LARGE_TRI_COUNT})"):
                grid_size[None] = 8
                compute_reference_intensity()
                _set_window_title(win, title_for_grid(8))

        # Apply slider values to simulation fields
        z[None] = z_val
        W[None] = W_val * 1e-3
        d[None] = d_val * 1e-3
        lam_field[None] = lam_val * 1e-9

        # Update the base color using the wavelength in nm
        update_base_color(lam_val)

        compute_intensity()
        normalize_intensity()
        draw_mask()
        combine_pixels()

        convert_combined_to_f32()
        canvas.set_image(combined_pixels_f32)
        win.show()
        # Attempt to bring to front after first frame as some WMs ignore early requests
        if ti.static(False):
            _bring_app_to_front_macos()