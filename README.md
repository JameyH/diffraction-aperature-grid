# Aperture Grid Diffraction Simulation

A Taichi-based simulation demonstrating diffraction patterns from a grid of circular apertures.

<video src="images/example.mp4" controls width="640" poster="images/example.png"></video>

## Overview

This project simulates diffraction patterns for multiple aperture-grid lattices (square, triangular, and hexagonal). The simulation displays both the aperture mask and the resulting diffraction pattern side-by-side.
## Grid Types

- **Square grids**: 3x3, 4x4, 5x5
- **Triangular grids**:
  - Small triangular: 8 apertures
  - Large triangular: 31 apertures
- **Hexagonal grids**:
  - Small hexagonal: 7 apertures (center + 6 neighbors)
  - Large hexagonal: 24 apertures

Use the buttons in the left control panel to switch between grid types.


## Physics Background

The simulation implements the physical optics calculation for:
- **Single aperture diffraction**: Uses Bessel functions to compute the Airy disk pattern
- **Multi-aperture interference**: Calculates phase differences between apertures in the grid
- **Far-field approximation**: Assumes Fraunhofer diffraction conditions

## Features

- **Interactive parameters**: Real-time adjustment of aperture size, separation, distance, and wavelength
- **Visual comparison**: Side-by-side display of aperture mask and diffraction pattern
- **Color mapping**: Diffraction pattern colored according to the incident light wavelength
- **High performance**: GPU-accelerated computation using Taichi

## Parameters

- **Aperture diameter (W)**: Size of each circular aperture (meters)
- **Center-to-center separation (d)**: Distance between adjacent apertures (meters)
- **Distance to screen (z)**: Propagation distance from apertures to observation plane (meters)
- **Wavelength (λ)**: Wavelength of incident light (meters)
- **Screen size**: Physical size of the observation area (meters)

## Usage

Run the simulation:
```bash
python main.py
```

Use the interactive sliders to adjust:
- Distance (1.0 - 40.0 m)
- Aperture size (0.01 - 0.10 mm)
- Separation (0.0 - 0.2 mm)
- Wavelength (400 - 700 nm)

## Implementation Details

- **Bessel function computation**: Custom implementation using series expansion and asymptotic approximation (see `physics.py`)
- **Phase calculation**: Accounts for geometric path differences between apertures
- **Color rendering**: Wavelength-to-RGB conversion for visual representation (see `color.py`)
- **Normalization**: Intensity scaling for proper display

## Project Structure

- `main.py`: Entry point, configuration, Taichi field definitions, kernels, GUI loop
- `physics.py`: Airy/Bessel-related Taichi functions
- `color.py`: Wavelength to RGB Taichi function
- `grids.py`: Grid constants (point sets and offsets) and grid title helper

## Dependencies

- Taichi: For high-performance GPU computing
  (Install via your Python environment manager or `pip install taichi`.)

## Mathematical Foundation

The diffraction pattern combines:
1. **Airy pattern** from individual apertures: `J₁(x)/x` envelope function
2. **Interference pattern** from aperture array: Complex amplitude summation
3. **Phase factor**: `(2π/λ) * (aperture_positions · direction) / z`

Where `J₁` is the first-order Bessel function of the first kind.
