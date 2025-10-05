"""Grid configuration constants and helpers for diffraction GUI."""

# Offsets for symmetric square grids
OFFSETS_4 = (-1.5, -0.5, 0.5, 1.5)
OFFSETS_5 = (-2.0, -1.0, 0.0, 1.0, 2.0)

# Angles (degrees) for 6-point circular/hex arrangements
ANGLES_6 = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)

# Small hexagonal arrangement (7 points: 1 center + 6 surrounding), relative positions in units of d
SMALL_HEXAGONAL_POINTS = (
    (1.0,   0.0),           # Right
    (0.5,   0.866025403784), # Top-right
    (-0.5,  0.866025403784), # Top-left
    (-1.0,  0.0),          # Left
    (-0.5, -0.866025403784), # Bottom-left
    (0.5,  -0.866025403784), # Bottom-right
)

# Small triangular arrangement (original 8 central points), relative positions in units of d
SMALL_TRIANGULAR_POINTS = (
    (0.0,   0.0),           # Center
    (1.0,   0.0),           # Right
    (0.5,   0.866025403784), # Top-right
    (-0.5,  0.866025403784), # Top-left
    (-1.0,  0.0),          # Left
    (-0.5, -0.866025403784), # Bottom-left
    (0.5,  -0.866025403784), # Bottom-right
)

# Large triangular arrangement (31 points), relative positions in units of d
LARGE_TRIANGULAR_POINTS = (
    (-0.5,  2.598076211353),
    (0.5,   2.598076211353),

    (-2.0,  1.732050807569),
    (-1.0,  1.732050807569),
    (0.0,   1.732050807569),
    (1.0,   1.732050807569),
    (2.0,   1.732050807569),

    (-2.5,  0.866025403784),
    (-1.5,  0.866025403784),
    (-0.5,  0.866025403784),
    (0.5,   0.866025403784),
    (1.5,   0.866025403784),
    (2.5,   0.866025403784),

    (-2.0,  0.0),
    (-1.0,  0.0),
    (0.0,   0.0),
    (1.0,   0.0),
    (2.0,   0.0),

    (-2.5, -0.866025403784),
    (-1.5, -0.866025403784),
    (-0.5, -0.866025403784),
    (0.5,  -0.866025403784),
    (1.5,  -0.866025403784),
    (2.5,  -0.866025403784),

    (-2.0, -1.732050807569),
    (-1.0, -1.732050807569),
    (0.0,  -1.732050807569),
    (1.0,  -1.732050807569),
    (2.0,  -1.732050807569),

    (-0.5, -2.598076211353),
    (0.5,  -2.598076211353),
)

# Large hexagonal arrangement (24 boundary points, no center rows fully filled)
LARGE_HEXAGONAL_POINTS = (
    (-0.5,  2.598076211353),
    (0.5,   2.598076211353),

    (-2.0,  1.732050807569),
    (-1.0,  1.732050807569),

    (1.0,   1.732050807569),
    (2.0,   1.732050807569),

    (-2.5,  0.866025403784),
    (-0.5,  0.866025403784),
    (0.5,   0.866025403784),
    (2.5,   0.866025403784),

    (-2.0,  0.0),
    (-1.0,  0.0),

    (1.0,   0.0),
    (2.0,   0.0),

    (-2.5, -0.866025403784),
    (-0.5, -0.866025403784),
    (0.5,  -0.866025403784),
    (2.5,  -0.866025403784),

    (-2.0, -1.732050807569),
    (-1.0, -1.732050807569),

    (1.0,  -1.732050807569),
    (2.0,  -1.732050807569),

    (-0.5, -2.598076211353),
    (0.5,  -2.598076211353),
)

def title_for_grid(size: int) -> str:
    if size == 6:
        return "Small Triangular Aperture Grid: Mask & Diffraction"
    if size == 7:
        return "Small Hexagonal Aperture Grid: Mask & Diffraction"
    if size == 8:
        return "Large Triangular Aperture Grid: Mask & Diffraction"
    if size == 9:
        return "Small Hexagonal (Coord) Aperture Grid: Mask & Diffraction"
    if size == 10:
        return "Large Hexagonal Aperture Grid: Mask & Diffraction"
    return f"{size}x{size} Aperture Grid: Mask & Diffraction"


