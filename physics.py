import taichi as ti
import taichi.math as tm

pi = tm.pi

# --- Bessel functions for the Airy envelope (single aperture) ---
@ti.func
def bessel_j1_series(x: ti.f64, n_terms: ti.i32 = 100) -> ti.f64:
    term = x * 0.5
    result = term
    for m in range(n_terms - 1):
        term = term * (-(x * x) / 4.0) / ((m + 1) * (m + 2))
        result += term
    return result


@ti.func
def bessel_j1_asymptotic(x: ti.f64) -> ti.f64:
    return ti.sqrt(2.0 / (pi * x)) * ti.cos(x - 3.0 * pi / 4.0)


@ti.func
def j1(x: ti.f64) -> ti.f64:
    ret = 0.0
    if ti.abs(x) < 30.0:
        ret = bessel_j1_series(x)
    else:
        ret = bessel_j1_asymptotic(x)
    # Smooth transition near x = 30
    if ti.abs(x) >= 28.0 and ti.abs(x) <= 32.0:
        alpha = (ti.abs(x) - 28.0) / 4.0
        series_val = bessel_j1_series(x)
        asymptotic_val = bessel_j1_asymptotic(x)
        ret = (1.0 - alpha) * series_val + alpha * asymptotic_val
    return ret


