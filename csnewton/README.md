# csnewton

**Newton-GMRES/LGMRES solver using the complex-step Jacobian-vector product.**

The Jacobian-vector product `J(x)·u` is never formed explicitly. Instead it is
approximated by a single extra function evaluation at a complex point:

```
J(x)·u  ≈  Im[ F(x + i·h·u) ] / h,    h = 1e-20
```

Because standard NumPy operations (`+`, `*`, `np.sin`, `np.exp`, …) extend
naturally to `complex128` arrays, **F only needs to be written once** — the same
Python function handles both real and complex inputs automatically.

The inner linear solver is matrix-free GMRES(m) with optional restart (default)
or LGMRES, which augments each restart cycle with correction vectors from previous
cycles to prevent stagnation on problems with isolated eigenvalues.

**Reference:**
D. Mitsotakis, *"The complex-step Newton method and its convergence"*,
Numerische Mathematik **157**(3), 993–1021, 2025.
<https://link.springer.com/article/10.1007/s00211-025-01471-w>

---

## Installation

```bash
pip install csnewton
```

OpenMP parallelisation is enabled automatically when available:

| Platform | Requirement |
|----------|-------------|
| macOS    | `brew install libomp` (optional; single-threaded if absent) |
| Linux    | GCC/Clang with OpenMP support (usually present by default) |
| Windows  | MSVC with `/openmp` (enabled automatically) |

Set the number of threads at runtime:

```bash
OMP_NUM_THREADS=4 python your_script.py
```

---

## Quick start

```python
import numpy as np
import csnewton

# --- vector problem ---------------------------------------------------------
n  = 200
ci = np.arange(1, n + 1, dtype=float)

def F(x):
    return x**2 - ci          # works for float64 AND complex128

x0 = 2.0 * np.ones(n)
x, converged, newton_iters, gmres_iters, residual = csnewton.csnewton(F, x0, tol=1e-10)
print(converged, residual)    # True  <1e-10

# --- scalar problem ---------------------------------------------------------
def f(x):
    return np.log(x) + x**2 - 1

x, converged, *_ = csnewton.csnewton(f, 2.0, tol=1e-10)
print(x)                      # ≈ 1.0
```

---

## API

### `csnewton.csnewton(F, x0, ...)` — Newton-GMRES/LGMRES solver

```
csnewton(F, x0,
         max_newton=50, tol=1e-10, h_cs=1e-20,
         gmres_restart=20, gmres_max_iter=200, gmres_tol=1e-6,
         precon=None,
         use_lgmres=False, lgmres_augment=2)
→ (x, converged, newton_iters, gmres_iters, residual)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `F` | — | `callable(x) → residual`, works for `float64` and `complex128` |
| `x0` | — | Initial guess (`ndarray` or `float`) |
| `max_newton` | 50 | Maximum Newton iterations |
| `tol` | 1e-10 | `‖F(x)‖₂` stopping tolerance |
| `h_cs` | 1e-20 | Complex-step size |
| `gmres_restart` | 20 | Krylov subspace size before restart |
| `gmres_max_iter` | 200 | Max GMRES iterations per Newton step |
| `gmres_tol` | 1e-6 | Relative GMRES residual tolerance |
| `precon` | `None` | Left preconditioner `M⁻¹`: `callable(v) → M⁻¹v` |
| `use_lgmres` | `False` | Use LGMRES instead of GMRES |
| `lgmres_augment` | 2 | LGMRES augmentation vectors `k_aug` |

### `csnewton.gmres(matvec, x0, b, ...)` — real GMRES(m)

```
gmres(matvec, x0, b, restart=20, max_iter=200, tol=1e-8, precon=None)
→ (x, converged, iterations, residual)
```

### `csnewton.gmres_z(matvec, x0, b, ...)` — complex GMRES(m)

Same signature as `gmres` but operates on `complex128` arrays.

### `csnewton.lgmres(matvec, x0, b, ...)` — real LGMRES

```
lgmres(matvec, x0, b, inner=20, k_aug=2, max_iter=200, tol=1e-8, precon=None)
→ (x, converged, iterations, residual)
```

### `csnewton.lgmres_z(matvec, x0, b, ...)` — complex LGMRES

Same signature as `lgmres` but operates on `complex128` arrays.

---

## Using a preconditioner

Any callable `M_inv(v) → w` satisfying `M·w ≈ v` works as a preconditioner.
The Thomas algorithm (exact O(n) LU for tridiagonal systems) is a natural
choice for discretised PDEs:

```python
class ThomasLU:
    def __init__(self, sub, diag, sup):
        n = len(diag)
        d, l = diag.copy().astype(float), sub.copy().astype(float)
        for i in range(1, n):
            f = l[i-1] / d[i-1]; d[i] -= f * sup[i-1]; l[i-1] = f
        self.d, self.l, self.u = d, l, sup.astype(float)

    def __call__(self, rhs):
        x = rhs.copy().astype(float)
        for i in range(1, len(x)):        x[i] -= self.l[i-1] * x[i-1]
        x[-1] /= self.d[-1]
        for i in range(len(x)-2, -1, -1): x[i] = (x[i] - self.u[i]*x[i+1]) / self.d[i]
        return x

x, ok, *_ = csnewton.csnewton(F, x0, precon=ThomasLU(sub, diag, sup))
```

---

## License

MIT — see [LICENSE](LICENSE).
