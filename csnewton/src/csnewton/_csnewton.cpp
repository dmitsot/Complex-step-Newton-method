// csnewton_py.cpp — pybind11 bindings for the csnewton module
//
// Exposes all algorithms in one Python module:
//   csnewton.gmres(matvec, x0, b, ...)   — real GMRES(m)
//   csnewton.gmres_z(matvec, x0, b, ...) — complex GMRES(m)
//   csnewton.csnewton(F, x0, ...)        — complex-step Newton (vector)
//   csnewton.csnewton(f, x0_scalar, ...) — complex-step Newton (scalar)
//
// F : callable that works for both ndarray float64 and ndarray complex128.
//     NumPy operations (arithmetic, np.sin, np.exp, etc.) satisfy this
//     automatically — the same Python function handles both dtypes.
//
// Build:  see Makefile target 'csnewton_py'

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "csnewton.hpp"

namespace py = pybind11;
using C = std::complex<double>;

// ---------------------------------------------------------------------------
// Bridge helpers: call Python F and return a C++ vector.
// forcecast ensures contiguous layout regardless of what the user returns.
// ---------------------------------------------------------------------------

static std::vector<double>
call_real_F(py::object& fn, int n, const std::vector<double>& x)
{
    using arr_t = py::array_t<double, py::array::c_style | py::array::forcecast>;
    auto x_arr = py::array_t<double>({(py::ssize_t)n}, x.data());
    auto res   = fn(x_arr).cast<arr_t>();
    if (static_cast<int>(res.size()) != n)
        throw std::runtime_error("newton_gmres: F returned wrong length");
    return std::vector<double>(res.data(), res.data() + n);
}

static std::vector<C>
call_cmplx_F(py::object& fn, int n, const std::vector<C>& x)
{
    using arr_t = py::array_t<C, py::array::c_style | py::array::forcecast>;
    auto x_arr = py::array_t<C>({(py::ssize_t)n}, x.data());
    auto res   = fn(x_arr).cast<arr_t>();
    if (static_cast<int>(res.size()) != n)
        throw std::runtime_error("newton_gmres: F returned wrong length (complex)");
    return std::vector<C>(res.data(), res.data() + n);
}

// ---------------------------------------------------------------------------
// BridgeF: wraps a single Python callable so it satisfies the generic
// C++ template Op that must be callable with both vector<double> and
// vector<complex<double>>.
// ---------------------------------------------------------------------------

struct BridgeF {
    py::object& fn;
    int n;
    std::vector<double> operator()(const std::vector<double>& x) const {
        return call_real_F(fn, n, x);
    }
    std::vector<C> operator()(const std::vector<C>& x) const {
        return call_cmplx_F(fn, n, x);
    }
};

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_csnewton, m)
{
    m.doc() =
        "Complex-step Newton solver + matrix-free GMRES(m).\n\n"
        "Reference: D. Mitsotakis, "
        "\"The complex-step Newton method and its convergence\", "
        "Numerische Mathematik 157(3), 993-1021, 2025. "
        "https://link.springer.com/article/10.1007/s00211-025-01471-w";

    // -----------------------------------------------------------------------
    // gmres — real double (float64) solver
    // -----------------------------------------------------------------------
    m.def("gmres",
        [](py::object matvec_fn,
           py::array_t<double, py::array::c_style | py::array::forcecast> x0,
           py::array_t<double, py::array::c_style | py::array::forcecast> b,
           int    restart,
           int    max_iter,
           double tol,
           py::object precon_fn)
        {
            int n = static_cast<int>(b.size());
            if (x0.size() != static_cast<py::ssize_t>(n))
                throw std::invalid_argument("gmres: x0 and b must have the same length");

            std::vector<double> xv(x0.data(), x0.data() + n);
            std::vector<double> bv(b.data(),  b.data()  + n);

            auto mv = [&](const std::vector<double>& in, std::vector<double>& out) {
                out = call_real_F(matvec_fn, n, in);
            };

            csnewton::GMRESResult<double> res;
            if (precon_fn.is_none()) {
                res = csnewton::gmres<double>(mv, xv, bv, restart, max_iter, tol);
            } else {
                auto pc = [&](const std::vector<double>& in, std::vector<double>& out) {
                    out = call_real_F(precon_fn, n, in);
                };
                res = csnewton::gmres<double>(mv, pc, xv, bv, restart, max_iter, tol);
            }

            py::array_t<double> x_out({(py::ssize_t)n});
            std::copy(xv.begin(), xv.end(), x_out.mutable_data());
            return py::make_tuple(x_out, res.converged, res.iterations, res.residual);
        },
        py::arg("matvec"),
        py::arg("x0"),
        py::arg("b"),
        py::arg("restart")  = 20,
        py::arg("max_iter") = 200,
        py::arg("tol")      = 1e-8,
        py::arg("precon")   = py::none(),
        R"doc(
Solve A*x = b using matrix-free GMRES(m) with restart (real/float64).

Parameters
----------
matvec   : callable(v: ndarray) -> ndarray   — computes A*v
x0       : ndarray float64                   — initial guess
b        : ndarray float64                   — right-hand side
restart  : int, default 20                   — Krylov subspace size before restart
max_iter : int, default 200                  — max total Arnoldi steps
tol      : float, default 1e-8               — relative residual tolerance ||r||/||b||
precon   : callable or None                  — optional left preconditioner M^{-1}*v

Returns
-------
(x, converged, iterations, residual)
)doc");

    // -----------------------------------------------------------------------
    // gmres_z — complex double (complex128) solver
    // -----------------------------------------------------------------------
    m.def("gmres_z",
        [](py::object matvec_fn,
           py::array_t<C, py::array::c_style | py::array::forcecast> x0,
           py::array_t<C, py::array::c_style | py::array::forcecast> b,
           int    restart,
           int    max_iter,
           double tol,
           py::object precon_fn)
        {
            int n = static_cast<int>(b.size());
            if (x0.size() != static_cast<py::ssize_t>(n))
                throw std::invalid_argument("gmres_z: x0 and b must have the same length");

            std::vector<C> xv(x0.data(), x0.data() + n);
            std::vector<C> bv(b.data(),  b.data()  + n);

            auto mv = [&](const std::vector<C>& in, std::vector<C>& out) {
                out = call_cmplx_F(matvec_fn, n, in);
            };

            csnewton::GMRESResult<C> res;
            if (precon_fn.is_none()) {
                res = csnewton::gmres<C>(mv, xv, bv, restart, max_iter, tol);
            } else {
                auto pc = [&](const std::vector<C>& in, std::vector<C>& out) {
                    out = call_cmplx_F(precon_fn, n, in);
                };
                res = csnewton::gmres<C>(mv, pc, xv, bv, restart, max_iter, tol);
            }

            py::array_t<C> x_out({(py::ssize_t)n});
            std::copy(xv.begin(), xv.end(), x_out.mutable_data());
            return py::make_tuple(x_out, res.converged, res.iterations, res.residual);
        },
        py::arg("matvec"),
        py::arg("x0"),
        py::arg("b"),
        py::arg("restart")  = 20,
        py::arg("max_iter") = 200,
        py::arg("tol")      = 1e-8,
        py::arg("precon")   = py::none(),
        R"doc(
Solve A*x = b using matrix-free GMRES(m) with restart (complex/complex128).

Parameters
----------
matvec   : callable(v: ndarray complex128) -> ndarray   — computes A*v
x0       : ndarray complex128                           — initial guess
b        : ndarray complex128                           — right-hand side
restart  : int, default 20
max_iter : int, default 200
tol      : float, default 1e-8
precon   : callable or None

Returns
-------
(x, converged, iterations, residual)
)doc");

    // -----------------------------------------------------------------------
    // lgmres — real double (float64) LGMRES solver
    // -----------------------------------------------------------------------
    m.def("lgmres",
        [](py::object matvec_fn,
           py::array_t<double, py::array::c_style | py::array::forcecast> x0,
           py::array_t<double, py::array::c_style | py::array::forcecast> b,
           int    inner,
           int    k_aug,
           int    max_iter,
           double tol,
           py::object precon_fn)
        {
            int n = static_cast<int>(b.size());
            if (x0.size() != static_cast<py::ssize_t>(n))
                throw std::invalid_argument("lgmres: x0 and b must have the same length");

            std::vector<double> xv(x0.data(), x0.data() + n);
            std::vector<double> bv(b.data(),  b.data()  + n);

            auto mv = [&](const std::vector<double>& in, std::vector<double>& out) {
                out = call_real_F(matvec_fn, n, in);
            };

            csnewton::GMRESResult<double> res;
            if (precon_fn.is_none()) {
                res = csnewton::lgmres<double>(mv, xv, bv, inner, k_aug, max_iter, tol);
            } else {
                auto pc = [&](const std::vector<double>& in, std::vector<double>& out) {
                    out = call_real_F(precon_fn, n, in);
                };
                res = csnewton::lgmres<double>(mv, pc, xv, bv, inner, k_aug, max_iter, tol);
            }

            py::array_t<double> x_out({(py::ssize_t)n});
            std::copy(xv.begin(), xv.end(), x_out.mutable_data());
            return py::make_tuple(x_out, res.converged, res.iterations, res.residual);
        },
        py::arg("matvec"),
        py::arg("x0"),
        py::arg("b"),
        py::arg("inner")    = 20,
        py::arg("k_aug")    = 2,
        py::arg("max_iter") = 200,
        py::arg("tol")      = 1e-8,
        py::arg("precon")   = py::none(),
        R"doc(
Solve A*x = b using LGMRES (Loose GMRES) with augmented Krylov subspace (real/float64).

At each restart cycle the subspace is augmented with k_aug correction vectors
retained from previous cycles, preventing information loss at restart.

Parameters
----------
matvec  : callable(v: ndarray) -> ndarray   — computes A*v
x0      : ndarray float64                   — initial guess
b       : ndarray float64                   — right-hand side
inner   : int, default 20                   — Arnoldi steps per cycle
k_aug   : int, default 2                    — augmentation vectors (0 = standard GMRES)
max_iter: int, default 200                  — max total steps
tol     : float, default 1e-8               — relative residual tolerance
precon  : callable or None                  — optional left preconditioner

Returns
-------
(x, converged, iterations, residual)
)doc");

    // -----------------------------------------------------------------------
    // lgmres_z — complex double (complex128) LGMRES solver
    // -----------------------------------------------------------------------
    m.def("lgmres_z",
        [](py::object matvec_fn,
           py::array_t<C, py::array::c_style | py::array::forcecast> x0,
           py::array_t<C, py::array::c_style | py::array::forcecast> b,
           int    inner,
           int    k_aug,
           int    max_iter,
           double tol,
           py::object precon_fn)
        {
            int n = static_cast<int>(b.size());
            if (x0.size() != static_cast<py::ssize_t>(n))
                throw std::invalid_argument("lgmres_z: x0 and b must have the same length");

            std::vector<C> xv(x0.data(), x0.data() + n);
            std::vector<C> bv(b.data(),  b.data()  + n);

            auto mv = [&](const std::vector<C>& in, std::vector<C>& out) {
                out = call_cmplx_F(matvec_fn, n, in);
            };

            csnewton::GMRESResult<C> res;
            if (precon_fn.is_none()) {
                res = csnewton::lgmres<C>(mv, xv, bv, inner, k_aug, max_iter, tol);
            } else {
                auto pc = [&](const std::vector<C>& in, std::vector<C>& out) {
                    out = call_cmplx_F(precon_fn, n, in);
                };
                res = csnewton::lgmres<C>(mv, pc, xv, bv, inner, k_aug, max_iter, tol);
            }

            py::array_t<C> x_out({(py::ssize_t)n});
            std::copy(xv.begin(), xv.end(), x_out.mutable_data());
            return py::make_tuple(x_out, res.converged, res.iterations, res.residual);
        },
        py::arg("matvec"),
        py::arg("x0"),
        py::arg("b"),
        py::arg("inner")    = 20,
        py::arg("k_aug")    = 2,
        py::arg("max_iter") = 200,
        py::arg("tol")      = 1e-8,
        py::arg("precon")   = py::none(),
        R"doc(
Solve A*x = b using LGMRES with augmented Krylov subspace (complex/complex128).

Parameters
----------
matvec  : callable(v: ndarray complex128) -> ndarray
x0      : ndarray complex128
b       : ndarray complex128
inner   : int, default 20
k_aug   : int, default 2
max_iter: int, default 200
tol     : float, default 1e-8
precon  : callable or None

Returns
-------
(x, converged, iterations, residual)
)doc");

    m.def("csnewton",
        [](py::object  F_fn,
           py::array_t<double, py::array::c_style | py::array::forcecast> x0,
           int    max_newton,
           double tol,
           double h_cs,
           int    gmres_restart,
           int    gmres_max_iter,
           double gmres_tol,
           py::object precon_fn,
           bool   use_lgmres,
           int    lgmres_augment)
        {
            int n = static_cast<int>(x0.size());
            if (n == 0)
                throw std::invalid_argument("newton_gmres: x0 must be non-empty");

            std::vector<double> xv(x0.data(), x0.data() + n);

            BridgeF F{F_fn, n};

            csnewton::NewtonResult res;
            if (precon_fn.is_none()) {
                res = csnewton::csnewton(F, xv,
                    max_newton, tol, h_cs,
                    gmres_restart, gmres_max_iter, gmres_tol,
                    use_lgmres, lgmres_augment);
            } else {
                using arr_t = py::array_t<double, py::array::c_style | py::array::forcecast>;
                auto pc = [&](const std::vector<double>& in, std::vector<double>& out) {
                    auto in_arr = py::array_t<double>({(py::ssize_t)n}, in.data());
                    auto pres   = precon_fn(in_arr).cast<arr_t>();
                    std::copy(pres.data(), pres.data() + n, out.begin());
                };
                res = csnewton::csnewton(F, pc, xv,
                    max_newton, tol, h_cs,
                    gmres_restart, gmres_max_iter, gmres_tol,
                    use_lgmres, lgmres_augment);
            }

            py::array_t<double> x_out({(py::ssize_t)n});
            std::copy(xv.begin(), xv.end(), x_out.mutable_data());
            return py::make_tuple(x_out,
                                  res.converged,
                                  res.newton_iters,
                                  res.gmres_iters,
                                  res.residual);
        },
        py::arg("F"),
        py::arg("x0"),
        py::arg("max_newton")     = 50,
        py::arg("tol")            = 1e-10,
        py::arg("h_cs")           = 1e-20,
        py::arg("gmres_restart")  = 20,
        py::arg("gmres_max_iter") = 200,
        py::arg("gmres_tol")      = 1e-6,
        py::arg("precon")         = py::none(),
        py::arg("use_lgmres")     = false,
        py::arg("lgmres_augment") = 2,
        R"doc(
Solve F(x) = 0 using Newton-GMRES (or Newton-LGMRES) with complex-step Jacobian-vector products.

The Jacobian-vector product J(x)*u is never formed explicitly; instead it is
approximated by the complex step:

    J(x)*u  ~=  Im[ F(x + 1j*h_cs*u) ] / h_cs

F must work for both float64 and complex128 NumPy arrays.  Any function written
with standard NumPy operations (arithmetic, np.sin, np.exp, etc.) satisfies
this automatically — no separate complex version is needed.

For scalar problems pass x0 as a Python float; the return value x will also
be a float.  For vector problems pass x0 as a 1-D ndarray.

Parameters
----------
F        : callable(x: ndarray or float) -> ndarray or float
               evaluates F for both float64 and complex128 inputs
x0       : ndarray float64 or float    initial guess
max_newton    : int,   default 50     maximum Newton iterations
tol           : float, default 1e-10  ||F(x)||_2 stopping tolerance
h_cs          : float, default 1e-20  complex step size
gmres_restart : int,   default 20     GMRES/LGMRES inner restart parameter
gmres_max_iter: int,   default 200    max iterations per Newton step
gmres_tol     : float, default 1e-6   relative residual tolerance
precon        : callable or None      optional left preconditioner M^{-1}
use_lgmres    : bool,  default False  use LGMRES instead of GMRES
lgmres_augment: int,   default 2      number of augmentation vectors (k_aug)

Returns
-------
(x, converged, newton_iters, gmres_iters, residual)
  x            float or ndarray float64  — solution (scalar if x0 was scalar)
  converged    bool             — True if ||F(x)|| < tol was achieved
  newton_iters int              — Newton steps taken
  gmres_iters  int              — total GMRES iterations (all Newton steps)
  residual     float            — ||F(x_final)||_2
)doc");

    // ------------------------------------------------------------------
    // Scalar overload: x0 is a Python float, f(x) -> scalar
    // ------------------------------------------------------------------
    m.def("csnewton",
        [](py::object  f_fn,
           double      x0,
           int    max_newton,
           double tol,
           double h_cs,
           int    gmres_restart,
           int    gmres_max_iter,
           double gmres_tol)
        {
            // Bridge: call Python f with a plain Python float or complex.
            // We pass a 0-d numpy array so the user's lambda/function sees
            // a numeric type naturally (numpy scalars support arithmetic).
            auto call_real  = [&](double xv) -> double {
                auto xa = py::array_t<double>(std::vector<py::ssize_t>{}, &xv);
                return f_fn(xa).cast<double>();
            };
            auto call_cmplx = [&](C xv) -> C {
                auto xa = py::array_t<C>(std::vector<py::ssize_t>{}, &xv);
                return f_fn(xa).cast<C>();
            };

            // Scalar-to-scalar generic wrapper satisfying the C++ template.
            auto F = [&](auto xv) -> decltype(xv) {
                using S = decltype(xv);
                if constexpr (std::is_same_v<S, double>)
                    return call_real(xv);
                else
                    return call_cmplx(xv);
            };

            double x = x0;
            auto res = csnewton::csnewton(F, x,
                max_newton, tol, h_cs,
                gmres_restart, gmres_max_iter, gmres_tol);

            return py::make_tuple(x,
                                  res.converged,
                                  res.newton_iters,
                                  res.gmres_iters,
                                  res.residual);
        },
        py::arg("f"),
        py::arg("x0"),
        py::arg("max_newton")     = 50,
        py::arg("tol")            = 1e-10,
        py::arg("h_cs")           = 1e-20,
        py::arg("gmres_restart")  = 20,
        py::arg("gmres_max_iter") = 200,
        py::arg("gmres_tol")      = 1e-6,
        R"doc(
Scalar overload: solve f(x) = 0 for a scalar unknown x.

f must accept both a float64 and a complex128 numpy scalar (0-d array).
Any function written with standard Python/NumPy arithmetic satisfies this.

Returns (x, converged, newton_iters, gmres_iters, residual) where x is a float.
)doc");
}
