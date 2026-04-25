// csnewton.hpp — GMRES(m) + Complex-step Newton solver  (namespace csnewton)
//
// Provides two algorithms, both in namespace csnewton:
//
//   csnewton::gmres(matvec, [precon,] x, b, restart, max_iter, tol)
//     Matrix-free GMRES(m) with restart. Supports real (double) and
//     complex (complex<double>) scalar types.
//
//   csnewton::csnewton(F, [precon,] x, [params...])
//     Newton-Krylov solver: F(x)=0 via Newton iteration, each linear system
//     solved by GMRES using the complex-step Jacobian-vector product:
//       J(x)*u  ~=  Im[ F(x + i*h*u) ] / h        (h ~ 1e-20)
//     F must be callable for both vector<double> and vector<complex<double>>.
//     A C++14 generic lambda (auto parameter) satisfies this naturally.
//
// Reference: D. Mitsotakis,
//   "The complex-step Newton method and its convergence",
//   Numerische Mathematik 157(3), 993-1021, 2025.
//   https://link.springer.com/article/10.1007/s00211-025-01471-w

#pragma once

#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <stdexcept>
#ifdef _OPENMP
#  include <omp.h>
#endif

namespace csnewton {

// ============================================================================
// GMRES(m) with restart
// ============================================================================

// ---------------------------------------------------------------------------
// Scalar traits
// ---------------------------------------------------------------------------

template<typename T> struct real_type                { using type = T; };
template<typename T> struct real_type<std::complex<T>>{ using type = T; };

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

template<typename Scalar>
struct GMRESResult {
    bool converged;
    int  iterations;
    typename real_type<Scalar>::type residual;    // final ||r||/||b||
};

// ---------------------------------------------------------------------------
// Internal helpers (anonymous namespace, visible only within this header)
// ---------------------------------------------------------------------------

namespace {

// Conjugate that is identity for real types and conj() for complex.
template<typename T> T               my_conj(const T& x)               { return x; }
template<typename T> std::complex<T> my_conj(const std::complex<T>& x) { return std::conj(x); }

template<typename T> using Real = typename real_type<T>::type;

// OpenMP does not know how to reduce std::complex out of the box.
#ifdef _OPENMP
#pragma omp declare reduction(+ : std::complex<float>  \
    : omp_out += omp_in) initializer(omp_priv = {})
#pragma omp declare reduction(+ : std::complex<double> \
    : omp_out += omp_in) initializer(omp_priv = {})
#endif

template<typename Scalar>
Scalar vec_dot(const std::vector<Scalar>& u, const std::vector<Scalar>& v)
{
    Scalar s{};
    const int n = static_cast<int>(u.size());
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int i = 0; i < n; ++i) s += my_conj(u[i]) * v[i];
    return s;
}

template<typename Scalar>
Real<Scalar> vec_norm(const std::vector<Scalar>& v)
{
    Real<Scalar> s{};
    const int n = static_cast<int>(v.size());
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int i = 0; i < n; ++i) s += std::real(my_conj(v[i]) * v[i]);
    return std::sqrt(s);
}

template<typename Scalar>
void vec_axpy(const Scalar& alpha, const std::vector<Scalar>& x, std::vector<Scalar>& y)
{
    const int n = static_cast<int>(x.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

template<typename Scalar>
void vec_scale(const Scalar& alpha, std::vector<Scalar>& v)
{
    const int n = static_cast<int>(v.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) v[i] *= alpha;
}

template<typename Scalar>
void generate_givens(const Scalar& dx, const Scalar& dy, Scalar& cs, Scalar& sn)
{
    using R = Real<Scalar>;
    R abs_dx = std::abs(dx), abs_dy = std::abs(dy);
    if (abs_dy == R(0)) {
        cs = Scalar(1); sn = Scalar(0);
    } else if (abs_dx == R(0)) {
        cs = Scalar(0); sn = Scalar(1);
    } else {
        R tau = std::sqrt(abs_dx * abs_dx + abs_dy * abs_dy);
        cs = Scalar(abs_dx / tau);
        sn = (my_conj(dx) / Scalar(abs_dx)) * Scalar(abs_dy / tau);
    }
}

template<typename Scalar>
void apply_givens(Scalar& dx, Scalar& dy, const Scalar& cs, const Scalar& sn)
{
    Scalar tmp = cs * dx + sn * dy;
    dy         = -my_conj(sn) * dx + cs * dy;
    dx         = tmp;
}

template<typename Scalar>
void update_solution(int k, int ldh,
                     std::vector<Scalar>& x,
                     const std::vector<Scalar>& H,
                     std::vector<Scalar>& s,
                     const std::vector<std::vector<Scalar>>& V)
{
    std::vector<Scalar> y(s.begin(), s.begin() + k + 1);
    for (int i = k; i >= 0; --i) {
        y[i] /= H[i + i * ldh];
        for (int j = 0; j < i; ++j) y[j] -= H[j + i * ldh] * y[i];
    }
    const int n = static_cast<int>(x.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= k; ++j)
            x[i] += y[j] * V[j][i];
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// gmres — full version with preconditioner
// ---------------------------------------------------------------------------

template<typename Scalar = double, typename Operator, typename Precond>
GMRESResult<Scalar>
gmres(Operator&& matvec, Precond&& precon,
      std::vector<Scalar>& x, const std::vector<Scalar>& b,
      int restart, int max_iter,
      typename real_type<Scalar>::type tol)
{
    using R = typename real_type<Scalar>::type;
    const int n   = static_cast<int>(b.size());
    const int ldh = restart + 1;

    if (restart <= 0 || max_iter <= 0)
        throw std::invalid_argument("gmres: restart and max_iter must be positive");
    if (static_cast<int>(x.size()) != n)
        throw std::invalid_argument("gmres: x and b must have the same length");

    std::vector<Scalar> H(ldh * restart, Scalar(0));
    std::vector<std::vector<Scalar>> V(restart + 1, std::vector<Scalar>(n));
    std::vector<Scalar> cs(restart), sn(restart), s(restart + 1);
    std::vector<Scalar> w(n), r(n), z(n);

    matvec(x, w);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) r[i] = b[i] - w[i];
    precon(r, z);

    R normb = vec_norm(b);
    if (normb == R(0)) normb = R(1);

    R beta = vec_norm(z);
    if (beta / normb < tol) return {true, 0, beta / normb};

    int total = 0;
    while (total < max_iter) {
        std::fill(H.begin(), H.end(), Scalar(0));
        std::fill(s.begin(), s.end(), Scalar(0));
        s[0] = Scalar(beta);
        V[0] = z;
        vec_scale(Scalar(R(1) / beta), V[0]);

        int i = 0;
        for (; i < restart && total < max_iter; ++i, ++total) {
            matvec(V[i], r);
            precon(r, w);

            for (int k = 0; k <= i; ++k) {
                H[k + i * ldh] = vec_dot(w, V[k]);
                vec_axpy(-H[k + i * ldh], V[k], w);
            }
            R h_next = vec_norm(w);
            H[(i + 1) + i * ldh] = Scalar(h_next);

            if (h_next == R(0)) {
                update_solution(i, ldh, x, H, s, V);
                return {true, total + 1, R(0)};
            }

            V[i + 1] = w;
            vec_scale(Scalar(R(1) / h_next), V[i + 1]);

            for (int k = 0; k < i; ++k)
                apply_givens(H[k + i * ldh], H[(k + 1) + i * ldh], cs[k], sn[k]);

            generate_givens(H[i + i * ldh], H[(i + 1) + i * ldh], cs[i], sn[i]);
            apply_givens(H[i + i * ldh], H[(i + 1) + i * ldh], cs[i], sn[i]);
            apply_givens(s[i], s[i + 1], cs[i], sn[i]);

            R resid_est = std::abs(s[i + 1]) / normb;
            if (resid_est < tol) {
                update_solution(i, ldh, x, H, s, V);
                return {true, total + 1, resid_est};
            }
        }

        update_solution(i - 1, ldh, x, H, s, V);

        matvec(x, w);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; ++k) r[k] = b[k] - w[k];
        precon(r, z);
        beta = vec_norm(z);

        R resid = beta / normb;
        if (resid < tol) return {true, total, resid};
    }
    return {false, total, beta / normb};
}

// ---------------------------------------------------------------------------
// gmres — convenience overload: identity preconditioner
// ---------------------------------------------------------------------------

template<typename Scalar = double, typename Operator>
GMRESResult<Scalar>
gmres(Operator&& matvec,
      std::vector<Scalar>& x, const std::vector<Scalar>& b,
      int restart, int max_iter,
      typename real_type<Scalar>::type tol)
{
    auto identity = [](const std::vector<Scalar>& in, std::vector<Scalar>& out) {
        out = in;
    };
    return gmres<Scalar>(std::forward<Operator>(matvec), identity,
                         x, b, restart, max_iter, tol);
}

// ============================================================================
// LGMRES(m, k) — Loose GMRES with augmented Krylov subspace
//
// Differs from GMRES(m) in that at each restart cycle the subspace is
// augmented with k correction vectors retained from previous cycles.
// This prevents information loss at restart and accelerates convergence
// for problems where GMRES(m) stagnates.
//
// Parameters:
//   inner    — number of new Arnoldi steps per cycle  (like 'restart' in GMRES)
//   k_aug    — number of augmentation vectors (default 2; 0 reduces to GMRES)
//   max_iter — maximum total steps (Arnoldi + augmentation combined)
//
// Reference: Baker, Jessup & Manteuffel,
//   "A Technique for Accelerating the Convergence of Restarted GMRES",
//   SIAM J. Matrix Anal. Appl. 26(4), 962-984, 2005.
// ============================================================================

template<typename Scalar = double, typename Operator, typename Precond>
GMRESResult<Scalar>
lgmres(Operator&& matvec, Precond&& precon,
       std::vector<Scalar>& x, const std::vector<Scalar>& b,
       int inner    = 20,
       int k_aug    = 2,
       int max_iter = 200,
       typename real_type<Scalar>::type tol = 1e-8)
{
    using R = typename real_type<Scalar>::type;
    const int n = static_cast<int>(b.size());

    if (inner <= 0 || max_iter <= 0)
        throw std::invalid_argument("lgmres: inner and max_iter must be positive");
    if (k_aug < 0)
        throw std::invalid_argument("lgmres: k_aug must be non-negative");
    if (static_cast<int>(x.size()) != n)
        throw std::invalid_argument("lgmres: x and b must have the same length");

    // Augmentation vectors from previous cycles.
    // Z_aug[j]    : correction in x-space  (Δx from cycle j)
    // MApZ_aug[j] : M^{-1} A * Z_aug[j]   (precomputed for reuse)
    std::vector<std::vector<Scalar>> Z_aug, MApZ_aug;

    R normb = vec_norm(b);
    if (normb == R(0)) normb = R(1);

    std::vector<Scalar> Ax(n), r(n), q(n), w(n);
    int total_iters = 0;

    while (total_iters < max_iter) {
        const int k = static_cast<int>(Z_aug.size()); // grows to k_aug over cycles
        const int m = inner + k;                       // total basis size this cycle
        const int ldh = m + 1;

        // ---- Residual and initial basis vector --------------------------------
        matvec(x, Ax);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
        precon(r, q);

        R beta = vec_norm(q);
        if (beta / normb < tol)
            return {true, total_iters, beta / normb};

        // ---- Allocate cycle workspace -----------------------------------------
        std::vector<Scalar> H(ldh * m, Scalar(0));
        std::vector<std::vector<Scalar>> V(m + 1, std::vector<Scalar>(n));
        std::vector<Scalar> cs(m), sn(m), s(m + 1, Scalar(0));

        V[0] = q;
        vec_scale(Scalar(R(1) / beta), V[0]);
        s[0] = Scalar(beta);

        int  final_j   = m - 1;
        bool converged = false;

        // ---- Phase 1: standard Arnoldi (inner steps) -------------------------
        const int inner_lim = std::min(inner, max_iter - total_iters);
        for (int j = 0; j < inner_lim; ++j, ++total_iters) {
            matvec(V[j], r);
            precon(r, w);

            for (int i = 0; i <= j; ++i) {
                H[i + j * ldh] = vec_dot(w, V[i]);
                vec_axpy(-H[i + j * ldh], V[i], w);
            }
            R h_next = vec_norm(w);
            H[(j + 1) + j * ldh] = Scalar(h_next);

            if (h_next > R(0)) {
                V[j + 1] = w;
                vec_scale(Scalar(R(1) / h_next), V[j + 1]);
            }

            for (int i = 0; i < j; ++i)
                apply_givens(H[i + j*ldh], H[(i+1) + j*ldh], cs[i], sn[i]);
            generate_givens(H[j + j*ldh], H[(j+1) + j*ldh], cs[j], sn[j]);
            apply_givens(H[j + j*ldh], H[(j+1) + j*ldh], cs[j], sn[j]);
            apply_givens(s[j], s[j + 1], cs[j], sn[j]);

            if (std::abs(s[j + 1]) / normb < tol || h_next == R(0)) {
                final_j = j; converged = true; break;
            }
        }

        // ---- Phase 2: augmentation (k steps using stored corrections) --------
        if (!converged) {
            const int aug_lim = std::min(k, max_iter - total_iters);
            for (int aug = 0; aug < aug_lim; ++aug, ++total_iters) {
                const int j = inner + aug;

                // w = M^{-1} A z_aug[aug]  (precomputed)
                w = MApZ_aug[aug];

                // Orthogonalize against all existing basis vectors V[0..j]
                for (int i = 0; i <= j; ++i) {
                    H[i + j * ldh] = vec_dot(w, V[i]);
                    vec_axpy(-H[i + j * ldh], V[i], w);
                }
                R h_next = vec_norm(w);
                H[(j + 1) + j * ldh] = Scalar(h_next);

                if (h_next > R(0)) {
                    V[j + 1] = w;
                    vec_scale(Scalar(R(1) / h_next), V[j + 1]);
                }

                for (int i = 0; i < j; ++i)
                    apply_givens(H[i + j*ldh], H[(i+1) + j*ldh], cs[i], sn[i]);
                generate_givens(H[j + j*ldh], H[(j+1) + j*ldh], cs[j], sn[j]);
                apply_givens(H[j + j*ldh], H[(j+1) + j*ldh], cs[j], sn[j]);
                apply_givens(s[j], s[j + 1], cs[j], sn[j]);

                if (std::abs(s[j + 1]) / normb < tol || h_next == R(0)) {
                    final_j = j; converged = true; break;
                }
            }
        }

        // ---- Solve upper triangular system for y -----------------------------
        std::vector<Scalar> y(s.begin(), s.begin() + final_j + 1);
        for (int i = final_j; i >= 0; --i) {
            y[i] /= H[i + i * ldh];
            for (int jj = 0; jj < i; ++jj)
                y[jj] -= H[jj + i * ldh] * y[i];
        }

        // ---- Update x and build correction vector for next cycle's augmentation
        // Krylov columns (j < inner): contribution from V[j]
        // Augmentation columns (j >= inner): contribution from Z_aug[j-inner]
        std::vector<Scalar> z_new(n, Scalar(0));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= final_j; ++j) {
                Scalar c = y[j] * (j < inner ? V[j][i] : Z_aug[j - inner][i]);
                x[i]     += c;
                z_new[i] += c;
            }
        }

        // ---- Store z_new and precompute M^{-1} A z_new for next cycle --------
        if (k_aug > 0) {
            std::vector<Scalar> Az_new(n), MAz_new(n);
            matvec(z_new, Az_new);
            precon(Az_new, MAz_new);
            Z_aug.insert(Z_aug.begin(), z_new);
            MApZ_aug.insert(MApZ_aug.begin(), MAz_new);
            if (static_cast<int>(Z_aug.size()) > k_aug) {
                Z_aug.pop_back();
                MApZ_aug.pop_back();
            }
        }

        if (converged) {
            matvec(x, Ax);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
            precon(r, q);
            return {true, total_iters, vec_norm(q) / normb};
        }
    }

    // Final residual
    matvec(x, Ax);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
    precon(r, q);
    return {false, total_iters, vec_norm(q) / normb};
}

// ---------------------------------------------------------------------------
// lgmres — convenience overload: identity preconditioner
// ---------------------------------------------------------------------------

template<typename Scalar = double, typename Operator>
GMRESResult<Scalar>
lgmres(Operator&& matvec,
       std::vector<Scalar>& x, const std::vector<Scalar>& b,
       int inner    = 20,
       int k_aug    = 2,
       int max_iter = 200,
       typename real_type<Scalar>::type tol = 1e-8)
{
    auto identity = [](const std::vector<Scalar>& in, std::vector<Scalar>& out) {
        out = in;
    };
    return lgmres<Scalar>(std::forward<Operator>(matvec), identity,
                          x, b, inner, k_aug, max_iter, tol);
}

// ============================================================================
// Complex-step Newton solver
// ============================================================================

struct NewtonResult {
    bool   converged;
    int    newton_iters;    // Newton steps taken
    int    gmres_iters;     // total GMRES iterations across all Newton steps
    double residual;        // ||F(x_final)||_2
};

// ---------------------------------------------------------------------------
// csnewton — full version: single callable F + left preconditioner
// ---------------------------------------------------------------------------

template<typename Op, typename Precond>
NewtonResult
csnewton(
    Op&&      F,
    Precond&& precon,
    std::vector<double>& x,
    int    max_newton     = 50,
    double tol            = 1e-10,
    double h_cs           = 1e-20,
    int    gmres_restart  = 20,
    int    gmres_max_iter = 200,
    double gmres_tol      = 1e-6,
    bool   use_lgmres     = false,  // use LGMRES instead of GMRES
    int    lgmres_augment = 2       // k_aug for LGMRES
) {
    using C = std::complex<double>;
    const int n = static_cast<int>(x.size());

    if (n == 0)
        throw std::invalid_argument("csnewton: x must be non-empty");

    auto l2 = [](const std::vector<double>& v) {
        double s = 0.0;
        const int m = static_cast<int>(v.size());
        #pragma omp parallel for reduction(+:s) schedule(static)
        for (int i = 0; i < m; ++i) s += v[i] * v[i];
        return std::sqrt(s);
    };

    std::vector<double> Fx = F(x);
    double norm_Fx = l2(Fx);

    if (norm_Fx < tol)
        return {true, 0, 0, norm_Fx};

    int total_gmres = 0;

    for (int it = 0; it < max_newton; ++it) {
        auto matvec = [&](const std::vector<double>& u, std::vector<double>& Ju) {
            std::vector<C> z(n);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
                z[i] = C{x[i], h_cs * u[i]};
            std::vector<C> Fz = F(z);           // user F — not parallelised
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
                Ju[i] = Fz[i].imag() / h_cs;
        };

        std::vector<double> u(n, 0.0);
        int g_iters;
        if (use_lgmres) {
            auto gres = lgmres<double>(matvec, precon, u, Fx,
                                       gmres_restart, lgmres_augment,
                                       gmres_max_iter, gmres_tol);
            g_iters = gres.iterations;
        } else {
            auto gres = gmres<double>(matvec, precon, u, Fx,
                                      gmres_restart, gmres_max_iter, gmres_tol);
            g_iters = gres.iterations;
        }
        total_gmres += g_iters;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            x[i] -= u[i];

        Fx      = F(x);
        norm_Fx = l2(Fx);

        if (norm_Fx < tol)
            return {true, it + 1, total_gmres, norm_Fx};
    }

    return {false, max_newton, total_gmres, norm_Fx};
}

// ---------------------------------------------------------------------------
// csnewton — convenience overload: identity preconditioner
// ---------------------------------------------------------------------------

template<typename Op>
NewtonResult
csnewton(
    Op&&  F,
    std::vector<double>& x,
    int    max_newton     = 50,
    double tol            = 1e-10,
    double h_cs           = 1e-20,
    int    gmres_restart  = 20,
    int    gmres_max_iter = 200,
    double gmres_tol      = 1e-6,
    bool   use_lgmres     = false,
    int    lgmres_augment = 2
) {
    auto identity = [](const std::vector<double>& in, std::vector<double>& out) {
        out = in;
    };
    return csnewton(std::forward<Op>(F), identity, x,
                    max_newton, tol, h_cs,
                    gmres_restart, gmres_max_iter, gmres_tol,
                    use_lgmres, lgmres_augment);
}

// ---------------------------------------------------------------------------
// csnewton — scalar overloads: f(x) scalar -> scalar
// ---------------------------------------------------------------------------

template<typename ScalarOp, typename Precond>
NewtonResult
csnewton(
    ScalarOp&& f,
    Precond&&  precon,
    double&    x,
    int    max_newton     = 50,
    double tol            = 1e-10,
    double h_cs           = 1e-20,
    int    gmres_restart  = 20,
    int    gmres_max_iter = 200,
    double gmres_tol      = 1e-6
) {
    auto F = [&f](const auto& xv) {
        using S = typename std::decay_t<decltype(xv)>::value_type;
        return std::vector<S>{ f(xv[0]) };
    };
    std::vector<double> xv{x};
    auto res = csnewton(F, std::forward<Precond>(precon), xv,
                        max_newton, tol, h_cs,
                        gmres_restart, gmres_max_iter, gmres_tol);
    x = xv[0];
    return res;
}

template<typename ScalarOp>
NewtonResult
csnewton(
    ScalarOp&& f,
    double&    x,
    int    max_newton     = 50,
    double tol            = 1e-10,
    double h_cs           = 1e-20,
    int    gmres_restart  = 20,
    int    gmres_max_iter = 200,
    double gmres_tol      = 1e-6
) {
    auto identity = [](const std::vector<double>& in, std::vector<double>& out) {
        out = in;
    };
    return csnewton(std::forward<ScalarOp>(f), identity, x,
                    max_newton, tol, h_cs,
                    gmres_restart, gmres_max_iter, gmres_tol);
}

} // namespace csnewton
