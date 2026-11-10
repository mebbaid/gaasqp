#include "geoqp/box_geometry.hpp"
#include "geoqp/internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace geoqp
{
    namespace geometry
    {
        namespace
        {
            using Vec = std::vector<double>;

            Vec matvec(const Vec &M, const Vec &x, int n)
            {
                Vec y(n, 0.0);
                for (int i = 0; i < n; ++i)
                {
                    double s = 0.0;
                    for (int j = 0; j < n; ++j)
                        s += M[i * n + j] * x[j];
                    y[i] = s;
                }
                return y;
            }

            Vec matmul(const Vec &A, const Vec &B, int n)
            {
                Vec C(n * n, 0.0);
                for (int i = 0; i < n; ++i)
                    for (int k = 0; k < n; ++k)
                    {
                        const double a = A[i * n + k];
                        if (a == 0.0)
                            continue;
                        for (int j = 0; j < n; ++j)
                            C[i * n + j] += a * B[k * n + j];
                    }
                return C;
            }

            Vec transpose(const Vec &A, int n)
            {
                Vec At(n * n);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                        At[j * n + i] = A[i * n + j];
                return At;
            }

            double h_inner(const Vec &H, const Vec &a, const Vec &b, int n)
            {
                double s = 0.0;
                for (int i = 0; i < n; ++i)
                {
                    double row = 0.0;
                    for (int j = 0; j < n; ++j)
                        row += H[i * n + j] * b[j];
                    s += a[i] * row;
                }
                return s;
            }

            Vec inverse(const Vec &T, int n)
            {
                Vec Tinv(n * n), e(n, 0.0), col(n);
                for (int j = 0; j < n; ++j)
                {
                    e[j] = 1.0;
                    internal::LinearAlgebra::solve_linear_system(T.data(), e.data(), col.data(), n);
                    e[j] = 0.0;
                    for (int i = 0; i < n; ++i)
                        Tinv[i * n + j] = col[i];
                }
                return Tinv;
            }

            void check_size(const Vec &v, size_t expected, const char *what)
            {
                if (v.size() != expected)
                    throw std::invalid_argument(std::string("box_geometry: bad size for ") + what);
            }
        } // namespace

        // ── AffineBox ───────────────────────────────────────────────────
        AffineBox AffineBox::axis_aligned(const std::vector<double> &lower,
                                          const std::vector<double> &upper)
        {
            AffineBox b;
            b.n = static_cast<int>(lower.size());
            b.T.assign(b.n * b.n, 0.0);
            for (int i = 0; i < b.n; ++i)
                b.T[i * b.n + i] = 1.0;
            b.c.assign(b.n, 0.0);
            b.lower = lower;
            b.upper = upper;
            return b;
        }

        void AffineBox::halfspace(std::vector<double> &A, std::vector<double> &b) const
        {
            const Vec Tinv = inverse(T, n);
            const Vec d = matvec(Tinv, c, n);
            A.assign(2 * n * n, 0.0);
            b.assign(2 * n, 0.0);
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    A[i * n + j] = Tinv[i * n + j];
                    A[(n + i) * n + j] = -Tinv[i * n + j];
                }
                b[i] = upper[i] + d[i];
                b[n + i] = -lower[i] - d[i];
            }
        }

        // ── BoxFrame ────────────────────────────────────────────────────
        bool BoxFrame::is_violated(int i) const
        {
            return ybar[i] > upper[i] || ybar[i] < lower[i];
        }

        bool BoxFrame::at_or_beyond_bound(int i) const
        {
            return ybar[i] >= upper[i] || ybar[i] <= lower[i];
        }

        BoxFrame make_frame(const AffineBox &box, const std::vector<double> &xbar)
        {
            const int n = box.n;
            check_size(box.T, static_cast<size_t>(n) * n, "T");
            check_size(box.c, n, "c");
            check_size(box.lower, n, "lower");
            check_size(box.upper, n, "upper");
            check_size(xbar, n, "xbar");
            for (int i = 0; i < n; ++i)
                if (!(box.lower[i] < box.upper[i]))
                    throw std::invalid_argument("box_geometry: need lower < upper");

            BoxFrame f;
            f.n = n;
            f.T = box.T;
            f.Tinv = inverse(box.T, n);
            f.c = box.c;
            f.lower = box.lower;
            f.upper = box.upper;
            f.mu.resize(n);
            f.h.resize(n);
            for (int i = 0; i < n; ++i)
            {
                f.mu[i] = 0.5 * (box.lower[i] + box.upper[i]);
                f.h[i] = 0.5 * (box.upper[i] - box.lower[i]);
            }
            f.c0 = matvec(f.T, f.mu, n);
            for (int i = 0; i < n; ++i)
                f.c0[i] += f.c[i];

            f.xbar = xbar;
            f.ybar = to_y(f, xbar);
            f.s.resize(n);
            f.xibar.resize(n);
            for (int i = 0; i < n; ++i)
            {
                f.s[i] = (f.ybar[i] >= f.mu[i]) ? 1.0 : -1.0;
                f.xibar[i] = f.s[i] * (f.ybar[i] - f.mu[i]) / f.h[i];
                if (f.is_violated(i))
                    f.violated.push_back(i);
            }
            return f;
        }

        // ── coordinate maps ─────────────────────────────────────────────
        std::vector<double> to_y(const BoxFrame &f, const std::vector<double> &x)
        {
            Vec d(f.n);
            for (int i = 0; i < f.n; ++i)
                d[i] = x[i] - f.c[i];
            return matvec(f.Tinv, d, f.n);
        }

        std::vector<double> to_xi(const BoxFrame &f, const std::vector<double> &x)
        {
            Vec y = to_y(f, x);
            for (int i = 0; i < f.n; ++i)
                y[i] = f.s[i] * (y[i] - f.mu[i]) / f.h[i];
            return y;
        }

        std::vector<double> from_xi(const BoxFrame &f, const std::vector<double> &xi)
        {
            Vec y(f.n);
            for (int i = 0; i < f.n; ++i)
                y[i] = f.mu[i] + f.s[i] * f.h[i] * xi[i];
            Vec x = matvec(f.T, y, f.n);
            for (int i = 0; i < f.n; ++i)
                x[i] += f.c[i];
            return x;
        }

        std::vector<double> D_matrix(const BoxFrame &f)
        {
            Vec D = f.Tinv;
            for (int i = 0; i < f.n; ++i)
                for (int j = 0; j < f.n; ++j)
                    D[i * f.n + j] *= f.s[i] / f.h[i];
            return D;
        }

        std::vector<double> D_inverse(const BoxFrame &f)
        {
            Vec Di = f.T;
            for (int i = 0; i < f.n; ++i)
                for (int j = 0; j < f.n; ++j)
                    Di[i * f.n + j] *= f.s[j] * f.h[j];
            return Di;
        }

        std::vector<double> transformed_hessian(const BoxFrame &f, const std::vector<double> &H)
        {
            const Vec Di = D_inverse(f);
            return matmul(transpose(Di, f.n), matmul(H, Di, f.n), f.n);
        }

        std::vector<double> original_hessian(const BoxFrame &f, const std::vector<double> &Ht)
        {
            const Vec D = D_matrix(f);
            return matmul(transpose(D, f.n), matmul(Ht, D, f.n), f.n);
        }

        // ── A. good vertices ────────────────────────────────────────────
        bool is_good_vertex_sign(const BoxFrame &f, const std::vector<int> &sigma)
        {
            for (int i = 0; i < f.n; ++i)
                if (f.at_or_beyond_bound(i) && sigma[i] != -1)
                    return false;
            return true;
        }

        std::vector<double> vertex_from_sign(const BoxFrame &f, const std::vector<int> &sigma)
        {
            Vec xi(f.n);
            for (int i = 0; i < f.n; ++i)
                xi[i] = static_cast<double>(sigma[i]);
            return from_xi(f, xi);
        }

        std::vector<double> antipodal_vertex(const BoxFrame &f)
        {
            return vertex_from_sign(f, std::vector<int>(f.n, -1));
        }

        std::vector<double> lemma1_vertex(const BoxFrame &f)
        {
            Vec y(f.n);
            for (int i = 0; i < f.n; ++i)
                y[i] = (f.ybar[i] > f.lower[i]) ? f.lower[i] : f.upper[i];
            Vec x = matvec(f.T, y, f.n);
            for (int i = 0; i < f.n; ++i)
                x[i] += f.c[i];
            return x;
        }

        // ── E. exit facet ───────────────────────────────────────────────
        std::vector<Wall> exit_walls_by_ray(const BoxFrame &f, const std::vector<double> &v,
                                            double rel_tie_tol)
        {
            const Vec yv = to_y(f, v);
            std::vector<Wall> walls;
            walls.reserve(f.n);
            double tmin = std::numeric_limits<double>::infinity();
            for (int i = 0; i < f.n; ++i)
            {
                const double d = f.ybar[i] - yv[i];
                if (d == 0.0)
                    continue;
                Wall w;
                w.coord = i;
                w.upper = d > 0.0;
                const double bound = w.upper ? f.upper[i] : f.lower[i];
                w.t = std::max(0.0, (bound - yv[i]) / d);
                tmin = std::min(tmin, w.t);
                walls.push_back(w);
            }
            std::vector<Wall> out;
            const double tol = rel_tie_tol * std::max(1.0, tmin);
            for (const Wall &w : walls)
                if (w.t <= tmin + tol)
                    out.push_back(w);
            return out;
        }

        std::vector<double> antipodal_exit_times(const BoxFrame &f)
        {
            Vec t(f.n);
            for (int i = 0; i < f.n; ++i)
                t[i] = 2.0 / (f.xibar[i] + 1.0);
            return t;
        }

        Wall xi_plus_wall(const BoxFrame &f, int i)
        {
            Wall w;
            w.coord = i;
            w.upper = f.s[i] > 0.0; // ξᵢ = 1 ⇔ yᵢ = μᵢ + sᵢhᵢ
            w.t = 2.0 / (f.xibar[i] + 1.0);
            return w;
        }

        Wall predicted_exit_wall(const BoxFrame &f)
        {
            int best = 0;
            for (int i = 1; i < f.n; ++i)
                if (f.xibar[i] > f.xibar[best])
                    best = i;
            return xi_plus_wall(f, best);
        }

        // ── B/C. visible facets and corrected cut ───────────────────────
        std::vector<int> visible_facets_containing(const BoxFrame &f, const std::vector<double> &x,
                                                   double tol)
        {
            const Vec xi = to_xi(f, x);
            std::vector<int> out;
            for (int i : f.violated)
                if (std::abs(xi[i] - 1.0) <= tol)
                    out.push_back(i);
            return out;
        }

        double old_halfspace_value(const BoxFrame &f, const std::vector<double> &x)
        {
            const Vec xi = to_xi(f, x);
            double s = 0.0;
            for (double v : xi)
                s += v;
            return s;
        }

        std::vector<double> corrected_eta(const BoxFrame &f)
        {
            Vec eta(f.n, 0.0);
            for (int i : f.violated)
                eta[i] = f.s[i] / f.h[i];
            return eta;
        }

        double corrected_cut_value(const BoxFrame &f, const std::vector<double> &x)
        {
            // ηᵀ T⁻¹(x − c₀) = ηᵀ(y − μ)
            const Vec eta = corrected_eta(f);
            const Vec y = to_y(f, x);
            double s = 0.0;
            for (int i = 0; i < f.n; ++i)
                s += eta[i] * (y[i] - f.mu[i]);
            return s;
        }

        double corrected_cut_threshold(const BoxFrame &f)
        {
            return 2.0 - static_cast<double>(f.k());
        }

        double corrected_cut_margin(const BoxFrame &f, const std::vector<double> &x)
        {
            return corrected_cut_value(f, x) - corrected_cut_threshold(f);
        }

        // ── F. H-metric central half ────────────────────────────────────
        double h_halfspace_margin(const BoxFrame &f, const std::vector<double> &H,
                                  const std::vector<double> &x)
        {
            Vec a(f.n), b(f.n);
            for (int i = 0; i < f.n; ++i)
            {
                a[i] = f.xbar[i] - f.c0[i];
                b[i] = x[i] - f.c0[i];
            }
            return h_inner(H, a, b, f.n) - h_inner(H, b, b, f.n);
        }

        double h_ball_margin(const BoxFrame &f, const std::vector<double> &H,
                             const std::vector<double> &x)
        {
            Vec a(f.n), b(f.n);
            for (int i = 0; i < f.n; ++i)
            {
                a[i] = f.xbar[i] - f.c0[i];
                b[i] = x[i] - 0.5 * (f.c0[i] + f.xbar[i]);
            }
            return 0.5 * std::sqrt(h_inner(H, a, a, f.n)) - std::sqrt(h_inner(H, b, b, f.n));
        }

        double antipodal_side_value(const BoxFrame &f, const std::vector<double> &H)
        {
            const Vec v = antipodal_vertex(f);
            Vec a(f.n), b(f.n);
            for (int i = 0; i < f.n; ++i)
            {
                a[i] = f.xbar[i] - f.c0[i];
                b[i] = v[i] - f.c0[i];
            }
            return h_inner(H, a, b, f.n);
        }

        double antipodal_side_criterion(const BoxFrame &f, const std::vector<double> &H)
        {
            const Vec Ht = transformed_hessian(f, H);
            const Vec w = matvec(Ht, f.xibar, f.n);
            double s = 0.0;
            for (double v : w)
                s += v;
            return s;
        }

        // ── D. tightness ────────────────────────────────────────────────
        std::vector<double> tightness_hessian_xi(const BoxFrame &f, const std::vector<double> &p,
                                                 int i, double eps)
        {
            const int n = f.n;
            if (!f.is_violated(i))
                throw std::invalid_argument("tightness_hessian: i must be in V");
            if (!(eps > 0.0))
                throw std::invalid_argument("tightness_hessian: eps must be > 0");
            Vec a(n);
            double a2 = 0.0;
            for (int j = 0; j < n; ++j)
            {
                a[j] = f.xibar[j] - p[j];
                a2 += a[j] * a[j];
            }
            Vec Ht(n * n, 0.0);
            for (int r = 0; r < n; ++r)
                for (int q = 0; q < n; ++q)
                    Ht[r * n + q] = eps * ((r == q ? 1.0 : 0.0) - a[r] * a[q] / a2);
            Ht[i * n + i] += 1.0 / a[i];
            return Ht;
        }

        std::vector<double> tightness_hessian(const BoxFrame &f, const std::vector<double> &p,
                                              int i, double eps)
        {
            return original_hessian(f, tightness_hessian_xi(f, p, i, eps));
        }

    } // namespace geometry
} // namespace geoqp
