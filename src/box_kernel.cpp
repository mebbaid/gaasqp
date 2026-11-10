#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

// Pure-box kernel.  See the class comment in internal.hpp.
//
// The arithmetic deliberately mirrors the row-based box path it replaces
// (Initializer::compute_antipodal_frame, FaceHopper::find_initial_face and the
// plain loop of FaceHopper::find_optimal_face with compute_projection_and_lambda):
// same formulas, same operation order, same tie-breaking.  Only the bookkeeping
// differs, so the pivot sequence is the same.

namespace geoqp
{
    namespace internal
    {
        void BoxKernel::allocate(int n)
        {
            n_ = n;
            const size_t N = static_cast<size_t>(n);
            if (darena_.size() < 17 * N)
                darena_.resize(17 * N);
            if (iarena_.size() < 16 * N)
                iarena_.resize(16 * N);

            double *d = darena_.data();
            lb_ = d;          d += N;
            ub_ = d;          d += N;
            pb_ = d;          d += 2 * N;
            psign_ = d;       d += 2 * N;
            q_diagonal_ = d;  d += N;
            x_ = d;           d += N;
            lam_ = d;         d += 2 * N;
            v_hat_ = d;       d += N;
            neg_grad_ = d;    d += N;
            s_frame_ = d;     d += N;
            h_frame_ = d;     d += N;
            unit_ = d;        d += N;
            // 2N spare doubles keep the layout simple if a field grows.

            int *i = iarena_.data();
            pvar_ = i;            i += 2 * N;
            prow_ = i;            i += 2 * N;
            pos_upper_ = i;       i += N;
            pos_lower_ = i;       i += N;
            W_ = i;               i += 2 * N;
            in_W_ = i;            i += 2 * N;
            violated_frame_ = i;  i += N;
            viol_ = i;            i += 2 * N;
            row_slot_ = i;        i += N;
            active_rows_ = i;     i += 2 * N;
        }

        bool BoxKernel::setup(int n, int m, const double *A, const double *b,
                              const std::vector<double> &lb, const std::vector<double> &ub)
        {
            const bool have_lb = static_cast<int>(lb.size()) == n;
            const bool have_ub = static_cast<int>(ub.size()) == n;

            // Common case: bounds only.  Rows are then [u₀..u_{n−1}, ℓ₀..ℓ_{n−1}],
            // already in row order, so the position tables are filled directly.
            if (m == 0)
            {
                if (!have_lb || !have_ub)
                    return false;
                for (int j = 0; j < n; ++j)
                    if (!(lb[j] > -1e30) || !(ub[j] < 1e30) || !(lb[j] < ub[j]))
                        return false;
                allocate(n);
                for (int j = 0; j < n; ++j)
                {
                    lb_[j] = lb[j];
                    ub_[j] = ub[j];
                    pvar_[j] = j;
                    prow_[j] = j;
                    psign_[j] = 1.0;
                    pb_[j] = ub[j];
                    pos_upper_[j] = j;
                    pvar_[n + j] = j;
                    prow_[n + j] = n + j;
                    psign_[n + j] = -1.0;
                    pb_[n + j] = -lb[j];
                    pos_lower_[j] = n + j;
                }
                return true;
            }

            // Explicit rows: every one must be ±eⱼ; merge with lb/ub, tightest wins.
            const double inf = std::numeric_limits<double>::infinity();
            std::vector<double> lo(n, -inf), hi(n, inf);
            std::vector<int> lo_row(n, -1), hi_row(n, -1);
            for (int i = 0; i < m; ++i)
            {
                const double *GEOQP_RESTRICT row = A + static_cast<size_t>(i) * n;
                int nz = 0, var = -1;
                double val = 0.0;
                for (int j = 0; j < n; ++j)
                    if (std::abs(row[j]) > 1e-10)
                    {
                        if (++nz > 1)
                            return false;
                        var = j;
                        val = row[j];
                    }
                if (nz != 1)
                    return false;
                if (std::abs(val - 1.0) < 1e-10)
                {
                    if (b[i] < hi[var])
                    {
                        hi[var] = b[i];
                        hi_row[var] = i;
                    }
                }
                else if (std::abs(val + 1.0) < 1e-10)
                {
                    if (-b[i] > lo[var])
                    {
                        lo[var] = -b[i];
                        lo_row[var] = i;
                    }
                }
                else
                    return false;
            }
            // Variable bounds in the solver's expansion order: finite ub, then finite lb.
            int row = m;
            if (!ub.empty())
                for (int j = 0; j < n; ++j)
                {
                    if (ub[j] >= 1e30)
                        continue;
                    if (ub[j] < hi[j])
                    {
                        hi[j] = ub[j];
                        hi_row[j] = row;
                    }
                    ++row;
                }
            if (!lb.empty())
                for (int j = 0; j < n; ++j)
                {
                    if (lb[j] <= -1e30)
                        continue;
                    if (lb[j] > lo[j])
                    {
                        lo[j] = lb[j];
                        lo_row[j] = row;
                    }
                    ++row;
                }
            for (int j = 0; j < n; ++j)
                if (hi_row[j] < 0 || lo_row[j] < 0 || !(lo[j] < hi[j]))
                    return false;

            std::vector<std::pair<int, int>> order(2 * n); // (row, code): code < n upper, else lower
            for (int j = 0; j < n; ++j)
            {
                order[j] = {hi_row[j], j};
                order[n + j] = {lo_row[j], n + j};
            }
            std::sort(order.begin(), order.end());

            allocate(n);
            for (int j = 0; j < n; ++j)
            {
                lb_[j] = lo[j];
                ub_[j] = hi[j];
            }
            for (int p = 0; p < 2 * n; ++p)
            {
                const int code = order[p].second;
                const bool upper = code < n;
                const int v = upper ? code : code - n;
                pvar_[p] = v;
                prow_[p] = order[p].first;
                psign_[p] = upper ? 1.0 : -1.0;
                pb_[p] = upper ? ub_[v] : -lb_[v];
                (upper ? pos_upper_ : pos_lower_)[v] = p;
            }
            return true;
        }

        bool BoxKernel::contains(const double *x, double tol) const
        {
            for (int p = 0; p < 2 * n_; ++p)
                if (psign_[p] * x[pvar_[p]] > pb_[p] + tol)
                    return false;
            return true;
        }

        const double *BoxKernel::q_inv_row(const double *L, int v)
        {
            int slot = row_slot_[v];
            if (slot < 0)
            {
                slot = rows_used_++;
                row_slot_[v] = slot;
                const size_t need = static_cast<size_t>(rows_used_) * n_;
                if (rows_.size() < need)
                    rows_.resize(std::max(need, 2 * rows_.size()));
                // Q⁻¹e_v = L⁻ᵀ(L⁻¹e_v); the forward solve skips the leading zeros.
                LinearAlgebra::forward_sub_unit(L, v, unit_, n_);
                LinearAlgebra::backward_sub(L, unit_, &rows_[static_cast<size_t>(slot) * n_], n_);
            }
            return &rows_[static_cast<size_t>(slot) * n_];
        }

        // Projection of x̄ onto the face W_ in the Q-metric: x_ and lam_.
        // Returns whether x_ is feasible, with the semantics of the row-based
        // path (dense Q: every bound is tested; diagonal Q: the inactive ones).
        bool BoxKernel::project(const double *L, const double *x_bar, const Config &config)
        {
            const int n = n_;
            const int k = W_size_;
            const double ctol = config.constraint_tolerance;
            std::copy(x_bar, x_bar + n, x_);
            solve_failed_ = false;

            if (k == 0)
                return contains(x_bar, ctol);

            if (q_diag_)
            {
                for (int i = 0; i < k; ++i)
                {
                    const int p = W_[i];
                    const int v = pvar_[p];
                    const double s = psign_[p];
                    lam_[i] = (s * x_bar[v] - pb_[p]) * q_diagonal_[v];
                    x_[v] = pb_[p] / s;
                }
                for (int p = 0; p < 2 * n; ++p)
                    if (!in_W_[p] && psign_[p] * x_[pvar_[p]] > pb_[p] + ctol)
                        return false;
                return true;
            }

            // Dense Q.  Make sure every needed row of Q⁻¹ exists before taking
            // pointers (computing a row may grow the pool).
            for (int i = 0; i < k; ++i)
                q_inv_row(L, pvar_[W_[i]]);

            if (rhs_.size() < static_cast<size_t>(k))
                rhs_.resize(std::max<size_t>(k, 2 * rhs_.size()));
            if (S_.size() < static_cast<size_t>(k) * k)
                S_.resize(std::max<size_t>(static_cast<size_t>(k) * k, 2 * S_.size()));
            for (int i = 0; i < k; ++i)
            {
                const int pi = W_[i];
                const int vi = pvar_[pi];
                const double si = psign_[pi];
                rhs_[i] = si * x_bar[vi] - pb_[pi];
                for (int j = 0; j < k; ++j)
                {
                    const int pj = W_[j];
                    const double *GEOQP_RESTRICT Rj = &rows_[static_cast<size_t>(row_slot_[pvar_[pj]]) * n];
                    S_[static_cast<size_t>(i) * k + j] = (Rj[vi] * psign_[pj]) * si;
                }
            }

            // S and rhs are rebuilt every time, so they can be eliminated in place.
            if (!LinearAlgebra::solve_linear_system_inplace(S_.data(), rhs_.data(), lam_, k))
            {
                solve_failed_ = true;
                return false; // x_ = x̄, as in the row-based path
            }

            double *GEOQP_RESTRICT x = x_;
            for (int j = 0; j < k; ++j)
            {
                const int pj = W_[j];
                const double sj = psign_[pj];
                const double lj = lam_[j];
                const double *GEOQP_RESTRICT Rj = &rows_[static_cast<size_t>(row_slot_[pvar_[pj]]) * n];
                for (int i = 0; i < n; ++i)
                    x[i] -= (Rj[i] * sj) * lj;
            }

            for (int p = 0; p < 2 * n; ++p)
                if (psign_[p] * x[pvar_[p]] > pb_[p] + ctol)
                    return false;
            return true;
        }

        bool BoxKernel::solve(const double *Q, const double *L, const double *x_bar, const Config &config)
        {
            const int n = n_;
            xbar_ = x_bar;
            iterations_ = 0;
            hops_ = 0;

            // Diagonal Hessian?  (same test as FaceHopper::init_common)
            q_diag_ = true;
            for (int i = 0; i < n; ++i)
            {
                q_diagonal_[i] = Q[static_cast<size_t>(i) * n + i];
                if (q_diag_)
                    for (int j = 0; j < n; ++j)
                        if (i != j && std::abs(Q[static_cast<size_t>(i) * n + j]) > 1e-12)
                        {
                            q_diag_ = false;
                            break;
                        }
            }

            // Antipodal frame (Initializer::compute_antipodal_frame).
            violated_frame_size_ = 0;
            int i0 = -1;
            double best = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < n; ++j)
            {
                const double lo = lb_[j], hi = ub_[j];
                const double mu = 0.5 * (lo + hi);
                const double h = 0.5 * (hi - lo);
                const double xb = x_bar[j];
                const bool up = (xb >= mu);
                s_frame_[j] = up ? 1.0 : -1.0;
                h_frame_[j] = h;
                v_hat_[j] = up ? lo : hi;
                const double zeta = (up ? (xb - lo) : (hi - xb)) / h;
                if (xb > hi || xb < lo)
                    violated_frame_[violated_frame_size_++] = j;
                if (zeta > best)
                {
                    best = zeta;
                    i0 = j;
                }
            }

            // H₁ hint: −∇f(v̂) = Q(x̄ − v̂).
            for (int j = 0; j < n; ++j)
                x_[j] = x_bar[j] - v_hat_[j];
            if (q_diag_)
                for (int j = 0; j < n; ++j)
                    neg_grad_[j] = q_diagonal_[j] * x_[j];
            else
                LinearAlgebra::matvec(Q, x_, neg_grad_, n, n);

            // Working set: the exit facet F₀.
            std::fill(in_W_, in_W_ + 2 * n, 0);
            W_size_ = 0;
            if (violated_frame_size_ > 0)
            {
                const int p0 = (s_frame_[i0] > 0.0) ? pos_upper_[i0] : pos_lower_[i0];
                W_[W_size_++] = p0;
                in_W_[p0] = 1;
            }
            std::fill(row_slot_, row_slot_ + n, -1);
            rows_used_ = 0;

            const double ctol = config.constraint_tolerance;
            bool last_was_add = false;
            bool verified = false;

            while (iterations_ < config.max_iterations)
            {
                ++iterations_;
                const bool feasible = project(L, x_bar, config);
                const int k = W_size_;

                if (feasible)
                {
                    int drop = -1;
                    double most_negative = -ctol;
                    for (int i = 0; i < k; ++i)
                        if (lam_[i] < most_negative)
                        {
                            most_negative = lam_[i];
                            drop = i;
                        }
                    if (drop < 0)
                    {
                        verified = true; // feasible, all multipliers ≥ −tol
                        break;
                    }
                    if (last_was_add)
                        ++hops_;
                    last_was_add = false;
                    in_W_[W_[drop]] = 0;
                    for (int i = drop; i + 1 < k; ++i)
                        W_[i] = W_[i + 1];
                    --W_size_;
                    continue;
                }

                // Infeasible projection: add violated bounds.
                last_was_add = true;
                int nviol = 0;
                for (int p = 0; p < 2 * n; ++p)
                {
                    if (in_W_[p])
                        continue;
                    const int v = pvar_[p];
                    const double s = psign_[p];
                    if (s * x_[v] > pb_[p] + ctol && neg_grad_[v] * s >= -config.tolerance)
                        viol_[nviol++] = p;
                }
                if (nviol == 0)
                    for (int p = 0; p < 2 * n; ++p)
                        if (!in_W_[p] && psign_[p] * x_[pvar_[p]] > pb_[p] + ctol)
                            viol_[nviol++] = p;
                if (nviol == 0)
                    break; // infeasible only on active bounds: numerical; project once more below

                if (nviol <= n - k)
                {
                    for (int i = 0; i < nviol; ++i)
                    {
                        W_[W_size_++] = viol_[i];
                        in_W_[viol_[i]] = 1;
                    }
                }
                else
                {
                    int most = viol_[0];
                    double max_violation = -std::numeric_limits<double>::infinity();
                    for (int i = 0; i < nviol; ++i)
                    {
                        const int p = viol_[i];
                        const double violation = psign_[p] * x_[pvar_[p]] - pb_[p];
                        if (violation > max_violation)
                        {
                            max_violation = violation;
                            most = p;
                        }
                    }
                    W_[W_size_++] = most;
                    in_W_[most] = 1;
                }
                std::sort(W_, W_ + W_size_);
            }

            if (!verified)
            {
                // The row-based path returned the face as is and projected onto it
                // in Phase III; do the same and fail only if that projection cannot
                // be computed.
                project(L, x_bar, config);
                if (solve_failed_)
                    return false;
            }

            for (int i = 0; i < W_size_; ++i)
                active_rows_[i] = prow_[W_[i]];
            return true;
        }

        int BoxKernel::violated_active_count(double tol) const
        {
            int count = 0;
            for (int i = 0; i < W_size_; ++i)
            {
                const int p = W_[i];
                if (lam_[i] > tol && psign_[p] * xbar_[pvar_[p]] > pb_[p])
                    ++count;
            }
            return count;
        }

        double BoxKernel::visible_cut_margin() const
        {
            if (violated_frame_size_ == 0)
                return std::numeric_limits<double>::quiet_NaN();
            double sum = 0.0;
            for (int t = 0; t < violated_frame_size_; ++t)
            {
                const int i = violated_frame_[t];
                sum += s_frame_[i] * (x_[i] - v_hat_[i]) / h_frame_[i];
            }
            return sum - 2.0;
        }

    } // namespace internal
} // namespace geoqp
