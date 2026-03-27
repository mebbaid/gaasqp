#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include <geoqp/timing.hpp>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <chrono>

namespace geoqp
{
    namespace internal
    {
        // Helper: Print timing statistics
        template <typename TimingStats>
        static inline void print_timing_summary(const TimingStats &stats)
        {
            std::cout << "FaceHopper timing summary:\n";
            std::cout << "  compute_intersection: " << stats.compute_intersection_calls
                      << " calls, " << stats.compute_intersection_total << " ms\n";
            std::cout << "  find_initial_face: " << stats.find_initial_face_calls
                      << " calls, " << stats.find_initial_face_total << " ms\n";
            std::cout << "  compute_projection: " << stats.compute_projection_calls
                      << " calls, " << stats.compute_projection_total << " ms\n";
            std::cout << "  find_optimal_face: " << stats.find_optimal_face_calls
                      << " calls, " << stats.find_optimal_face_total << " ms\n";
        }

        FaceHopper::FaceHopper(
            const double *Q, const double *A, const double *b,
            const std::vector<double> &x_bar,
            int m, int n,
            const Config &config) : Q_(Q), A_(A), b_(b), x_bar_(x_bar), m_(m), n_(n), config_(config),
                                    constraint_cache_(m, n)
        {
            // No pre-supplied Cholesky — compute it here.
            Q_chol_.resize(n_ * n_);
            try
            {
                LinearAlgebra::cholesky(Q_, Q_chol_.data(), n_);
            }
            catch (...)
            {
                Q_chol_.clear();
            }
            init_common();
        }

        FaceHopper::FaceHopper(
            const double *Q, const double *A, const double *b,
            const std::vector<double> &x_bar,
            int m, int n,
            const Config &config,
            const std::vector<double> &Q_chol)
            : Q_(Q), A_(A), b_(b), x_bar_(x_bar), m_(m), n_(n), config_(config),
              constraint_cache_(m, n), Q_chol_(Q_chol)
        {
            // Q_chol already set from caller — skip O(n³) factorization.
            init_common();
        }

        void FaceHopper::init_common()
        {
            const int total_elements = n_ * n_;
            const double *GEOQP_RESTRICT Q_ptr = Q_;
            const double tol = 1e-12;

            int nonzero_q = 0;
            int i = 0;
            GEOQP_UNROLL_LOOP_4
            for (; i + 3 < total_elements; i += 4)
            {
                nonzero_q += (std::abs(Q_ptr[i]) > tol);
                nonzero_q += (std::abs(Q_ptr[i + 1]) > tol);
                nonzero_q += (std::abs(Q_ptr[i + 2]) > tol);
                nonzero_q += (std::abs(Q_ptr[i + 3]) > tol);
            }
            for (; i < total_elements; ++i)
            {
                nonzero_q += (std::abs(Q_ptr[i]) > tol);
            }
            q_is_dense_ = (nonzero_q > (total_elements / 2));

            // Detect diagonal Q and extract diagonal in one pass
            q_is_diagonal_ = true;
            Q_diag_.resize(n_);
            for (int i = 0; i < n_; ++i)
            {
                Q_diag_[i] = Q_[i * n_ + i];
                if (q_is_diagonal_)
                {
                    for (int j = 0; j < n_; ++j)
                    {
                        if (i != j && std::abs(Q_[i * n_ + j]) > 1e-12)
                        {
                            q_is_diagonal_ = false;
                            break;
                        }
                    }
                }
            }

            // Precompute Q_inv if Q is dense and n is manageable
            if (q_is_dense_ && n_ <= 256 && !Q_chol_.empty())
            {
                Q_inv_.resize(n_ * n_);
                std::vector<double> e(n_, 0.0);
                for (int i = 0; i < n_; ++i)
                {
                    e[i] = 1.0;
                    LinearAlgebra::solve_cholesky(Q_chol_.data(), e.data(), &Q_inv_[i * n_], n_);
                    e[i] = 0.0;
                }
                if (config_.verbosity >= 2)
                    std::cout << "  Q is dense - precomputed Q_inv for fast access\n";
            }

            // Preallocate all workspaces up-front
            ws_residual_.resize(m_);
            ws_w_.resize(n_);
            ws_correction_.resize(n_);
            ws_x_star_.resize(n_);
            ws_lambda_.resize(m_);
            ws_Axbar_.resize(m_);
            ws_is_active_.assign(m_, 0);
            ws_opt_x_star_.resize(n_);
            ws_opt_lambda_.resize(m_);
            ws_opt_violated_.reserve(m_);
            ws_opt_smaller_face_.reserve(m_);
            ws_opt_is_active_set_.resize(m_, 0);

            LinearAlgebra::matvec(A_, x_bar_.data(), ws_Axbar_.data(), m_, n_);

            // Single-pass: detect box constraints and populate constraint_var_/constraint_sign_
            constraint_var_.resize(m_);
            constraint_sign_.resize(m_);
            has_diagonal_constraints_ = true;
            for (int i = 0; i < m_; ++i)
            {
                const double *GEOQP_RESTRICT a_row = A_ + i * n_;
                int nz_count = 0, nz_idx = -1;
                double nz_val = 0.0;
                for (int j = 0; j < n_; ++j)
                {
                    const double v = a_row[j];
                    if (std::abs(v) > 1e-10)
                    {
                        ++nz_count;
                        nz_idx = j;
                        nz_val = v;
                        if (nz_count > 1 || std::abs(std::abs(v) - 1.0) > 1e-10)
                        {
                            has_diagonal_constraints_ = false;
                            // still populate for later fallback paths
                            break;
                        }
                    }
                }
                constraint_var_[i]  = (nz_idx >= 0) ? nz_idx : 0;
                constraint_sign_[i] = (nz_val > 0.0) ? 1.0 : -1.0;
                if (nz_count != 1) has_diagonal_constraints_ = false;
            }

            if (has_diagonal_constraints_ && config_.verbosity >= 2)
                std::cout << "  Detected diagonal/box constraints - using fast path\n";
            if (config_.use_caching && !Q_chol_.empty() && config_.verbosity >= 2)
                std::cout << "  Incremental Schur complement caching enabled\n";
        }

        double FaceHopper::compute_intersection_parameter(
            const std::vector<double> &v,
            int constraint_idx) const
        {
            Timer t;
            double av, ad;

            if (GEOQP_LIKELY(has_diagonal_constraints_))
            {
                // O(1) fast path for box/diagonal constraints
                const int var       = constraint_var_[constraint_idx];
                const double sign   = constraint_sign_[constraint_idx];
                av = sign * v[var];
                ad = sign * (x_bar_[var] - v[var]);
            }
            else
            {
                const double *GEOQP_RESTRICT a_row  = A_ + constraint_idx * n_;
                const double *GEOQP_RESTRICT v_ptr  = v.data();
                const double *GEOQP_RESTRICT xb_ptr = x_bar_.data();

                double av0 = 0.0, av1 = 0.0, av2 = 0.0, av3 = 0.0;
                double ad0 = 0.0, ad1 = 0.0, ad2 = 0.0, ad3 = 0.0;
                int j = 0;
                for (; j + 3 < n_; j += 4)
                {
                    av0 += a_row[j]     * v_ptr[j];
                    av1 += a_row[j + 1] * v_ptr[j + 1];
                    av2 += a_row[j + 2] * v_ptr[j + 2];
                    av3 += a_row[j + 3] * v_ptr[j + 3];
                    ad0 += a_row[j]     * (xb_ptr[j]     - v_ptr[j]);
                    ad1 += a_row[j + 1] * (xb_ptr[j + 1] - v_ptr[j + 1]);
                    ad2 += a_row[j + 2] * (xb_ptr[j + 2] - v_ptr[j + 2]);
                    ad3 += a_row[j + 3] * (xb_ptr[j + 3] - v_ptr[j + 3]);
                }
                av = (av0 + av1) + (av2 + av3);
                ad = (ad0 + ad1) + (ad2 + ad3);
                for (; j < n_; ++j)
                {
                    av += a_row[j] * v_ptr[j];
                    ad += a_row[j] * (xb_ptr[j] - v_ptr[j]);
                }
            }

            if (std::abs(ad) < 1e-14)
            {
                double res = (std::abs(av - b_[constraint_idx]) < 1e-10) ? 0.0 : std::numeric_limits<double>::infinity();
                if (config_.timing_verbosity > 0)
                {
                    timing_stats_.compute_intersection_total += t.elapsed_ms();
                    ++timing_stats_.compute_intersection_calls;
                }
                return res;
            }
            double out = (b_[constraint_idx] - av) / ad;
            if (config_.timing_verbosity > 0)
            {
                timing_stats_.compute_intersection_total += t.elapsed_ms();
                ++timing_stats_.compute_intersection_calls;
            }
            return out;
        }

        std::vector<int> FaceHopper::find_initial_face(
            const std::vector<double> &v)
        {
            // ── Precompute H1/H2 hint vectors (Proposition 2) ────────────────────
            // These are fixed for the entire Phase II loop.
            //   d         = x̄ − v̂               (H2 inward normal, O(n))
            //   -∇f(v̂)   = Q·d = Q(x̄ − v̂)      (H1 inward normal, O(n) or O(n²))
            // Stored in ws_d_vhat_ / ws_neg_grad_vhat_; guarded by has_vhat_hint_.
            has_vhat_hint_ = false;
            if (has_diagonal_constraints_)
            {
                ws_d_vhat_.resize(n_);
                ws_neg_grad_vhat_.resize(n_);
                const double *GEOQP_RESTRICT xb  = x_bar_.data();
                const double *GEOQP_RESTRICT vp  = v.data();
                double       *GEOQP_RESTRICT dv  = ws_d_vhat_.data();
                for (int j = 0; j < n_; ++j)
                    dv[j] = xb[j] - vp[j];
                if (q_is_diagonal_)
                {
                    const double *GEOQP_RESTRICT qd = Q_diag_.data();
                    double       *GEOQP_RESTRICT ng = ws_neg_grad_vhat_.data();
                    for (int j = 0; j < n_; ++j)
                        ng[j] = qd[j] * dv[j];
                }
                else
                {
                    // Dense Q: one matvec Q·d, O(n²) — done once per solve.
                    LinearAlgebra::matvec(Q_, dv, ws_neg_grad_vhat_.data(), n_, n_);
                }
                has_vhat_hint_ = true;
            }

            Timer t;

            // Active/blocking checks: O(1) per constraint for box, O(n) otherwise
            std::vector<int> active_at_v;
            std::vector<int> blocking_at_v;

            if (has_diagonal_constraints_)
            {
                const int *GEOQP_RESTRICT var_ptr = constraint_var_.data();
                const double *GEOQP_RESTRICT sgn_ptr = constraint_sign_.data();
                const double *GEOQP_RESTRICT v_ptr = v.data();
                const double *GEOQP_RESTRICT xb_ptr = x_bar_.data();
                for (int i = 0; i < m_; ++i)
                {
                    const int var     = var_ptr[i];
                    const double sign = sgn_ptr[i];
                    const double av   = sign * v_ptr[var];
                    if (std::abs(av - b_[i]) < 1e-8)
                    {
                        active_at_v.push_back(i);
                        if (sign * xb_ptr[var] > b_[i] + 1e-8)
                            blocking_at_v.push_back(i);
                    }
                }
            }
            else
            {
                for (int i = 0; i < m_; ++i)
                {
                    const double *a_row = A_ + i * n_;
                    double av = LinearAlgebra::dot(a_row, v.data(), n_);
                    if (std::abs(av - b_[i]) < 1e-8)
                    {
                        active_at_v.push_back(i);
                        if (LinearAlgebra::dot(a_row, x_bar_.data(), n_) > b_[i] + 1e-8)
                            blocking_at_v.push_back(i);
                    }
                }
            }

            if (!blocking_at_v.empty())
            {
                if (config_.timing_verbosity > 0)
                {
                    timing_stats_.find_initial_face_total += t.elapsed_ms();
                    ++timing_stats_.find_initial_face_calls;
                }
                return blocking_at_v;
            }

            // Single pass: compute all t values, find t_max_in01 and t_min simultaneously
            ws_t_values_.resize(m_);
            double t_max_in01 = -std::numeric_limits<double>::infinity();
            double t_min      =  std::numeric_limits<double>::infinity();
            for (int i = 0; i < m_; ++i)
            {
                const double ti = compute_intersection_parameter(v, i);
                ws_t_values_[i] = ti;
                if (ti > 1e-10 && ti < 1.0 - 1e-10 && ti > t_max_in01)
                    t_max_in01 = ti;
                if (ti > 1e-10 && ti < t_min)
                    t_min = ti;
            }

            const double tol = 1e-8;
            std::vector<int> initial_face;

            if (std::isfinite(t_max_in01))
            {
                for (int i = 0; i < m_; ++i)
                {
                    if (std::abs(ws_t_values_[i] - t_max_in01) <= tol)
                        initial_face.push_back(i);
                }
                if (std::getenv("GEOQP_DBG_INIT_PATH") && std::string(std::getenv("GEOQP_DBG_INIT_PATH")) == "1")
                {
                    std::fprintf(stderr, "[DBG] FaceHopper: picked t_max_in01=%.12g with %zu constraints\n", t_max_in01, initial_face.size());
                }
            }
            else
            {
                // fallback: smallest positive t (legacy behavior)
                if (t_min > 1.0 + 1e-8)
                {
                    bool x_bar_feasible = true;
                    for (int i = 0; i < m_; ++i)
                    {
                        if (has_diagonal_constraints_)
                        {
                            if (constraint_sign_[i] * x_bar_[constraint_var_[i]] > b_[i] + config_.tolerance)
                            {
                                x_bar_feasible = false;
                                break;
                            }
                        }
                        else
                        {
                            const double *a_row = A_ + i * n_;
                            if (LinearAlgebra::dot(a_row, x_bar_.data(), n_) > b_[i] + config_.tolerance)
                            {
                                x_bar_feasible = false;
                                break;
                            }
                        }
                    }

                    if (x_bar_feasible)
                    {
                        if (config_.timing_verbosity > 0)
                        {
                            timing_stats_.find_initial_face_total += t.elapsed_ms();
                            ++timing_stats_.find_initial_face_calls;
                        }
                        return std::vector<int>();
                    }
                }

                if (!std::isfinite(t_min))
                {
                    if (config_.timing_verbosity > 0)
                    {
                        timing_stats_.find_initial_face_total += t.elapsed_ms();
                        ++timing_stats_.find_initial_face_calls;
                    }
                    return std::vector<int>();
                }

                for (int i = 0; i < m_; ++i)
                {
                    if (std::abs(ws_t_values_[i] - t_min) <= tol)
                        initial_face.push_back(i);
                }
            }

            if (config_.timing_verbosity > 0)
            {
                timing_stats_.find_initial_face_total += t.elapsed_ms();
                ++timing_stats_.find_initial_face_calls;
            }

            return initial_face;
        }

        bool FaceHopper::compute_projection_and_lambda(
            const std::vector<int> &face,
            std::vector<double> &x_star,
            std::vector<double> &lambda) const
        {
            Timer t;

            if (config_.use_caching && !Q_chol_.empty() && !has_diagonal_constraints_)
            {
                bool success = compute_projection_cached(face, x_star, lambda);

                if (success)
                {
                    for (int i = 0; i < m_; ++i)
                    {
                        const double *a_row = A_ + i * n_;
                        double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
                        if (ax > b_[i] + config_.constraint_tolerance)
                        {
                            success = false;
                            break;
                        }
                    }
                }

                if (config_.timing_verbosity > 0)
                {
                    timing_stats_.compute_projection_total += t.elapsed_ms();
                    ++timing_stats_.compute_projection_calls;
                }

                if (success)
                    return true;
            }

            const int k = static_cast<int>(face.size());
            x_star = x_bar_;
            lambda.resize(k);

            if (k == 0)
            {
                for (int i = 0; i < m_; ++i)
                {
                    const double *a_row = A_ + i * n_;
                    double ax = LinearAlgebra::dot(a_row, x_bar_.data(), n_);
                    if (ax > b_[i] + config_.constraint_tolerance)
                    {
                        if (config_.timing_verbosity > 0)
                        {
                            timing_stats_.compute_projection_total += t.elapsed_ms();
                            ++timing_stats_.compute_projection_calls;
                        }
                        return false;
                    }
                }
                if (config_.timing_verbosity > 0)
                {
                    timing_stats_.compute_projection_total += t.elapsed_ms();
                    ++timing_stats_.compute_projection_calls;
                }
                return true;
            }

            if (has_diagonal_constraints_ && q_is_diagonal_)
            {
                // ── O(k) closed-form projection for diagonal Q + box faces ──────────
                // Each active constraint i: sign_i * x[var_i] <= b_i.
                // KKT stationarity: Q·(x*−x̄) + Aᵀλ = 0
                //   → q[v]·(x*[v] − x̄[v]) + sign_i·λ_i = 0   for the variable v = var_i
                //   → x*[v] = x̄[v] − sign_i·λ_i / q[v]
                // Primal complementarity: sign_i·x*[v] = b_i
                //   → sign_i·(x̄[v] − sign_i·λ_i/q[v]) = b_i
                //   → λ_i = (sign_i·x̄[v] − b_i)·q[v]          (sign_i² = 1)
                //   → x*[v] = b_i / sign_i                      (= the bound itself)
                //
                // If two active constraints share the same variable (conflicting bounds),
                // fall through to the general path.

                x_star = x_bar_;
                lambda.resize(k);

                double *GEOQP_RESTRICT xstar_ptr = x_star.data();
                double *GEOQP_RESTRICT lam_ptr   = lambda.data();

                // Stack-allocated touched-variable list — k ≤ n_ ≤ 512 for typical problems
                int touched_arr[512];
                int n_touched = 0;
                bool conflict = false;

                // ws_is_active_ was pre-sized to m_ (≥ n_ for box problems); use as per-var flag
                for (int i = 0; i < k; ++i)
                {
                    const int idx  = face[i];
                    const int v    = constraint_var_[idx];
                    const double s = constraint_sign_[idx];
                    const double b = b_[idx];

                    if (GEOQP_UNLIKELY(ws_is_active_[v]))
                    {
                        conflict = true;
                        break;
                    }
                    ws_is_active_[v] = 1;
                    touched_arr[n_touched++] = v;

                    // λ_i = (s·x̄[v] − b)·q[v]
                    lam_ptr[i] = (s * x_bar_[v] - b) * Q_diag_[v];
                    // x*[v] = b / s  (snap to bound)
                    xstar_ptr[v] = b / s;
                }

                // Restore per-variable flags (only k entries, not full O(m) fill)
                for (int t = 0; t < n_touched; ++t)
                    ws_is_active_[touched_arr[t]] = 0;

                if (GEOQP_LIKELY(!conflict))
                {
                    // Mark active constraints; feasibility scan; restore — all O(k)+O(m)
                    for (int i = 0; i < k; ++i)
                        ws_opt_is_active_set_[face[i]] = 1;

                    const int *GEOQP_RESTRICT var_ptr = constraint_var_.data();
                    const double *GEOQP_RESTRICT sgn_ptr = constraint_sign_.data();
                    const char *GEOQP_RESTRICT is_active = ws_opt_is_active_set_.data();
                    bool feasible = true;
                    for (int i = 0; i < m_; ++i)
                    {
                        if (is_active[i]) continue;
                        if (sgn_ptr[i] * xstar_ptr[var_ptr[i]] > b_[i] + config_.constraint_tolerance)
                        {
                            feasible = false;
                            break;
                        }
                    }

                    // Restore active-set flags (O(k) restore, not O(m) fill)
                    for (int i = 0; i < k; ++i)
                        ws_opt_is_active_set_[face[i]] = 0;

                    return feasible;
                }
                // Fall through to general path if two constraints share a variable
            }

            if (has_diagonal_constraints_)
            {
                // Use workspace members — no heap allocation per call
                if (static_cast<int>(ws_constrained_vars_.size()) < k)
                {
                    ws_constrained_vars_.resize(k);
                    ws_signs_.resize(k);
                }
                if (static_cast<int>(ws_residual_.size()) < k)
                    ws_residual_.resize(k);

                int *GEOQP_RESTRICT c_vars   = ws_constrained_vars_.data();
                double *GEOQP_RESTRICT c_signs   = ws_signs_.data();
                double *GEOQP_RESTRICT c_residual = ws_residual_.data();

                for (int i = 0; i < k; ++i)
                {
                    const int idx  = face[i];
                    c_vars[i]     = constraint_var_[idx];
                    c_signs[i]    = constraint_sign_[idx];
                    c_residual[i] = ws_Axbar_[idx] - b_[idx];
                }

                // Use a pre-allocated workspace for Q_inv_columns instead of std::vector
                if (static_cast<int>(ws_Q_inv_cols_.size()) < n_ * k)
                {
                    ws_Q_inv_cols_.resize(n_ * k);
                }
                double *GEOQP_RESTRICT Q_inv_columns = ws_Q_inv_cols_.data();

                for (int j = 0; j < k; ++j)
                {
                    double *GEOQP_RESTRICT target_col = &Q_inv_columns[j * n_];
                    const double s = c_signs[j];
                    const int v = c_vars[j];

                    if (q_is_diagonal_)
                    {
                        std::fill(target_col, target_col + n_, 0.0);
                        target_col[v] = s / Q_diag_[v];
                    }
                    else if (!Q_inv_.empty())
                    {
                        const double *GEOQP_RESTRICT Qi = &Q_inv_[v * n_];
                        int i = 0;
                        GEOQP_UNROLL_LOOP_4
                        for (; i + 3 < n_; i += 4)
                        {
                            target_col[i] = Qi[i] * s;
                            target_col[i + 1] = Qi[i + 1] * s;
                            target_col[i + 2] = Qi[i + 2] * s;
                            target_col[i + 3] = Qi[i + 3] * s;
                        }
                        for (; i < n_; ++i)
                            target_col[i] = Qi[i] * s;
                    }
                    else if (!Q_chol_.empty())
                    {
                        std::vector<double> rhs(n_, 0.0);
                        rhs[v] = s;
                        LinearAlgebra::solve_cholesky(Q_chol_.data(), rhs.data(), target_col, n_);
                    }
                    else
                    {
                        return false;
                    }
                }

                if (static_cast<int>(ws_S_.size()) < k * k)
                    ws_S_.resize(k * k);
                double *GEOQP_RESTRICT S = ws_S_.data();

                for (int i = 0; i < k; ++i)
                {
                    const int var_i = c_vars[i];
                    const double sign_i = c_signs[i];
                    for (int j = 0; j < k; ++j)
                    {
                        S[i * k + j] = Q_inv_columns[j * n_ + var_i] * sign_i;
                    }
                }

                try
                {
                    LinearAlgebra::solve_linear_system(S, c_residual, lambda.data(), k);
                }
                catch (...)
                {
                    return false;
                }

                x_star = x_bar_;
                double *GEOQP_RESTRICT x_ptr = x_star.data();
                const double *GEOQP_RESTRICT lambda_ptr = lambda.data();
                for (int j = 0; j < k; ++j)
                {
                    const double l_j = lambda_ptr[j];
                    const double *GEOQP_RESTRICT col = &Q_inv_columns[j * n_];
                    int i = 0;
                    GEOQP_UNROLL_LOOP_4
                    for (; i + 3 < n_; i += 4)
                    {
                        x_ptr[i] -= col[i] * l_j;
                        x_ptr[i + 1] -= col[i + 1] * l_j;
                        x_ptr[i + 2] -= col[i + 2] * l_j;
                        x_ptr[i + 3] -= col[i + 3] * l_j;
                    }
                    for (; i < n_; ++i)
                        x_ptr[i] -= col[i] * l_j;
                }

                for (int i = 0; i < m_; ++i)
                {
                    const int v = constraint_var_[i];
                    if (constraint_sign_[i] * x_star[v] > b_[i] + config_.constraint_tolerance)
                        return false;
                }

                return true;
            }

            bool success = ProjectionSolver::solve(
                face, Q_, A_, b_, x_bar_, Q_chol_,
                m_, n_, x_star, lambda,
                Q_inv_, q_is_diagonal_, Q_diag_);

            if (!success)
            {
                return false;
            }

            for (int i = 0; i < m_; ++i)
            {
                const double *a_row = A_ + i * n_;
                double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
                if (ax > b_[i] + config_.constraint_tolerance)
                {
                    return false;
                }
            }

            return true;
        }

        bool FaceHopper::compute_projection_cached(
            const std::vector<int> &face,
            std::vector<double> &x_star,
            std::vector<double> &lambda) const
        {
            const int k = static_cast<int>(face.size());

            if (k == 0)
            {
                x_star = x_bar_;
                lambda.clear();
                return true;
            }

            // Ensure residual workspace matches schur dimension exactly
            if (static_cast<int>(ws_residual_.size()) != k)
                ws_residual_.resize(k);
            if (static_cast<int>(ws_w_.size()) < n_)
                ws_w_.resize(n_);
            if (static_cast<int>(ws_correction_.size()) < n_)
                ws_correction_.resize(n_);

            const double *GEOQP_RESTRICT xbar_ptr = x_bar_.data();
            double *GEOQP_RESTRICT res_ptr = ws_residual_.data();

            for (int i = 0; i < k; ++i)
            {
                const int idx = face[i];
                const double *GEOQP_RESTRICT a_row = A_ + idx * n_;
                res_ptr[i] = LinearAlgebra::dot(a_row, xbar_ptr, n_) - b_[idx];
            }

            if (!schur_manager_.initialize(face, A_, Q_chol_.data(), n_, constraint_cache_))
            {
                return false;
            }

            if (!schur_manager_.solve(ws_residual_, lambda))
            {
                return false;
            }

            std::fill(ws_w_.begin(), ws_w_.begin() + n_, 0.0);

            double *GEOQP_RESTRICT w_ptr = ws_w_.data();
            const double *GEOQP_RESTRICT lambda_ptr = lambda.data();

            for (int j = 0; j < k; ++j)
            {
                const auto &z_j = constraint_cache_.get_z(face[j], A_, Q_chol_.data());
                const double *GEOQP_RESTRICT z_ptr = z_j.data();
                const double lambda_j = lambda_ptr[j];

                int i = 0;
                for (; i + 3 < n_; i += 4)
                {
                    w_ptr[i] += lambda_j * z_ptr[i];
                    w_ptr[i + 1] += lambda_j * z_ptr[i + 1];
                    w_ptr[i + 2] += lambda_j * z_ptr[i + 2];
                    w_ptr[i + 3] += lambda_j * z_ptr[i + 3];
                }
                for (; i < n_; ++i)
                {
                    w_ptr[i] += lambda_j * z_ptr[i];
                }
            }

            LinearAlgebra::backward_sub(Q_chol_.data(), w_ptr, ws_correction_.data(), n_);

            x_star.resize(n_);
            double *GEOQP_RESTRICT xstar_ptr = x_star.data();
            const double *GEOQP_RESTRICT corr_ptr = ws_correction_.data();

            for (int i = 0; i < n_; ++i)
            {
                xstar_ptr[i] = xbar_ptr[i] - corr_ptr[i];
            }

            return true;
        }

        std::vector<int> FaceHopper::find_optimal_face_cached(
            const std::vector<int> &initial_face,
            int &iterations,
            Timer &t) const
        {
            iterations = 0;
            const int max_iter = config_.max_iterations;

            if (!schur_manager_.initialize(initial_face, A_, Q_chol_.data(), n_, constraint_cache_))
            {
                // We need to use the non-cached projection method instead
                std::vector<int> current_face = initial_face;
                std::vector<double> x_proj, lambda;

                for (int iter = 0; iter < config_.max_iterations; ++iter)
                {
                    // count an iteration only when the face actually changes
                    if (!compute_projection_and_lambda(current_face, x_proj, lambda))
                    {
                        break;
                    }

                    int most_neg = -1;
                    double most_neg_val = -config_.constraint_tolerance;
                    for (size_t i = 0; i < current_face.size(); ++i)
                    {
                        if (lambda[i] < most_neg_val)
                        {
                            most_neg_val = lambda[i];
                            most_neg = static_cast<int>(i);
                        }
                    }

                    if (most_neg >= 0)
                    {
                        current_face.erase(current_face.begin() + most_neg);
                        iterations++;
                        continue;
                    }

                    std::vector<int> violated;
                    for (int i = 0; i < m_; ++i)
                    {
                        bool is_active = false;
                        for (int idx : current_face)
                        {
                            if (idx == i)
                            {
                                is_active = true;
                                break;
                            }
                        }
                        if (is_active)
                            continue;

                        const double *a_row = A_ + i * n_;
                        double ax = LinearAlgebra::dot(a_row, x_proj.data(), n_);
                        if (ax > b_[i] + config_.constraint_tolerance)
                        {
                            violated.push_back(i);
                        }
                    }

                    if (violated.empty())
                    {
                        return current_face;
                    }

                    for (int idx : violated)
                    {
                        current_face.push_back(idx);
                    }
                    std::sort(current_face.begin(), current_face.end());
                    iterations++;
                }

                return current_face;
            }

            while (iterations < max_iter)
            {
                const auto prev_face = schur_manager_.get_active_constraints();
                const auto &current_face = schur_manager_.get_active_constraints();
                const int k = schur_manager_.size();

                // Always resize to exact size needed
                ws_residual_.resize(k);
                ws_w_.resize(n_);
                ws_correction_.resize(n_);
                ws_x_star_.resize(n_);
                ws_lambda_.resize(k);

                const double *GEOQP_RESTRICT xbar_ptr = x_bar_.data();

                for (int i = 0; i < k; ++i)
                {
                    const int idx = current_face[i];
                    const double *GEOQP_RESTRICT a_row = A_ + idx * n_;
                    ws_residual_[i] = LinearAlgebra::dot(a_row, xbar_ptr, n_) - b_[idx];
                }

                if (!schur_manager_.solve(ws_residual_, ws_lambda_))
                {
                    // Return current face (caller should check KKT conditions)
                    break;
                }

                std::fill(ws_w_.begin(), ws_w_.begin() + n_, 0.0);

                double *GEOQP_RESTRICT w_ptr = ws_w_.data();
                const double *GEOQP_RESTRICT lambda_ptr = ws_lambda_.data();

                for (int j = 0; j < k; ++j)
                {
                    const auto &z_j = constraint_cache_.get_z(current_face[j], A_, Q_chol_.data());
                    const double *GEOQP_RESTRICT z_ptr = z_j.data();
                    const double lambda_j = lambda_ptr[j];

                    int i = 0;
                    for (; i + 3 < n_; i += 4)
                    {
                        w_ptr[i] += lambda_j * z_ptr[i];
                        w_ptr[i + 1] += lambda_j * z_ptr[i + 1];
                        w_ptr[i + 2] += lambda_j * z_ptr[i + 2];
                        w_ptr[i + 3] += lambda_j * z_ptr[i + 3];
                    }
                    for (; i < n_; ++i)
                    {
                        w_ptr[i] += lambda_j * z_ptr[i];
                    }
                }

                LinearAlgebra::backward_sub(Q_chol_.data(), w_ptr, ws_correction_.data(), n_);

                double *GEOQP_RESTRICT xstar_ptr = ws_x_star_.data();
                const double *GEOQP_RESTRICT corr_ptr = ws_correction_.data();

                for (int i = 0; i < n_; ++i)
                {
                    xstar_ptr[i] = xbar_ptr[i] - corr_ptr[i];
                }

                bool primal_feasible = true;
                int most_violated = -1;
                double max_violation = 0.0;

                if (static_cast<int>(ws_is_active_.size()) < m_)
                    ws_is_active_.assign(m_, 0);
                else
                    std::fill(ws_is_active_.begin(), ws_is_active_.end(), 0);
                for (int idx : current_face)
                    ws_is_active_[idx] = 1;

                for (int i = 0; i < m_; ++i)
                {
                    if (ws_is_active_[i])
                        continue;

                    const double *a_row = A_ + i * n_;
                    double ax = LinearAlgebra::dot(a_row, ws_x_star_.data(), n_);
                    double violation = ax - b_[i];

                    if (violation > max_violation + config_.constraint_tolerance)
                    {
                        max_violation = violation;
                        most_violated = i;
                        primal_feasible = false;
                    }
                }

                if (!primal_feasible && most_violated >= 0)
                {
                    std::vector<int> to_add;
                    for (int i = 0; i < m_; ++i)
                    {
                        if (ws_is_active_[i])
                            continue;

                        const double *GEOQP_RESTRICT a_row = A_ + i * n_;
                        double ax = LinearAlgebra::dot(a_row, ws_x_star_.data(), n_);
                        if (ax > b_[i] + config_.constraint_tolerance)
                        {
                            to_add.push_back(i);
                        }
                    }

                    bool all_added = true;
                    for (int c : to_add)
                    {
                        if (!schur_manager_.add_constraint(c, A_, Q_chol_.data(), n_, constraint_cache_))
                        {
                            all_added = false;
                            break;
                        }
                    }

                    if (!all_added)
                    {
                        // Schur complement can't handle these constraints - fall back
                        std::vector<int> fallback_face = schur_manager_.get_active_constraints();
                        for (int c : to_add)
                        {
                            fallback_face.push_back(c);
                        }
                        std::sort(fallback_face.begin(), fallback_face.end());

                        for (int iter = iterations; iter < config_.max_iterations; ++iter)
                        {

                            std::vector<double> x_proj, lambda;
                            if (!compute_projection_and_lambda(fallback_face, x_proj, lambda))
                            {
                                break;
                            }

                            int most_neg = -1;
                            double most_neg_val = -config_.constraint_tolerance;
                            for (size_t i = 0; i < fallback_face.size(); ++i)
                            {
                                if (lambda[i] < most_neg_val)
                                {
                                    most_neg_val = lambda[i];
                                    most_neg = static_cast<int>(i);
                                }
                            }

                            if (most_neg >= 0)
                            {
                                fallback_face.erase(fallback_face.begin() + most_neg);
                                iterations++;
                                continue;
                            }

                            std::vector<int> violated;
                            for (int i = 0; i < m_; ++i)
                            {
                                bool is_active = std::find(fallback_face.begin(), fallback_face.end(), i) != fallback_face.end();
                                if (is_active)
                                    continue;

                                const double *a_row = A_ + i * n_;
                                double ax = LinearAlgebra::dot(a_row, x_proj.data(), n_);
                                if (ax > b_[i] + config_.constraint_tolerance)
                                {
                                    violated.push_back(i);
                                }
                            }

                            if (violated.empty())
                            {
                                return fallback_face;
                            }

                            for (int idx : violated)
                            {
                                fallback_face.push_back(idx);
                            }
                            std::sort(fallback_face.begin(), fallback_face.end());
                            iterations++;
                        }

                        return fallback_face;
                    }
                    continue;
                }

                int drop_idx = -1;
                double min_lambda = -config_.constraint_tolerance;
                for (int i = 0; i < k; ++i)
                {
                    if (lambda_ptr[i] < min_lambda)
                    {
                        min_lambda = lambda_ptr[i];
                        drop_idx = i;
                    }
                }

                if (drop_idx >= 0)
                {
                    if (!schur_manager_.remove_constraint(drop_idx))
                    {
                        std::vector<int> fallback_face = schur_manager_.get_active_constraints();
                        fallback_face.erase(fallback_face.begin() + drop_idx);

                        for (int iter = iterations; iter < config_.max_iterations; ++iter)
                        {
                            iterations++;

                            std::vector<double> x_proj, lambda;
                            if (!compute_projection_and_lambda(fallback_face, x_proj, lambda))
                            {
                                break;
                            }

                            int most_neg = -1;
                            double most_neg_val = -config_.constraint_tolerance;
                            for (size_t i = 0; i < fallback_face.size(); ++i)
                            {
                                if (lambda[i] < most_neg_val)
                                {
                                    most_neg_val = lambda[i];
                                    most_neg = static_cast<int>(i);
                                }
                            }

                            if (most_neg >= 0)
                            {
                                fallback_face.erase(fallback_face.begin() + most_neg);
                                continue;
                            }

                            std::vector<int> violated;
                            for (int i = 0; i < m_; ++i)
                            {
                                bool is_active = std::find(fallback_face.begin(), fallback_face.end(), i) != fallback_face.end();
                                if (is_active)
                                    continue;

                                const double *a_row = A_ + i * n_;
                                double ax = LinearAlgebra::dot(a_row, x_proj.data(), n_);
                                if (ax > b_[i] + config_.constraint_tolerance)
                                {
                                    violated.push_back(i);
                                }
                            }

                            if (violated.empty())
                            {
                                return fallback_face;
                            }

                            for (int idx : violated)
                            {
                                fallback_face.push_back(idx);
                            }
                            std::sort(fallback_face.begin(), fallback_face.end());
                        }

                        return fallback_face;
                    }
                    continue;
                }

                if (config_.timing_verbosity > 0)
                {
                    timing_stats_.find_optimal_face_total += t.elapsed_ms();
                    ++timing_stats_.find_optimal_face_calls;
                    print_timing_summary(timing_stats_);
                }
                return schur_manager_.get_active_constraints();
            }

            if (config_.timing_verbosity > 0)
            {
                timing_stats_.find_optimal_face_total += t.elapsed_ms();
                ++timing_stats_.find_optimal_face_calls;
                print_timing_summary(timing_stats_);
            }
            return schur_manager_.get_active_constraints();
        }

        std::vector<int> FaceHopper::find_optimal_face(
            const std::vector<int> &initial_face,
            int &iterations)
        {
            Timer t;
            iterations = 0;

            if (initial_face.empty())
            {
                if (config_.timing_verbosity > 0)
                {
                    timing_stats_.find_optimal_face_total += t.elapsed_ms();
                    ++timing_stats_.find_optimal_face_calls;
                    print_timing_summary(timing_stats_);
                }
                return initial_face;
            }
            if (config_.use_caching && !Q_chol_.empty() && !has_diagonal_constraints_)
            {
                switch (config_.face_hopping_strategy)
                {
                case FaceHoppingStrategy::BATCH:
                    return hopping_strategy_batch(initial_face, iterations, t,
                                                  Q_, A_, b_, x_bar_, m_, n_, config_,
                                                  constraint_cache_, schur_manager_, Q_chol_.data());
                case FaceHoppingStrategy::SENSITIVITY:
                    return hopping_strategy_sensitivity(initial_face, iterations, t,
                                                        Q_, A_, b_, x_bar_, m_, n_, config_,
                                                        constraint_cache_, schur_manager_, Q_chol_.data());
                case FaceHoppingStrategy::NEIGHBORS:
                    return hopping_strategy_neighbors(initial_face, iterations, t,
                                                      Q_, A_, b_, x_bar_, m_, n_, config_,
                                                      constraint_cache_, schur_manager_, Q_chol_.data());
                case FaceHoppingStrategy::STANDARD:
                default:
                    return find_optimal_face_cached(initial_face, iterations, t);
                }
            }

            std::vector<int> current_face = initial_face;
            const int max_iter = config_.max_iterations;

            while (iterations < max_iter)
            {
                iterations++;
                ::geoqp::timing::face_hops_count.fetch_add(1, std::memory_order_relaxed);

                // Sampled timing: time 1-in-8 hops to avoid perturbation
                const bool do_sample = ((iterations & 7) == 0);
                std::chrono::high_resolution_clock::time_point samp_start;
                if (do_sample)
                    samp_start = std::chrono::high_resolution_clock::now();

                // Use preallocated workspace instead of creating new vectors
                bool feasible = compute_projection_and_lambda(current_face, ws_opt_x_star_, ws_opt_lambda_);

                if (do_sample)
                {
                    auto samp_end = std::chrono::high_resolution_clock::now();
                    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(samp_end - samp_start).count();
                    ::geoqp::timing::face_hop_ns_total.fetch_add(static_cast<uint64_t>(ns), std::memory_order_relaxed);
                    ::geoqp::timing::face_hop_sample_count.fetch_add(1, std::memory_order_relaxed);
                    ::geoqp::timing::face_hop_face_size_sum.fetch_add(static_cast<uint64_t>(current_face.size()), std::memory_order_relaxed);
                    ::geoqp::timing::face_hop_projection_calls.fetch_add(1, std::memory_order_relaxed);
                    if (!feasible)
                        ::geoqp::timing::face_hop_projection_failures.fetch_add(1, std::memory_order_relaxed);
                }

                if (feasible)
                {
                    int most_negative_idx = -1;
                    double most_negative_value = -config_.constraint_tolerance;

                    const int face_size = static_cast<int>(current_face.size());
                    const double *GEOQP_RESTRICT lambda_ptr = ws_opt_lambda_.data();

                    for (int i = 0; i < face_size; ++i)
                    {
                        if (lambda_ptr[i] < most_negative_value)
                        {
                            most_negative_value = lambda_ptr[i];
                            most_negative_idx = i;
                        }
                    }

                    if (most_negative_idx >= 0)
                    {
                        // Use workspace to build smaller face (avoid allocation)
                        ws_opt_smaller_face_.clear();
                        for (int j = 0; j < face_size; ++j)
                        {
                            if (j != most_negative_idx)
                                ws_opt_smaller_face_.push_back(current_face[j]);
                        }

                        if (config_.verbosity >= 3)
                        {
                            std::cout << "  Removing constraint with negative lambda = "
                                      << most_negative_value << "\n";
                        }

                        current_face = ws_opt_smaller_face_; // Copy from workspace
                    }
                    else
                    {
                        // Use char array as bitset instead of std::set (much faster!)
                        std::fill(ws_opt_is_active_set_.begin(), ws_opt_is_active_set_.end(), 0);
                        for (int idx : current_face)
                        {
                            ws_opt_is_active_set_[idx] = 1;
                        }

                        // Use workspace for violated list
                        ws_opt_violated_.clear();

                        const double *GEOQP_RESTRICT x_ptr = ws_opt_x_star_.data();
                        const char *GEOQP_RESTRICT is_active = ws_opt_is_active_set_.data();

                        if (has_diagonal_constraints_)
                        {
                            // O(1) per constraint via box metadata.
                            // H1/H2 filter (Proposition 2): only add violated constraint i
                            // if its normal a_i = s_i·e_j satisfies
                            //   H2: (x̄_j − v̂_j)·s_i ≥ 0   and
                            //   H1: [Q(x̄−v̂)]_j  ·s_i ≥ 0.
                            // Fallback to unfiltered scan when hint is absent or filter
                            // removes all candidates (degenerate / numerical case).
                            const int    *GEOQP_RESTRICT var_ptr = constraint_var_.data();
                            const double *GEOQP_RESTRICT sgn_ptr = constraint_sign_.data();
                            if (has_vhat_hint_)
                            {
                                const double *GEOQP_RESTRICT ng = ws_neg_grad_vhat_.data();
                                const double *GEOQP_RESTRICT dv = ws_d_vhat_.data();
                                for (int i = 0; i < m_; ++i)
                                {
                                    if (is_active[i]) continue;
                                    const int    j = var_ptr[i];
                                    const double s = sgn_ptr[i];
                                    if (s * x_ptr[j] > b_[i] + config_.constraint_tolerance &&
                                        dv[j] * s >= -config_.tolerance &&
                                        ng[j]  * s >= -config_.tolerance)
                                        ws_opt_violated_.push_back(i);
                                }
                                if (ws_opt_violated_.empty()) // fallback
                                {
                                    for (int i = 0; i < m_; ++i)
                                    {
                                        if (is_active[i]) continue;
                                        if (sgn_ptr[i] * x_ptr[var_ptr[i]] > b_[i] + config_.constraint_tolerance)
                                            ws_opt_violated_.push_back(i);
                                    }
                                }
                            }
                            else
                            {
                                for (int i = 0; i < m_; ++i)
                                {
                                    if (is_active[i]) continue;
                                    if (sgn_ptr[i] * x_ptr[var_ptr[i]] > b_[i] + config_.constraint_tolerance)
                                        ws_opt_violated_.push_back(i);
                                }
                            }
                        }
                        else
                        {
                            for (int i = 0; i < m_; ++i)
                            {
                                if (is_active[i])
                                    continue;

                                const double *GEOQP_RESTRICT a_row = A_ + i * n_;

                                // Inline dot product with unrolling
                                double ax = 0.0;
                                int j = 0;
                                for (; j + 3 < n_; j += 4)
                                {
                                    ax += a_row[j] * x_ptr[j] + a_row[j + 1] * x_ptr[j + 1] + a_row[j + 2] * x_ptr[j + 2] + a_row[j + 3] * x_ptr[j + 3];
                                }
                                for (; j < n_; ++j)
                                {
                                    ax += a_row[j] * x_ptr[j];
                                }

                                if (ax > b_[i] + config_.constraint_tolerance)
                                {
                                    ws_opt_violated_.push_back(i);
                                }
                            }
                        }

                        if (ws_opt_violated_.empty())
                        {
                            // Both dual and primal feasibility satisfied - optimal!
                            if (config_.verbosity >= 3)
                            {
                                std::cout << "  Optimal face with " << current_face.size() << " constraints\n";
                            }
                            if (config_.timing_verbosity > 0)
                            {
                                timing_stats_.find_optimal_face_total += t.elapsed_ms();
                                ++timing_stats_.find_optimal_face_calls;
                                std::cout << "FaceHopper timing summary:\n";
                                std::cout << "  compute_intersection: " << timing_stats_.compute_intersection_calls
                                          << " calls, " << timing_stats_.compute_intersection_total << " ms\n";
                                std::cout << "  find_initial_face: " << timing_stats_.find_initial_face_calls
                                          << " calls, " << timing_stats_.find_initial_face_total << " ms\n";
                                std::cout << "  compute_projection: " << timing_stats_.compute_projection_calls
                                          << " calls, " << timing_stats_.compute_projection_total << " ms\n";
                                std::cout << "  find_optimal_face: " << timing_stats_.find_optimal_face_calls
                                          << " calls, " << timing_stats_.find_optimal_face_total << " ms\n";
                            }
                            return current_face;
                        }
                        else
                        {
                            // Adaptive strategy: add all violated if safe, else add most violated
                            const int current_size = static_cast<int>(current_face.size());
                            const int violated_size = static_cast<int>(ws_opt_violated_.size());
                            const int space_left = n_ - current_size;

                            if (GEOQP_LIKELY(violated_size <= space_left))
                            {
                                // Safe to add all violated constraints without over-constraining
                                for (int c : ws_opt_violated_)
                                {
                                    current_face.push_back(c);
                                }
                            }
                            else
                            {
                                // Would over-constrain: add only the most violated
                                int most_violated = ws_opt_violated_[0];
                                double max_violation = -std::numeric_limits<double>::infinity();

                                const double *GEOQP_RESTRICT x_ptr = ws_opt_x_star_.data();

                                for (int i = 0; i < violated_size; ++i)
                                {
                                    const int c = ws_opt_violated_[i];
                                    const double *GEOQP_RESTRICT a_row = A_ + c * n_;
                                    const double bi = b_[c];

                                    // Compute a^T * x inline with loop unrolling
                                    double ax = 0.0;
                                    int j = 0;

                                    // Unroll by 4
                                    for (; j + 3 < n_; j += 4)
                                    {
                                        ax += a_row[j] * x_ptr[j] + a_row[j + 1] * x_ptr[j + 1] + a_row[j + 2] * x_ptr[j + 2] + a_row[j + 3] * x_ptr[j + 3];
                                    }

                                    // Handle remainder
                                    for (; j < n_; ++j)
                                    {
                                        ax += a_row[j] * x_ptr[j];
                                    }

                                    const double violation = ax - bi;
                                    if (violation > max_violation)
                                    {
                                        max_violation = violation;
                                        most_violated = c;
                                    }
                                }

                                current_face.push_back(most_violated);
                            }
                            std::sort(current_face.begin(), current_face.end());
                        }
                    }
                }
                else
                {
                    // Infeasible projection - need to add violated constraints
                    std::fill(ws_opt_is_active_set_.begin(), ws_opt_is_active_set_.end(), 0);
                    for (int idx : current_face)
                    {
                        ws_opt_is_active_set_[idx] = 1;
                    }

                    ws_opt_violated_.clear();
                    const double *GEOQP_RESTRICT x_ptr = ws_opt_x_star_.data();
                    const char *GEOQP_RESTRICT is_active = ws_opt_is_active_set_.data();

                    if (has_diagonal_constraints_)
                    {
                        // O(1) per constraint; H1/H2 filter applied when hint is available.
                        const int    *GEOQP_RESTRICT var_ptr = constraint_var_.data();
                        const double *GEOQP_RESTRICT sgn_ptr = constraint_sign_.data();
                        if (has_vhat_hint_)
                        {
                            const double *GEOQP_RESTRICT ng = ws_neg_grad_vhat_.data();
                            const double *GEOQP_RESTRICT dv = ws_d_vhat_.data();
                            for (int i = 0; i < m_; ++i)
                            {
                                if (is_active[i]) continue;
                                const int    j = var_ptr[i];
                                const double s = sgn_ptr[i];
                                if (s * x_ptr[j] > b_[i] + config_.constraint_tolerance &&
                                    dv[j] * s >= -config_.tolerance &&
                                    ng[j]  * s >= -config_.tolerance)
                                    ws_opt_violated_.push_back(i);
                            }
                            if (ws_opt_violated_.empty()) // fallback
                            {
                                for (int i = 0; i < m_; ++i)
                                {
                                    if (is_active[i]) continue;
                                    if (sgn_ptr[i] * x_ptr[var_ptr[i]] > b_[i] + config_.constraint_tolerance)
                                        ws_opt_violated_.push_back(i);
                                }
                            }
                        }
                        else
                        {
                            for (int i = 0; i < m_; ++i)
                            {
                                if (is_active[i]) continue;
                                if (sgn_ptr[i] * x_ptr[var_ptr[i]] > b_[i] + config_.constraint_tolerance)
                                    ws_opt_violated_.push_back(i);
                            }
                        }
                    }
                    else
                    {
                        for (int i = 0; i < m_; ++i)
                        {
                            if (is_active[i])
                                continue;

                            const double *GEOQP_RESTRICT a_row = A_ + i * n_;

                            // Inline dot product
                            double ax = 0.0;
                            int j = 0;
                            for (; j + 3 < n_; j += 4)
                            {
                                ax += a_row[j] * x_ptr[j] + a_row[j + 1] * x_ptr[j + 1] + a_row[j + 2] * x_ptr[j + 2] + a_row[j + 3] * x_ptr[j + 3];
                            }
                            for (; j < n_; ++j)
                            {
                                ax += a_row[j] * x_ptr[j];
                            }

                            if (ax > b_[i] + config_.constraint_tolerance)
                            {
                                ws_opt_violated_.push_back(i);
                            }
                        }
                    }

                    if (ws_opt_violated_.empty())
                    {
                        if (config_.verbosity >= 2)
                        {
                            std::cout << "  Warning: no violated constraint but projection infeasible\n";
                        }
                        if (config_.timing_verbosity > 0)
                        {
                            timing_stats_.find_optimal_face_total += t.elapsed_ms();
                            ++timing_stats_.find_optimal_face_calls;
                            print_timing_summary(timing_stats_);
                        }
                        return current_face;
                    }

                    // Adaptive strategy: add all violated if safe, else add most violated
                    const int current_size = static_cast<int>(current_face.size());
                    const int violated_size = static_cast<int>(ws_opt_violated_.size());
                    const int space_left = n_ - current_size;

                    if (GEOQP_LIKELY(violated_size <= space_left))
                    {
                        // Safe to add all violated constraints without over-constraining
                        for (int c : ws_opt_violated_)
                        {
                            current_face.push_back(c);
                        }
                    }
                    else
                    {
                        // Would over-constrain: add only the most violated
                        int most_violated = ws_opt_violated_[0];
                        double max_violation = -std::numeric_limits<double>::infinity();

                        for (int i = 0; i < violated_size; ++i)
                        {
                            const int c = ws_opt_violated_[i];
                            const double *GEOQP_RESTRICT a_row = A_ + c * n_;
                            const double bi = b_[c];

                            // Compute a^T * x inline with loop unrolling
                            double ax = 0.0;
                            int j = 0;

                            // Unroll by 4
                            for (; j + 3 < n_; j += 4)
                            {
                                ax += a_row[j] * x_ptr[j] + a_row[j + 1] * x_ptr[j + 1] + a_row[j + 2] * x_ptr[j + 2] + a_row[j + 3] * x_ptr[j + 3];
                            }

                            // Handle remainder
                            for (; j < n_; ++j)
                            {
                                ax += a_row[j] * x_ptr[j];
                            }

                            const double violation = ax - bi;
                            if (violation > max_violation)
                            {
                                max_violation = violation;
                                most_violated = c;
                            }
                        }

                        current_face.push_back(most_violated);
                    }
                    std::sort(current_face.begin(), current_face.end());
                }
            }

            if (config_.verbosity >= 1)
            {
                std::cout << "Warning: Face exploration reached max iterations\n";
            }
            if (config_.timing_verbosity > 0)
            {
                timing_stats_.find_optimal_face_total += t.elapsed_ms();
                ++timing_stats_.find_optimal_face_calls;
                print_timing_summary(timing_stats_);
            }

            return current_face;
        }

    } // namespace internal
} // namespace geoqp
