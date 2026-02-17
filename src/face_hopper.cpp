#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include <geoqp/timing.hpp>
#include <cmath>
#include <limits>
#include <algorithm>
#include <set>
#include <cstring>
#include <iostream>
#include <chrono>

namespace geoqp
{
    namespace internal
    {

        FaceHopper::FaceHopper(
            const double *Q, const double *A, const double *b,
            const std::vector<double> &x_bar,
            int m, int n,
            const Config &config) : Q_(Q), A_(A), b_(b), x_bar_(x_bar), m_(m), n_(n), config_(config),
                                    constraint_cache_(m, n)
        {
            Q_chol_.resize(n_ * n_);
            try
            {
                LinearAlgebra::cholesky(Q_, Q_chol_.data(), n_);
            }
            catch (...)
            {
                Q_chol_.clear();
            }

            // Optimized density check for Q
            int nonzero_q = 0;
            const int total_elements = n_ * n_;
            const double * GEOQP_RESTRICT Q_ptr = Q_;
            const double tol = 1e-12;

            int i = 0;
            GEOQP_UNROLL_LOOP_4
            for (; i + 3 < total_elements; i += 4) {
                nonzero_q += (std::abs(Q_ptr[i]) > tol);
                nonzero_q += (std::abs(Q_ptr[i + 1]) > tol);
                nonzero_q += (std::abs(Q_ptr[i + 2]) > tol);
                nonzero_q += (std::abs(Q_ptr[i + 3]) > tol);
            }
            for (; i < total_elements; ++i) {
                nonzero_q += (std::abs(Q_ptr[i]) > tol);
            }
            q_is_dense_ = (nonzero_q > (total_elements / 2));

            // Detect diagonal Q
            q_is_diagonal_ = true;
            Q_diag_.resize(n_);
            for (int i = 0; i < n_; ++i) {
                Q_diag_[i] = Q_[i * n_ + i];
                for (int j = 0; j < n_; ++j) {
                    if (i != j && std::abs(Q_[i * n_ + j]) > 1e-12) {
                        q_is_diagonal_ = false;
                        break;
                    }
                }
                if (!q_is_diagonal_) break;
            }

            // Precompute Q_inv if Q is dense and n is manageable
            if (q_is_dense_ && n_ <= 256 && !Q_chol_.empty()) {
                Q_inv_.resize(n_ * n_);
                for (int i = 0; i < n_; ++i) {
                    std::vector<double> e(n_, 0.0);
                    e[i] = 1.0;
                    LinearAlgebra::solve_cholesky(Q_chol_.data(), e.data(), &Q_inv_[i * n_], n_);
                }
                if (config_.verbosity >= 2) {
                    std::cout << "  Q is dense - precomputed Q_inv for fast access\n";
                }
            }

            ws_residual_.resize(m);
            ws_w_.resize(n);
            ws_correction_.resize(n);
            ws_x_star_.resize(n);
            ws_lambda_.resize(m);

            ws_Axbar_.resize(m_);
            LinearAlgebra::matvec(A_, x_bar_.data(), ws_Axbar_.data(), m_, n_);

            if (static_cast<int>(ws_is_active_.size()) < m_) ws_is_active_.assign(m_, 0);
            
            // Preallocate workspace for find_optimal_face hot loop
            ws_opt_x_star_.resize(n);
            ws_opt_lambda_.resize(m);
            ws_opt_violated_.reserve(m);  // Reserve max possible
            ws_opt_smaller_face_.reserve(m);  // Reserve max possible
            ws_opt_is_active_set_.resize(m, 0);  // Bitset for O(1) membership test

            has_diagonal_constraints_ = true;
            for (int i = 0; i < m_ && has_diagonal_constraints_; ++i)
            {
                const double *a_row = A_ + i * n_;
                int nonzero_count = 0;
                int nonzero_idx = -1;
                for (int j = 0; j < n_; ++j)
                {
                    if (std::abs(a_row[j]) > 1e-10)
                    {
                        nonzero_count++;
                        nonzero_idx = j;
                        if (std::abs(std::abs(a_row[j]) - 1.0) > 1e-10)
                        {
                            has_diagonal_constraints_ = false;
                            break;
                        }
                    }
                }
                if (nonzero_count != 1)
                {
                    has_diagonal_constraints_ = false;
                }
            }

            if (has_diagonal_constraints_ && config_.verbosity >= 2)
            {
                std::cout << "  Detected diagonal/box constraints - using fast path\n";
            }

            if (config_.use_caching && !Q_chol_.empty() && config_.verbosity >= 2)
            {
                std::cout << "  Incremental Schur complement caching enabled\n";
            }
        }

        double FaceHopper::compute_intersection_parameter(
            const std::vector<double> &v,
            int constraint_idx) const
        {
            Timer t;
            const double *GEOQP_RESTRICT a_row = A_ + constraint_idx * n_;
            const double *GEOQP_RESTRICT v_ptr = v.data();
            const double *GEOQP_RESTRICT xbar_ptr = x_bar_.data();

            double av0 = 0.0, av1 = 0.0, av2 = 0.0, av3 = 0.0;
            double ad0 = 0.0, ad1 = 0.0, ad2 = 0.0, ad3 = 0.0;

            int j = 0;
            for (; j + 3 < n_; j += 4)
            {
                av0 += a_row[j] * v_ptr[j];
                av1 += a_row[j + 1] * v_ptr[j + 1];
                av2 += a_row[j + 2] * v_ptr[j + 2];
                av3 += a_row[j + 3] * v_ptr[j + 3];

                ad0 += a_row[j] * (xbar_ptr[j] - v_ptr[j]);
                ad1 += a_row[j + 1] * (xbar_ptr[j + 1] - v_ptr[j + 1]);
                ad2 += a_row[j + 2] * (xbar_ptr[j + 2] - v_ptr[j + 2]);
                ad3 += a_row[j + 3] * (xbar_ptr[j + 3] - v_ptr[j + 3]);
            }

            double av = (av0 + av1) + (av2 + av3);
            double ad = (ad0 + ad1) + (ad2 + ad3);

            for (; j < n_; ++j)
            {
                av += a_row[j] * v_ptr[j];
                ad += a_row[j] * (xbar_ptr[j] - v_ptr[j]);
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
            Timer t;
            std::vector<int> active_at_v;
            for (int i = 0; i < m_; ++i)
            {
                const double *a_row = A_ + i * n_;
                double av = LinearAlgebra::dot(a_row, v.data(), n_);
                if (std::abs(av - b_[i]) < 1e-8)
                {
                    active_at_v.push_back(i);
                }
            }

            std::vector<int> blocking_at_v;
            for (int i : active_at_v)
            {
                const double *a_row = A_ + i * n_;
                double a_xbar = LinearAlgebra::dot(a_row, x_bar_.data(), n_);
                if (a_xbar > b_[i] + 1e-8)
                {
                    blocking_at_v.push_back(i);
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

            // Prefer the largest feasible intersection parameter t in (0,1).
            // If none exist, fall back to the previous smallest-positive-t behavior.
            double t_max_in01 = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < m_; ++i)
            {
                double t = compute_intersection_parameter(v, i);
                if (t > 1e-10 && t < 1.0 - 1e-10)
                {
                    if (t > t_max_in01) t_max_in01 = t;
                }
            }

            const double tol = 1e-8;
            std::vector<int> initial_face;

            if (std::isfinite(t_max_in01))
            {
                for (int i = 0; i < m_; ++i)
                {
                    double t = compute_intersection_parameter(v, i);
                    if (std::abs(t - t_max_in01) <= tol)
                    {
                        initial_face.push_back(i);
                    }
                }
                if (std::getenv("GEOQP_DBG_INIT_PATH") && std::string(std::getenv("GEOQP_DBG_INIT_PATH")) == "1") {
                    std::fprintf(stderr, "[DBG] FaceHopper: picked t_max_in01=%.12g with %zu constraints\n", t_max_in01, initial_face.size());
                }
            }
            else
            {
                // fallback: smallest positive t (legacy behavior)
                double t_min = std::numeric_limits<double>::infinity();
                for (int i = 0; i < m_; ++i)
                {
                    double t = compute_intersection_parameter(v, i);
                    if (t > 1e-10 && t < t_min)
                    {
                        t_min = t;
                    }
                }

                if (t_min > 1.0 + 1e-8)
                {
                    bool x_bar_feasible = true;
                    for (int i = 0; i < m_; ++i)
                    {
                        const double *a_row = A_ + i * n_;
                        double ax = LinearAlgebra::dot(a_row, x_bar_.data(), n_);
                        if (ax > b_[i] + config_.tolerance)
                        {
                            x_bar_feasible = false;
                            break;
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
                    double t = compute_intersection_parameter(v, i);
                    if (std::abs(t - t_min) <= tol)
                    {
                        initial_face.push_back(i);
                    }
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

            if (has_diagonal_constraints_)
            {

                std::vector<int> constrained_vars(k);
                std::vector<double> signs(k);
                std::vector<double> bounds(k);

                for (int i = 0; i < k; ++i)
                {
                    const int idx = face[i];
                    const double *a_row = A_ + idx * n_;
                    bounds[i] = b_[idx];
                    for (int j = 0; j < n_; ++j)
                    {
                        if (std::abs(a_row[j]) > 1e-10)
                        {
                            constrained_vars[i] = j;
                            signs[i] = a_row[j];
                            break;
                        }
                    }
                }

                std::vector<double> residual(k);
                for (int i = 0; i < k; ++i)
                {
                    residual[i] = signs[i] * x_bar_[constrained_vars[i]] - bounds[i];
                }

                // Use workspace to avoid allocations
                if (static_cast<int>(ws_constrained_vars_.size()) < k) {
                    ws_constrained_vars_.resize(k);
                    ws_signs_.resize(k);
                    ws_residual_.resize(k);
                }

                int* GEOQP_RESTRICT c_vars = ws_constrained_vars_.data();
                double* GEOQP_RESTRICT c_signs = ws_signs_.data();
                double* GEOQP_RESTRICT c_residual = ws_residual_.data();

                for (int i = 0; i < k; ++i)
                {
                    const int idx = face[i];
                    c_residual[i] = ws_Axbar_[idx] - b_[idx]; 
                    
                    const double * GEOQP_RESTRICT a_row = A_ + idx * n_;
                    for (int j = 0; j < n_; ++j)
                    {
                        if (std::abs(a_row[j]) > 1e-10)
                        {
                            c_vars[i] = j;
                            c_signs[i] = a_row[j];
                            break;
                        }
                    }
                }

                // Use a pre-allocated workspace for Q_inv_columns instead of std::vector
                if (static_cast<int>(ws_Q_inv_cols_.size()) < n_ * k) {
                    ws_Q_inv_cols_.resize(n_ * k);
                }
                double* GEOQP_RESTRICT Q_inv_columns = ws_Q_inv_cols_.data();

                for (int j = 0; j < k; ++j)
                {
                    double* GEOQP_RESTRICT target_col = &Q_inv_columns[j * n_];
                    const double s = c_signs[j];
                    const int v = c_vars[j];

                    if (q_is_diagonal_) {
                        std::fill(target_col, target_col + n_, 0.0);
                        target_col[v] = s / Q_diag_[v];
                    } else if (!Q_inv_.empty()) {
                        const double* GEOQP_RESTRICT Qi = &Q_inv_[v * n_];
                        int i = 0;
                        GEOQP_UNROLL_LOOP_4
                        for (; i + 3 < n_; i += 4) {
                            target_col[i] = Qi[i] * s;
                            target_col[i + 1] = Qi[i+1] * s;
                            target_col[i + 2] = Qi[i+2] * s;
                            target_col[i + 3] = Qi[i+3] * s;
                        }
                        for (; i < n_; ++i) target_col[i] = Qi[i] * s;
                    } else if (!Q_chol_.empty()) {
                        std::vector<double> rhs(n_, 0.0);
                        rhs[v] = s;
                        LinearAlgebra::solve_cholesky(Q_chol_.data(), rhs.data(), target_col, n_);
                    } else {
                        return false;
                    }
                }

                if (static_cast<int>(ws_S_.size()) < k * k) ws_S_.resize(k * k);
                double* GEOQP_RESTRICT S = ws_S_.data();

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
                double* GEOQP_RESTRICT x_ptr = x_star.data();
                const double* GEOQP_RESTRICT lambda_ptr = lambda.data();
                for (int j = 0; j < k; ++j)
                {
                    const double l_j = lambda_ptr[j];
                    const double* GEOQP_RESTRICT col = &Q_inv_columns[j * n_];
                    int i = 0;
                    GEOQP_UNROLL_LOOP_4
                    for (; i + 3 < n_; i += 4) {
                        x_ptr[i] -= col[i] * l_j;
                        x_ptr[i+1] -= col[i+1] * l_j;
                        x_ptr[i+2] -= col[i+2] * l_j;
                        x_ptr[i+3] -= col[i+3] * l_j;
                    }
                    for (; i < n_; ++i) x_ptr[i] -= col[i] * l_j;
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
                        if (is_active) continue;
                        
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

                if (static_cast<int>(ws_is_active_.size()) < m_) ws_is_active_.assign(m_, 0);
                else std::fill(ws_is_active_.begin(), ws_is_active_.end(), 0);
                for (int idx : current_face) ws_is_active_[idx] = 1;

                for (int i = 0; i < m_; ++i)
                {
                    if (ws_is_active_[i]) continue;
                    
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
                                if (is_active) continue;
                                
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
                                if (is_active) continue;
                                
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
                return schur_manager_.get_active_constraints();
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
                if (do_sample) samp_start = std::chrono::high_resolution_clock::now();

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
                    if (!feasible) ::geoqp::timing::face_hop_projection_failures.fetch_add(1, std::memory_order_relaxed);
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
                        
                        current_face = ws_opt_smaller_face_;  // Copy from workspace
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
                                ax += a_row[j] * x_ptr[j]
                                    + a_row[j+1] * x_ptr[j+1]
                                    + a_row[j+2] * x_ptr[j+2]
                                    + a_row[j+3] * x_ptr[j+3];
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
                                        ax += a_row[j] * x_ptr[j]
                                            + a_row[j+1] * x_ptr[j+1]
                                            + a_row[j+2] * x_ptr[j+2]
                                            + a_row[j+3] * x_ptr[j+3];
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
                            ax += a_row[j] * x_ptr[j]
                                + a_row[j+1] * x_ptr[j+1]
                                + a_row[j+2] * x_ptr[j+2]
                                + a_row[j+3] * x_ptr[j+3];
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
                                ax += a_row[j] * x_ptr[j]
                                    + a_row[j+1] * x_ptr[j+1]
                                    + a_row[j+2] * x_ptr[j+2]
                                    + a_row[j+3] * x_ptr[j+3];
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

    } // namespace internal
} // namespace geoqp
