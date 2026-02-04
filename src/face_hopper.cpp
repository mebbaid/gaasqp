#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
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

            ws_residual_.resize(m);
            ws_w_.resize(n);
            ws_correction_.resize(n);
            ws_x_star_.resize(n);
            ws_lambda_.resize(m);

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

            std::vector<int> initial_face;
            initial_face.reserve(n_);
            const double tol = 1e-8;
            for (int i = 0; i < m_; ++i)
            {
                double t = compute_intersection_parameter(v, i);
                if (std::abs(t - t_min) <= tol)
                {
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
                        if (ax > b_[i] + config_.tolerance)
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
                    if (ax > b_[i] + config_.tolerance)
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

                std::vector<double> Q_inv_columns(n_ * k);
                for (int j = 0; j < k; ++j)
                {
                    std::vector<double> rhs(n_, 0.0);
                    rhs[constrained_vars[j]] = signs[j];

                    std::vector<double> col(n_);
                    if (!Q_chol_.empty())
                    {
                        LinearAlgebra::solve_cholesky(Q_chol_.data(), rhs.data(), col.data(), n_);
                    }
                    else
                    {
                        std::vector<double> Q_copy(n_ * n_);
                        std::memcpy(Q_copy.data(), Q_, n_ * n_ * sizeof(double));
                        try
                        {
                            LinearAlgebra::solve_linear_system(Q_copy.data(), rhs.data(), col.data(), n_);
                        }
                        catch (...)
                        {
                            if (config_.timing_verbosity > 0)
                            {
                                timing_stats_.compute_projection_total += t.elapsed_ms();
                                ++timing_stats_.compute_projection_calls;
                            }
                            return false;
                        }
                    }
                    std::memcpy(Q_inv_columns.data() + j * n_, col.data(), n_ * sizeof(double));
                }

                std::vector<double> S(k * k);
                for (int i = 0; i < k; ++i)
                {
                    for (int j = 0; j < k; ++j)
                    {
                        S[i * k + j] = Q_inv_columns[j * n_ + constrained_vars[i]] * signs[i];
                    }
                }

                try
                {
                    LinearAlgebra::solve_linear_system(S.data(), residual.data(), lambda.data(), k);
                }
                catch (...)
                {
                    if (config_.timing_verbosity > 0)
                    {
                        timing_stats_.compute_projection_total += t.elapsed_ms();
                        ++timing_stats_.compute_projection_calls;
                    }
                    return false;
                }

                x_star = x_bar_;
                for (int i = 0; i < n_; ++i)
                {
                    double correction = 0.0;
                    for (int j = 0; j < k; ++j)
                    {
                        correction += Q_inv_columns[j * n_ + i] * lambda[j];
                    }
                    x_star[i] -= correction;
                }

                for (int i = 0; i < m_; ++i)
                {
                    const double *a_row = A_ + i * n_;
                    double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
                    if (ax > b_[i] + config_.tolerance)
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

            bool success = ProjectionSolver::solve(
                face, Q_, A_, b_, x_bar_, Q_chol_,
                m_, n_, x_star, lambda);

            if (!success)
            {
                if (config_.timing_verbosity > 0)
                {
                    timing_stats_.compute_projection_total += t.elapsed_ms();
                    ++timing_stats_.compute_projection_calls;
                }
                return false;
            }

            for (int i = 0; i < m_; ++i)
            {
                const double *a_row = A_ + i * n_;
                double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
                if (ax > b_[i] + config_.tolerance)
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

            if (static_cast<int>(ws_residual_.size()) < k)
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
                return initial_face;
            }

            while (iterations < max_iter)
            {
                iterations++;
                const auto &current_face = schur_manager_.get_active_constraints();
                const int k = schur_manager_.size();

                if (static_cast<int>(ws_residual_.size()) < k)
                    ws_residual_.resize(k);
                if (static_cast<int>(ws_w_.size()) < n_)
                    ws_w_.resize(n_);
                if (static_cast<int>(ws_correction_.size()) < n_)
                    ws_correction_.resize(n_);
                if (static_cast<int>(ws_x_star_.size()) < n_)
                    ws_x_star_.resize(n_);
                if (static_cast<int>(ws_lambda_.size()) < k)
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

                std::set<int> active_set(current_face.begin(), current_face.end());
                for (int i = 0; i < m_; ++i)
                {
                    if (active_set.count(i))
                        continue;

                    const double *GEOQP_RESTRICT a_row = A_ + i * n_;
                    double ax = LinearAlgebra::dot(a_row, xstar_ptr, n_);
                    double violation = ax - b_[i];

                    if (violation > max_violation + config_.tolerance)
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
                        if (active_set.count(i))
                            continue;

                        const double *GEOQP_RESTRICT a_row = A_ + i * n_;
                        double ax = LinearAlgebra::dot(a_row, xstar_ptr, n_);
                        if (ax > b_[i] + config_.tolerance)
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
                        break;
                    continue;
                }

                int drop_idx = -1;
                double min_lambda = -config_.tolerance;
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
                        break;
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

                std::vector<double> x_star, lambda;
                bool feasible = compute_projection_and_lambda(current_face, x_star, lambda);

                if (feasible)
                {
                    bool improved = false;

                    for (size_t i = 0; i < current_face.size(); ++i)
                    {
                        if (lambda[i] < -config_.tolerance)
                        {
                            std::vector<int> smaller_face;
                            for (size_t j = 0; j < current_face.size(); ++j)
                            {
                                if (i != j)
                                    smaller_face.push_back(current_face[j]);
                            }

                            std::vector<double> x2, lambda2;
                            if (compute_projection_and_lambda(smaller_face, x2, lambda2))
                            {
                                bool dual_feasible = true;
                                for (double l : lambda2)
                                {
                                    if (l < -config_.tolerance)
                                    {
                                        dual_feasible = false;
                                        break;
                                    }
                                }

                                if (!dual_feasible)
                                {
                                    continue;
                                }

                                double d1 = 0.0, d2 = 0.0;
                                bool early_exit = false;

                                std::vector<double> diff1(n_), diff2(n_);
                                for (int j = 0; j < n_; ++j)
                                {
                                    diff1[j] = x_star[j] - x_bar_[j];
                                    diff2[j] = x2[j] - x_bar_[j];
                                }

                                for (int j = 0; j < n_ && !early_exit; ++j)
                                {
                                    const double *Q_row = Q_ + j * n_;
                                    double row_contrib1 = 0.0, row_contrib2 = 0.0;
                                    for (int k = 0; k < n_; ++k)
                                    {
                                        row_contrib1 += Q_row[k] * diff1[k];
                                        row_contrib2 += Q_row[k] * diff2[k];
                                    }
                                    d1 += diff1[j] * row_contrib1;
                                    d2 += diff2[j] * row_contrib2;

                                    if (j % 10 == 9 && d2 > d1 * 1.1)
                                    {
                                        early_exit = true;
                                    }
                                }

                                if (!early_exit && d2 < d1 - config_.tolerance)
                                {
                                    current_face = std::move(smaller_face);
                                    improved = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (!improved)
                    {
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
                }
                else
                {
                    std::set<int> face_set(current_face.begin(), current_face.end());
                    std::vector<int> violated;

                    for (int i = 0; i < m_; ++i)
                    {
                        if (face_set.find(i) != face_set.end())
                            continue;

                        const double *a_row = A_ + i * n_;
                        double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
                        if (ax > b_[i] + config_.tolerance)
                        {
                            violated.push_back(i);
                        }
                    }

                    if (violated.empty())
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

                    for (int c : violated)
                    {
                        current_face.push_back(c);
                    }
                    std::sort(current_face.begin(), current_face.end());

                    if (config_.verbosity >= 3)
                    {
                        std::cout << "  Added " << violated.size() << " violated constraints\n";
                    }
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
