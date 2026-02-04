#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>

namespace geoqp
{
    namespace internal
    {

        Initializer::Initializer(
            const double *Q, const double *A, const double *b,
            const std::vector<double> &x_bar,
            int m, int n,
            const Config &config) : Q_(Q), A_(A), b_(b), x_bar_(x_bar), m_(m), n_(n), config_(config)
        {
            detect_box_constraints();

            ws_direction_.resize(n);
            ws_Ax_.resize(m);
            ws_Ad_.resize(m);
            ws_candidate_.resize(n);
        }

        void Initializer::detect_box_constraints()
        {
            is_box_constrained_ = false;
            box_lb_.clear();
            box_ub_.clear();

            if (m_ != 2 * n_)
            {
                return;
            }

            box_lb_.resize(n_, -std::numeric_limits<double>::infinity());
            box_ub_.resize(n_, std::numeric_limits<double>::infinity());

            for (int i = 0; i < m_; ++i)
            {
                const double *a_row = A_ + i * n_;
                int nonzero_idx = -1;
                double nonzero_val = 0.0;
                bool is_single_nonzero = true;

                for (int j = 0; j < n_; ++j)
                {
                    if (std::abs(a_row[j]) > 1e-10)
                    {
                        if (nonzero_idx >= 0)
                        {
                            is_single_nonzero = false;
                            break;
                        }
                        nonzero_idx = j;
                        nonzero_val = a_row[j];
                    }
                }

                if (!is_single_nonzero || nonzero_idx < 0)
                {
                    box_lb_.clear();
                    box_ub_.clear();
                    return;
                }

                if (std::abs(nonzero_val - 1.0) < 1e-10)
                {
                    box_ub_[nonzero_idx] = std::min(box_ub_[nonzero_idx], b_[i]);
                }
                else if (std::abs(nonzero_val + 1.0) < 1e-10)
                {
                    box_lb_[nonzero_idx] = std::max(box_lb_[nonzero_idx], -b_[i]);
                }
                else
                {
                    box_lb_.clear();
                    box_ub_.clear();
                    return;
                }
            }

            for (int j = 0; j < n_; ++j)
            {
                if (!std::isfinite(box_lb_[j]) || !std::isfinite(box_ub_[j]))
                {
                    box_lb_.clear();
                    box_ub_.clear();
                    return;
                }
            }

            is_box_constrained_ = true;
        }

        std::vector<double> Initializer::find_good_vertex_box_fast() const
        {

            std::vector<double> vertex(n_);

            for (int j = 0; j < n_; ++j)
            {
                if (x_bar_[j] > box_ub_[j])
                {
                    vertex[j] = box_lb_[j];
                }
                else if (x_bar_[j] < box_lb_[j])
                {
                    vertex[j] = box_ub_[j];
                }
                else
                {
                    double dist_to_lb = x_bar_[j] - box_lb_[j];
                    double dist_to_ub = box_ub_[j] - x_bar_[j];
                    vertex[j] = (dist_to_lb > dist_to_ub) ? box_lb_[j] : box_ub_[j];
                }
            }

            return vertex;
        }

        double Initializer::compute_q_weighted_distance_squared(
            const std::vector<double> &v) const
        {
            std::vector<double> diff(n_);
            for (int i = 0; i < n_; ++i)
            {
                diff[i] = v[i] - x_bar_[i];
            }
            std::vector<double> Q_diff(n_);
            LinearAlgebra::matvec(Q_, diff.data(), Q_diff.data(), n_, n_);
            return LinearAlgebra::dot(diff.data(), Q_diff.data(), n_);
        }

        double Initializer::compute_euclidean_distance_squared(
            const std::vector<double> &v) const
        {
            const double *GEOQP_RESTRICT v_ptr = v.data();
            const double *GEOQP_RESTRICT xbar_ptr = x_bar_.data();

            double sum0 = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            double sum3 = 0.0;

            int i = 0;
            for (; i + 3 < n_; i += 4)
            {
                double d0 = v_ptr[i] - xbar_ptr[i];
                double d1 = v_ptr[i + 1] - xbar_ptr[i + 1];
                double d2 = v_ptr[i + 2] - xbar_ptr[i + 2];
                double d3 = v_ptr[i + 3] - xbar_ptr[i + 3];

                sum0 += d0 * d0;
                sum1 += d1 * d1;
                sum2 += d2 * d2;
                sum3 += d3 * d3;
            }

            double dist_sq = (sum0 + sum1) + (sum2 + sum3);

            for (; i < n_; ++i)
            {
                double d = v_ptr[i] - xbar_ptr[i];
                dist_sq += d * d;
            }

            return dist_sq;
        }

        bool Initializer::is_good_vertex(
            const std::vector<double> &v,
            const std::vector<int> &active_set) const
        {
            const double *GEOQP_RESTRICT v_ptr = v.data();
            const double *GEOQP_RESTRICT xbar_ptr = x_bar_.data();

            for (int idx : active_set)
            {
                const double *GEOQP_RESTRICT a_row = A_ + idx * n_;

                double sum0 = 0.0;
                double sum1 = 0.0;
                double sum2 = 0.0;
                double sum3 = 0.0;

                int j = 0;
                for (; j + 3 < n_; j += 4)
                {
                    sum0 += a_row[j] * (xbar_ptr[j] - v_ptr[j]);
                    sum1 += a_row[j + 1] * (xbar_ptr[j + 1] - v_ptr[j + 1]);
                    sum2 += a_row[j + 2] * (xbar_ptr[j + 2] - v_ptr[j + 2]);
                    sum3 += a_row[j + 3] * (xbar_ptr[j + 3] - v_ptr[j + 3]);
                }

                double dot = (sum0 + sum1) + (sum2 + sum3);

                for (; j < n_; ++j)
                {
                    dot += a_row[j] * (xbar_ptr[j] - v_ptr[j]);
                }

                if (dot > config_.tolerance)
                {
                    return false;
                }
            }
            return true;
        }

        std::vector<double> Initializer::compute_vertex_from_active_set(
            const std::vector<int> &active_indices) const
        {
            if (static_cast<int>(active_indices.size()) != n_)
            {
                return std::vector<double>();
            }

            std::vector<double> A_active(n_ * n_);
            std::vector<double> b_active(n_);

            for (int i = 0; i < n_; ++i)
            {
                int idx = active_indices[i];
                b_active[i] = b_[idx];
                std::memcpy(A_active.data() + i * n_, A_ + idx * n_, n_ * sizeof(double));
            }

            std::vector<double> vertex(n_);
            try
            {
                LinearAlgebra::solve_linear_system(A_active.data(), b_active.data(), vertex.data(), n_);
            }
            catch (...)
            {
                return std::vector<double>();
            }

            for (int i = 0; i < m_; ++i)
            {
                const double *a_row = A_ + i * n_;
                double ax = LinearAlgebra::dot(a_row, vertex.data(), n_);
                if (ax > b_[i] + 1e-6)
                {
                    return std::vector<double>();
                }
            }

            return vertex;
        }

        std::vector<double> Initializer::find_good_vertex(
            const std::vector<double> &initial_vertex,
            int &iterations,
            int &vertices_explored)
        {
            iterations = 0;
            vertices_explored = 0;

            // Fast path for box constraints
            if (is_box_constrained_)
            {
                vertices_explored = 1;
                iterations = 1;
                return find_good_vertex_box_fast();
            }

            // Geometric approach: Find the "farthest good vertex"
            // Starting from initial_vertex (feasible), shoot a ray away from x_bar
            // to find the antipodal vertex (farthest feasible point in that direction)

            Polytope polytope(A_, b_, m_, n_);
            std::vector<int> active_set = polytope.get_active_constraints(initial_vertex, 1e-8);

            iterations = 1;
            vertices_explored = 1;

            if (is_good_vertex(initial_vertex, active_set))
            {
                return initial_vertex;
            }

            const double *GEOQP_RESTRICT init_ptr = initial_vertex.data();
            const double *GEOQP_RESTRICT xbar_ptr = x_bar_.data();
            double *GEOQP_RESTRICT dir_ptr = ws_direction_.data();

            for (int j = 0; j < n_; ++j)
            {
                dir_ptr[j] = init_ptr[j] - xbar_ptr[j];
            }

            double norm = LinearAlgebra::norm(dir_ptr, n_);
            if (norm > 1e-10)
            {
                for (int j = 0; j < n_; ++j)
                {
                    dir_ptr[j] /= norm;
                }
            }
            else
            {
                for (int j = 0; j < n_; ++j)
                {
                    dir_ptr[j] = 1.0 / std::sqrt(static_cast<double>(n_));
                }
            }

            LinearAlgebra::matvec(A_, init_ptr, ws_Ax_.data(), m_, n_);

            LinearAlgebra::matvec(A_, dir_ptr, ws_Ad_.data(), m_, n_);

            double min_blocking_t = std::numeric_limits<double>::infinity();
            int blocking_constraint = -1;

            const double *GEOQP_RESTRICT Ax_ptr = ws_Ax_.data();
            const double *GEOQP_RESTRICT Ad_ptr = ws_Ad_.data();

            for (int i = 0; i < m_; ++i)
            {
                if (Ad_ptr[i] > config_.tolerance)
                {
                    double val = b_[i] - Ax_ptr[i];

                    if (val < 0.0)
                        val = 0.0;

                    double t = val / Ad_ptr[i];

                    if (t < min_blocking_t)
                    {
                        min_blocking_t = t;
                        blocking_constraint = i;
                    }
                }
            }

            if (!std::isfinite(min_blocking_t) || min_blocking_t < config_.tolerance)
            {
                return initial_vertex;
            }

            double *GEOQP_RESTRICT cand_ptr = ws_candidate_.data();

            for (int j = 0; j < n_; ++j)
            {
                cand_ptr[j] = init_ptr[j] + min_blocking_t * dir_ptr[j];
            }

            for (int i = 0; i < m_; ++i)
            {
                const double *GEOQP_RESTRICT a_row = A_ + i * n_;
                double ax = LinearAlgebra::dot(a_row, cand_ptr, n_);
                if (ax > b_[i] + config_.tolerance)
                {
                    return initial_vertex;
                }
            }

            std::vector<double> candidate(ws_candidate_.begin(), ws_candidate_.begin() + n_);

            std::vector<int> candidate_active = polytope.get_active_constraints(candidate, 1e-8);

            if (is_good_vertex(candidate, candidate_active))
            {
                vertices_explored = 2;
                return candidate;
            }

            return initial_vertex;
        }

        bool Initializer::is_direction_feasible(
            const std::vector<double> &v,
            const std::vector<double> &d,
            const std::vector<int> &active_set,
            double tol) const
        {
            for (int idx : active_set)
            {
                const double *a_row = A_ + idx * n_;
                double dot = LinearAlgebra::dot(a_row, d.data(), n_);
                if (dot > tol)
                    return false;
            }
            return true;
        }

        int Initializer::find_most_violated_constraint(
            const std::vector<double> &v,
            const std::vector<double> &d,
            const std::vector<int> &active_set) const
        {
            int most_violated = -1;
            double max_violation = 0.0;
            for (int idx : active_set)
            {
                const double *a_row = A_ + idx * n_;
                double dot = LinearAlgebra::dot(a_row, d.data(), n_);
                if (dot > max_violation)
                {
                    max_violation = dot;
                    most_violated = idx;
                }
            }
            return most_violated;
        }

        std::vector<double> Initializer::pivot_to_neighbor(
            const std::vector<double> &v,
            int constraint_to_drop)
        {
            return v;
        }

    } // namespace internal
} // namespace geoqp
