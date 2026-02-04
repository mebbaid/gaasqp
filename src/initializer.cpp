#include "geoqp/internal.hpp"
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
            double dist_sq = 0.0;
            for (int i = 0; i < n_; ++i)
            {
                double d = v[i] - x_bar_[i];
                dist_sq += d * d;
            }
            return dist_sq;
        }

        bool Initializer::is_good_vertex(
            const std::vector<double> &v,
            const std::vector<int> &active_set) const
        {
            // A vertex v is "good" if: a_i^T (x_bar - v) <= 0 for all active constraints i
            for (int idx : active_set)
            {
                const double *a_row = A_ + idx * n_;
                double dot = 0.0;
                for (int j = 0; j < n_; ++j)
                {
                    dot += a_row[j] * (x_bar_[j] - v[j]);
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

            if (is_box_constrained_)
            {
                vertices_explored = 1;
                iterations = 1;
                return find_good_vertex_box_fast();
            }

            Polytope polytope(A_, b_, m_, n_);

            std::vector<int> current_active = polytope.get_active_constraints(initial_vertex, 1e-8);
            std::vector<double> current_vertex = initial_vertex;
            vertices_explored = 1;

            std::vector<double> best_good_vertex;
            double best_distance_sq = -1.0;

            if (is_good_vertex(current_vertex, current_active))
            {
                best_good_vertex = current_vertex;
                best_distance_sq = compute_euclidean_distance_squared(current_vertex);
            }

            // pivot toward vertices farther from x_bar
            const int max_iter = config_.max_simplex_iterations;

            while (iterations < max_iter)
            {
                iterations++;

                if (current_active.size() < static_cast<size_t>(n_))
                {
                    break;
                }

                std::vector<double> direction(n_);
                for (int j = 0; j < n_; ++j)
                {
                    direction[j] = current_vertex[j] - x_bar_[j];
                }

                int best_drop = -1;
                double best_ratio = 0.0;

                for (int k = 0; k < n_ && k < static_cast<int>(current_active.size()); ++k)
                {
                    int idx = current_active[k];
                    const double *a_row = A_ + idx * n_;
                    double a_dot_d = LinearAlgebra::dot(a_row, direction.data(), n_);

                    if (a_dot_d < -config_.tolerance && a_dot_d < best_ratio)
                    {
                        best_ratio = a_dot_d;
                        best_drop = k;
                    }
                }

                if (best_drop < 0)
                {
                    break; 
                }

                std::vector<int> reduced_active;
                for (int k = 0; k < n_ && k < static_cast<int>(current_active.size()); ++k)
                {
                    if (k != best_drop)
                    {
                        reduced_active.push_back(current_active[k]);
                    }
                }

                double min_t = std::numeric_limits<double>::infinity();
                int blocking = -1;

                for (int c = 0; c < m_; ++c)
                {
                    bool in_reduced = false;
                    for (int a : reduced_active)
                    {
                        if (a == c)
                        {
                            in_reduced = true;
                            break;
                        }
                    }
                    if (in_reduced)
                        continue;

                    const double *a_row = A_ + c * n_;
                    double a_dot_d = LinearAlgebra::dot(a_row, direction.data(), n_);
                    double a_dot_x = LinearAlgebra::dot(a_row, current_vertex.data(), n_);

                    if (a_dot_d > config_.tolerance)
                    {
                        double t = (b_[c] - a_dot_x) / a_dot_d;
                        if (t > config_.tolerance && t < min_t)
                        {
                            min_t = t;
                            blocking = c;
                        }
                    }
                }

                if (blocking < 0 || !std::isfinite(min_t))
                {
                    break;
                }

                std::vector<int> new_active = reduced_active;
                new_active.push_back(blocking);

                if (new_active.size() != static_cast<size_t>(n_))
                {
                    break;
                }

                std::vector<double> new_vertex = compute_vertex_from_active_set(new_active);
                if (new_vertex.empty())
                {
                    break;
                }

                vertices_explored++;

                double new_dist_sq = compute_euclidean_distance_squared(new_vertex);

                if (is_good_vertex(new_vertex, new_active))
                {
                    if (new_dist_sq > best_distance_sq)
                    {
                        best_good_vertex = new_vertex;
                        best_distance_sq = new_dist_sq;
                    }
                }

                double current_dist_sq = compute_euclidean_distance_squared(current_vertex);
                if (new_dist_sq <= current_dist_sq + config_.tolerance)
                {
                    break;
                }

                current_vertex = std::move(new_vertex);
                current_active = std::move(new_active);
            }

            if (best_good_vertex.empty())
            {
                return initial_vertex;
            }
            return best_good_vertex;
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
