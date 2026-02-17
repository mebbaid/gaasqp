#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include "geoqp/timing.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace geoqp
{
    namespace internal
    {

        Polytope::Polytope(const double *A, const double *b, int m, int n)
            : A_(A), b_(b), m_(m), n_(n) {}

        std::vector<double> Polytope::find_initial_vertex() const
        {
            geoqp::timing::ScopedTimer _t_init(geoqp::timing::initializer_ns, geoqp::timing::initializer_count);
            // TODO: 1d problem is delicate
            if (n_ == 1)
            {
                double min_bound = std::numeric_limits<double>::infinity();
                for (int i = 0; i < m_; ++i)
                {
                    if (std::abs(A_[i]) > 1e-14)
                    {
                        double bound = b_[i] / A_[i];
                        if (A_[i] > 0 && bound < min_bound)
                        {
                            min_bound = bound;
                        }
                    }
                }
                if (std::isfinite(min_bound))
                {
                    return std::vector<double>{min_bound};
                }
                return std::vector<double>{0.0};
            }

            std::vector<double> x(n_, 0.0);

            bool origin_feasible = true;
            for (int i = 0; i < m_; ++i)
            {
                const double *a_row = A_ + i * n_;
                double lhs = LinearAlgebra::dot(a_row, x.data(), n_);
                if (lhs > b_[i] + 1e-10)
                {
                    origin_feasible = false;
                    break;
                }
            }

            if (origin_feasible)
            {
                std::vector<int> active = get_active_constraints(x, 1e-8);
                if (active.size() >= static_cast<size_t>(n_))
                {
                    return x; 
                }
            }

            std::vector<double> point(n_, 0.0);
            const int max_iter = 100;
            const double step_size = 0.1;

            // reuse direction buffer to avoid per-iteration allocations
            std::vector<double> direction(n_, 0.0);

            for (int iter = 0; iter < max_iter; ++iter)
            {
                geoqp::timing::ScopedTimer __t_iter(geoqp::timing::poly_find_iter_ns, geoqp::timing::poly_find_iter_count);
                double max_violation = 0.0;
                std::fill(direction.begin(), direction.end(), 0.0);

                for (int i = 0; i < m_; ++i)
                {
                    const double *a_row = A_ + i * n_;
                    double lhs = LinearAlgebra::dot(a_row, point.data(), n_);
                    double violation = lhs - b_[i];

                    if (violation > max_violation)
                    {
                        max_violation = violation;
                    }

                    if (violation > 1e-8)
                    {
                        for (int j = 0; j < n_; ++j)
                        {
                            direction[j] -= a_row[j] * violation;
                        }
                    }
                }

                if (max_violation < 1e-8)
                {
                    break;
                }

                double norm = 0.0;
                for (int j = 0; j < n_; ++j)
                {
                    norm += direction[j] * direction[j];
                }
                if (norm > 1e-14)
                {
                    norm = std::sqrt(norm);
                    for (int j = 0; j < n_; ++j)
                    {
                        point[j] += step_size * direction[j] / norm;
                    }
                }
            }

            std::vector<int> active_constraints;
            std::vector<double> vertex = point;

            std::vector<std::pair<double, int>> distances;
            distances.reserve(m_);
            for (int i = 0; i < m_; ++i)
            {
                const double *a_row = A_ + i * n_;
                double lhs = LinearAlgebra::dot(a_row, vertex.data(), n_);
                double dist = b_[i] - lhs;
                if (dist >= -1e-8)
                {
                    distances.push_back({dist, i});
                }
            }

            std::sort(distances.begin(), distances.end());

            for (size_t i = 0; i < distances.size() && active_constraints.size() < static_cast<size_t>(n_); ++i)
            {
                active_constraints.push_back(distances[i].second);
            }

            if (active_constraints.size() < static_cast<size_t>(n_))
            {
                throw std::runtime_error("Could not find initial vertex - polytope may be unbounded or degenerate");
            }

            std::vector<double> A_active(n_ * n_);
            std::vector<double> b_active(n_);

            for (int i = 0; i < n_; ++i)
            {
                int constraint_idx = active_constraints[i];
                b_active[i] = b_[constraint_idx];
                for (int j = 0; j < n_; ++j)
                {
                    A_active[i * n_ + j] = A_[constraint_idx * n_ + j];
                }
            }

            try
            {
                {
                    geoqp::timing::ScopedTimer __t_solve(geoqp::timing::poly_find_solve_ns, geoqp::timing::poly_find_solve_count);
                    LinearAlgebra::solve_linear_system(A_active.data(), b_active.data(), vertex.data(), n_);
                }
            }
            catch (...)
            {
                return point;
            }

            for (int i = 0; i < m_; ++i)
            {
                geoqp::timing::ScopedTimer __t_act(geoqp::timing::poly_find_active_ns, geoqp::timing::poly_find_active_count);
                const double *a_row = A_ + i * n_;
                double lhs = LinearAlgebra::dot(a_row, vertex.data(), n_);
                if (lhs > b_[i] + 1e-6)
                {
                    return point;
                }
            }

            return vertex;
        }

        std::vector<std::vector<double>> Polytope::enumerate_vertices() const
        {
            throw std::runtime_error("Vertex enumeration not implemented");
        }

        bool Polytope::is_vertex(const std::vector<double> &x, double tol) const
        {
            auto active = get_active_constraints(x, tol);
            return active.size() >= static_cast<size_t>(n_);
        }

        std::vector<int> Polytope::get_active_constraints(
            const std::vector<double> &x,
            double tol) const
        {
            geoqp::timing::ScopedTimer _t_act(geoqp::timing::projection_ns, geoqp::timing::projection_count);
            std::vector<int> active;
            active.reserve(n_); 

            const double *GEOQP_RESTRICT x_ptr = x.data();

            for (int i = 0; i < m_; ++i)
            {
                const double *GEOQP_RESTRICT a_row = A_ + i * n_;

                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                int j = 0;
                for (; j + 3 < n_; j += 4)
                {
                    sum0 += a_row[j] * x_ptr[j];
                    sum1 += a_row[j + 1] * x_ptr[j + 1];
                    sum2 += a_row[j + 2] * x_ptr[j + 2];
                    sum3 += a_row[j + 3] * x_ptr[j + 3];
                }
                double lhs = (sum0 + sum1) + (sum2 + sum3);
                for (; j < n_; ++j)
                {
                    lhs += a_row[j] * x_ptr[j];
                }

                if (std::abs(lhs - b_[i]) <= tol)
                {
                    active.push_back(i);
                }
            }
            return active;
        }

        std::vector<std::vector<double>> Polytope::get_neighbors(
            const std::vector<double> &vertex) const
        {
            geoqp::timing::ScopedTimer _t_nei(geoqp::timing::initializer_ns, geoqp::timing::initializer_count);
            std::vector<std::vector<double>> neighbors;

            auto active = get_active_constraints(vertex, 1e-8);

            // Pre-allocate reusable buffers to avoid inner-loop allocations
            std::vector<int> new_active;
            new_active.reserve(active.size());

            std::vector<int> candidate_active;
            candidate_active.reserve(static_cast<size_t>(n_));

            std::vector<char> is_in_new(m_, 0);

            std::vector<double> A_active(n_ * n_);
            std::vector<double> b_active(n_);
            std::vector<double> new_vertex(n_);

            for (size_t drop_idx = 0; drop_idx < active.size(); ++drop_idx)
            {
                new_active.clear();
                // build new_active (active with one dropped)
                for (size_t i = 0; i < active.size(); ++i)
                {
                    if (i != drop_idx) new_active.push_back(active[i]);
                }

                if (new_active.size() < static_cast<size_t>(n_ - 1)) continue;

                // mark flags for fast membership test
                std::fill(is_in_new.begin(), is_in_new.end(), 0);
                for (int a : new_active) is_in_new[a] = 1;

                for (int i = 0; i < m_; ++i)
                {
                    if (is_in_new[i]) continue;

                    // build candidate active set = new_active + i
                    candidate_active.clear();
                    candidate_active.insert(candidate_active.end(), new_active.begin(), new_active.end());
                    candidate_active.push_back(i);

                    if (candidate_active.size() != static_cast<size_t>(n_)) continue;

                    // fill A_active and b_active
                    for (int k = 0; k < n_; ++k)
                    {
                        int constraint_idx = candidate_active[k];
                        b_active[k] = b_[constraint_idx];
                        for (int j = 0; j < n_; ++j)
                        {
                            A_active[k * n_ + j] = A_[constraint_idx * n_ + j];
                        }
                    }

                    try
                    {
                        LinearAlgebra::solve_linear_system(A_active.data(), b_active.data(),
                                                           new_vertex.data(), n_);

                        bool feasible = true;
                        for (int j = 0; j < m_; ++j)
                        {
                            const double *a_row = A_ + j * n_;
                            double lhs = LinearAlgebra::dot(a_row, new_vertex.data(), n_);
                            if (lhs > b_[j] + 1e-8)
                            {
                                feasible = false;
                                break;
                            }
                        }

                        if (feasible)
                        {
                            neighbors.push_back(new_vertex);
                        }
                    }
                    catch (...)
                    {
                        // Singular system - skip
                    }
                }
            }

            return neighbors;
        }

        void Polytope::get_neighbors_flat(
            const std::vector<double> &vertex,
            std::vector<double> &out_flat) const
        {
            geoqp::timing::ScopedTimer _t_nei(geoqp::timing::initializer_ns, geoqp::timing::initializer_count);

            // count neighbor expansion calls
            ::geoqp::timing::init_counter_neighbor_expansions.fetch_add(1, std::memory_order_relaxed);

            auto active = get_active_constraints(vertex, 1e-8);

            std::vector<int> new_active;
            new_active.reserve(active.size());

            std::vector<int> candidate_active;
            candidate_active.reserve(static_cast<size_t>(n_));

            std::vector<char> is_in_new(m_, 0);

            std::vector<double> A_active(n_ * n_);
            std::vector<double> b_active(n_);
            std::vector<double> new_vertex(n_);

            for (size_t drop_idx = 0; drop_idx < active.size(); ++drop_idx)
            {
                new_active.clear();
                for (size_t i = 0; i < active.size(); ++i)
                {
                    if (i != drop_idx) new_active.push_back(active[i]);
                }

                if (new_active.size() < static_cast<size_t>(n_ - 1)) continue;

                std::fill(is_in_new.begin(), is_in_new.end(), 0);
                for (int a : new_active) is_in_new[a] = 1;

                for (int i = 0; i < m_; ++i)
                {
                    if (is_in_new[i]) continue;

                    candidate_active.clear();
                    candidate_active.insert(candidate_active.end(), new_active.begin(), new_active.end());
                    candidate_active.push_back(i);

                    if (candidate_active.size() != static_cast<size_t>(n_)) continue;

                    for (int k = 0; k < n_; ++k)
                    {
                        int constraint_idx = candidate_active[k];
                        b_active[k] = b_[constraint_idx];
                        for (int j = 0; j < n_; ++j)
                        {
                            A_active[k * n_ + j] = A_[constraint_idx * n_ + j];
                        }
                    }

                    try
                    {
                        {
                            geoqp::timing::ScopedTimer __t_cand(geoqp::timing::poly_neighbors_solve_ns, geoqp::timing::poly_neighbors_solve_count);
                            LinearAlgebra::solve_linear_system(A_active.data(), b_active.data(),
                                                               new_vertex.data(), n_);
                        }

                        bool feasible = true;
                        {
                            geoqp::timing::ScopedTimer __t_feas(geoqp::timing::poly_neighbors_feas_ns, geoqp::timing::poly_neighbors_feas_count);
                            for (int j = 0; j < m_; ++j)
                            {
                                const double *a_row = A_ + j * n_;
                                double lhs = LinearAlgebra::dot(a_row, new_vertex.data(), n_);
                                if (lhs > b_[j] + 1e-8)
                                {
                                    feasible = false;
                                    break;
                                }
                            }
                        }

                        if (feasible)
                        {
                            ::geoqp::timing::init_counter_frontier_pushes.fetch_add(1, std::memory_order_relaxed);
                            size_t old = out_flat.size();
                            out_flat.resize(old + static_cast<size_t>(n_));
                            for (int t = 0; t < n_; ++t) out_flat[old + t] = new_vertex[t];
                        }
                    }
                    catch (...)
                    {
                        // Singular system - skip
                    }
                }
            }
        }

    } // namespace internal
} // namespace geoqp
