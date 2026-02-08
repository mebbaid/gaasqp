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
            
            ws_Axbar_.resize(m);
            compute_axbar();

            ws_direction_.resize(n);
            ws_Ax_.resize(m);
            ws_Ad_.resize(m);
            ws_candidate_.resize(n);
        }

        void Initializer::update(const double *b, const std::vector<double> &x_bar)
        {
            b_ = b;
            x_bar_ = x_bar;
            compute_axbar();
        }

        void Initializer::compute_axbar()
        {
            LinearAlgebra::matvec(A_, x_bar_.data(), ws_Axbar_.data(), m_, n_);
        }

        void Initializer::detect_box_constraints()
        {
            is_box_constrained_ = false;
            box_lb_.clear();
            box_ub_.clear();
            has_lb_.clear();
            has_ub_.clear();

            box_lb_.assign(n_, -std::numeric_limits<double>::infinity());
            box_ub_.assign(n_, std::numeric_limits<double>::infinity());
            has_lb_.assign(n_, false);
            has_ub_.assign(n_, false);

            int box_count = 0;
            for (int i = 0; i < m_; ++i)
            {
                const double *a_row = A_ + i * n_;
                int nonzero_idx = -1;
                double nonzero_val = 0.0;
                int count = 0;

                for (int j = 0; j < n_; ++j)
                {
                    if (std::abs(a_row[j]) > 1e-10)
                    {
                        count++;
                        nonzero_idx = j;
                        nonzero_val = a_row[j];
                        if (count > 1) break;
                    }
                }

                if (count == 1)
                {
                    if (std::abs(nonzero_val - 1.0) < 1e-10)
                    {
                        box_ub_[nonzero_idx] = std::min(box_ub_[nonzero_idx], b_[i]);
                        has_ub_[nonzero_idx] = true;
                        box_count++;
                    }
                    else if (std::abs(nonzero_val + 1.0) < 1e-10)
                    {
                        box_lb_[nonzero_idx] = std::max(box_lb_[nonzero_idx], -b_[i]);
                        has_lb_[nonzero_idx] = true;
                        box_count++;
                    }
                }
            }

            if (box_count >= 2 * n_)
            {
                bool all_bounded = true;
                for (int j = 0; j < n_; ++j)
                {
                    if (!has_lb_[j] || !has_ub_[j])
                    {
                        all_bounded = false;
                        break;
                    }
                }
                is_box_constrained_ = all_bounded;
            }
        }

        std::vector<double> Initializer::find_good_vertex_box_fast() const
        {
            std::vector<double> vertex(n_);
            for (int j = 0; j < n_; ++j)
            {
                double dist_to_lb = has_lb_[j] ? (x_bar_[j] - box_lb_[j]) : std::numeric_limits<double>::infinity();
                double dist_to_ub = has_ub_[j] ? (box_ub_[j] - x_bar_[j]) : std::numeric_limits<double>::infinity();

                if (dist_to_lb > dist_to_ub)
                {
                    vertex[j] = box_lb_[j];
                }
                else
                {
                    vertex[j] = box_ub_[j];
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

            if (is_box_constrained_)
            {
                vertices_explored = 1;
                iterations = 1;
                return find_good_vertex_box_fast();
            }


            // Step A: Determine search direction d
            // Ideally d = centroid - x_bar.
            // We use initial_vertex as a proxy for the centroid (assuming it's somewhat central/feasible).
            const double *GEOQP_RESTRICT init_ptr = initial_vertex.data();
            const double *GEOQP_RESTRICT xbar_ptr = x_bar_.data();
            
            double *GEOQP_RESTRICT d_ptr = ws_direction_.data(); 
            for (int j = 0; j < n_; ++j) {
                d_ptr[j] = init_ptr[j] - xbar_ptr[j];
            }
            
            double norm_d = LinearAlgebra::norm(d_ptr, n_);
            if (norm_d > 1e-10) {
                double inv = 1.0/norm_d;
                for(int j=0; j<n_; ++j) d_ptr[j] *= inv;
            }

            // Step B: Score constraints by projection (a_i^T * d)
            LinearAlgebra::matvec(A_, d_ptr, ws_Ad_.data(), m_, n_);
            
            // Store pairs of <score, index>
            std::vector<std::pair<double, int>> ranked_constraints(m_);
            for(int i=0; i<m_; ++i) {
                ranked_constraints[i] = { ws_Ad_[i], i };
            }

            if (m_ >= n_) {
                std::partial_sort(ranked_constraints.begin(), 
                                  ranked_constraints.begin() + n_, 
                                  ranked_constraints.end(),
                                  [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                                      return a.first > b.first;
                                  });
            }

            if (m_ >= n_) 
            {
                // Build small linear system A_sub * v = b_sub
                // We use the Q matrix workspace or factorization workspace since we haven't started main solve yet
                // WARNING: We need n*n space. ws_matrix_Q_ (n*n) is available/unused during init phase?
                // Yes, Q is read-only in 'Q_', 'ws_matrix_Q_' is usually for factorization.
                
                // We need to act carefully not to overwrite data needed later. 
                // Let's assume we can use a temporary buffer or part of the workspace.
                std::vector<double> A_sub(n_ * n_);
                std::vector<double> b_sub(n_);
                std::vector<double> v_candidate(n_);

                // Extract rows
                for(int i=0; i<n_; ++i) {
                    int original_idx = ranked_constraints[i].second;
                    // Copy row A[original_idx]
                    const double* row_ptr = A_ + original_idx * n_;
                    for(int j=0; j<n_; ++j) {
                        A_sub[i*n_ + j] = row_ptr[j];
                    }
                    b_sub[i] = b_[original_idx];
                }

                try {
                    // Solve A_sub * v = b_sub
                    // Use standard Gaussian elimination or QR. 
                    // Since we implemented LinearAlgebra::solve_linear_system:
                    LinearAlgebra::solve_linear_system(A_sub.data(), b_sub.data(), v_candidate.data(), n_);

                    // Check if this candidate is feasible (against ALL constraints)
                    // We allow a small tolerance since we are just finding a start point
                    Polytope polytope(A_, b_, m_, n_);
                    if (polytope.is_vertex(v_candidate))
                    {
                        // Check if it is "Good" (sees x_bar through interior)
                        // Since we constructed it from constraints facing 'd', it is highly likely to be good.
                        std::vector<int> active_set; 
                        active_set.reserve(n_);
                        for(int i=0; i<n_; ++i) active_set.push_back(ranked_constraints[i].second);
                        
                        if (is_good_vertex(v_candidate, active_set)) {
                            vertices_explored = 1;
                            iterations = 1;
                            // This is a robust VERTEX (n constraints active). 
                            // Much better than a random boundary point.
                            return v_candidate; 
                        }
                    }
                } catch (...) {
                    // Singular matrix (constraints were parallel/dependent)
                    // Fall through to ray casting
                }
            }

            // ---------------------------------------------------------
            // 3. Fallback: Ray Casting (The "Back Wall" approach)
            // ---------------------------------------------------------

            LinearAlgebra::matvec(A_, init_ptr, ws_Ax_.data(), m_, n_);
            
            // We already computed Ad (projection of direction) in Step B -> ws_Ad_
            const double *GEOQP_RESTRICT Ax_ptr = ws_Ax_.data();
            const double *GEOQP_RESTRICT Ad_ptr = ws_Ad_.data();

            double t_max = std::numeric_limits<double>::infinity();
            
            for (int i = 0; i < m_; ++i)
            {
                // We only care about constraints we are moving TOWARDS (Ad > 0)
                if (Ad_ptr[i] > 1e-12)
                {
                    double dist = b_[i] - Ax_ptr[i];
                    if (dist < 0.0) dist = 0.0; // clamp small violations
                    double t = dist / Ad_ptr[i];
                    if (t < t_max) t_max = t;
                }
            }

            if (!std::isfinite(t_max) || t_max < 0.0) t_max = 0.0;

            std::vector<double> furthest_v(n_);
            for (int j = 0; j < n_; ++j)
                furthest_v[j] = init_ptr[j] + t_max * d_ptr[j];

            // Note: furthest_v is likely on a Face (1 active constraint), not a Vertex.
            // The solver must handle rank-deficient initial active sets or perform Phase I pivots.
            // However, this is the best we can do without expensive vertex enumeration.
            
            vertices_explored++;
            iterations++;
            return furthest_v;
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
