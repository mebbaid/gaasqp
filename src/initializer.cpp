#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include "geoqp/timing.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>
#include <cstdint>

namespace geoqp
{
    namespace internal
    {
        inline uint64_t compute_vertex_fingerprint(const double *vptr, int n, double tol)
        {
            const double use_tol = std::max(tol, 1e-12);
            uint64_t h = 14695981039346656037ULL;
            for (int i = 0; i < n; ++i)
            {
                int64_t q = llround(vptr[i] / use_tol);
                uint64_t x = static_cast<uint64_t>(q);
                h ^= x;
                h *= 1099511628211ULL;
            }
            return h;
        }

        inline bool vertices_approx_equal(const double *a, const double *b, int n, double tol)
        {
            for (int i = 0; i < n; ++i)
                if (std::abs(a[i] - b[i]) > tol)
                    return false;
            return true;
        }

        inline bool is_vertex_visited(
            const double *vptr, int n, double tol,
            const std::vector<uint64_t> &visited_hash,
            const std::vector<int> &visited_idx,
            const std::vector<double> &vertices)
        {
            uint64_t vh = compute_vertex_fingerprint(vptr, n, tol);
            for (uint64_t hv : visited_hash)
            {
                ::geoqp::timing::init_counter_hash_checks.fetch_add(1, std::memory_order_relaxed);
                if (hv == vh)
                {
                    for (int vid : visited_idx)
                    {
                        ::geoqp::timing::init_counter_eq_checks.fetch_add(1, std::memory_order_relaxed);
                        const double *visited_ptr = vertices.data() + static_cast<size_t>(vid) * static_cast<size_t>(n);
                        if (vertices_approx_equal(visited_ptr, vptr, n, tol))
                            return true;
                    }
                }
            }
            return false;
        }

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
            ws_Ldiff_.resize(n);
            ws_nu_.resize(n, 0.0);
        }

        void Initializer::update(const double *b, const std::vector<double> &x_bar)
        {
            b_ = b;
            x_bar_ = x_bar;
            compute_axbar();
        }

        void Initializer::set_Q_chol(const std::vector<double> &Q_chol)
        {
            Q_chol_ = Q_chol;
            has_Q_chol_ = !Q_chol.empty();
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
                        if (count > 1)
                            break;
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

        double Initializer::compute_q_weighted_distance_squared_ptr(const double *vptr) const
        {
            const double *GEOQP_RESTRICT xb = x_bar_.data();

            if (has_Q_chol_)
            {
                // ‖v - x̄‖_Q² = ‖L(v - x̄)‖₂²  where Q = L Lᵀ.
                // Forward substitution: solve L · w = (v - x̄).
                // L is lower-triangular, stored row-major (L[i*n+j], j<=i).
                const double *GEOQP_RESTRICT L = Q_chol_.data();
                double *GEOQP_RESTRICT w = ws_Ldiff_.data();
                for (int i = 0; i < n_; ++i)
                {
                    const double *GEOQP_RESTRICT Li = L + i * n_;
                    double s = vptr[i] - xb[i];
                    for (int j = 0; j < i; ++j)
                        s -= Li[j] * w[j];
                    w[i] = s / Li[i];
                }
                // ‖w‖₂²
                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                int i = 0;
                for (; i + 3 < n_; i += 4)
                {
                    sum0 += w[i]   * w[i];
                    sum1 += w[i+1] * w[i+1];
                    sum2 += w[i+2] * w[i+2];
                    sum3 += w[i+3] * w[i+3];
                }
                double res = (sum0 + sum1) + (sum2 + sum3);
                for (; i < n_; ++i) res += w[i] * w[i];
                return res;
            }
            else
            {
                // Fallback: Q-weighted via full matvec (v - x̄)ᵀ Q (v - x̄).
                double *GEOQP_RESTRICT w = ws_Ldiff_.data();
                for (int i = 0; i < n_; ++i) w[i] = vptr[i] - xb[i];

                const double *GEOQP_RESTRICT Q = Q_;
                double sum = 0.0;
                for (int i = 0; i < n_; ++i)
                {
                    const double *GEOQP_RESTRICT Qi = Q + i * n_;
                    double row = 0.0;
                    int j = 0;
                    double r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
                    for (; j + 3 < n_; j += 4)
                    {
                        r0 += Qi[j]   * w[j];
                        r1 += Qi[j+1] * w[j+1];
                        r2 += Qi[j+2] * w[j+2];
                        r3 += Qi[j+3] * w[j+3];
                    }
                    row = (r0 + r1) + (r2 + r3);
                    for (; j < n_; ++j) row += Qi[j] * w[j];
                    sum += w[i] * row;
                }
                return sum;
            }
        }

        double Initializer::compute_separation_direction(const double *closest_boundary_pt) const
        {
            // Compute ν ∝ Q(x̄ - x̂) where x̂ ≈ closest_boundary_pt.
            // With Cholesky L available: Q(x̄ - x̂) = Lᵀ(L(x̄ - x̂)).
            // We compute and normalise in-place into ws_nu_.
            const double *GEOQP_RESTRICT xb = x_bar_.data();
            double *GEOQP_RESTRICT nu = ws_nu_.data();

            if (has_Q_chol_)
            {
                const double *GEOQP_RESTRICT L = Q_chol_.data();
                double *GEOQP_RESTRICT w = ws_Ldiff_.data();
                // Forward sub: L w = (x̄ - x̂)
                for (int i = 0; i < n_; ++i)
                {
                    const double *GEOQP_RESTRICT Li = L + i * n_;
                    double s = xb[i] - closest_boundary_pt[i];
                    for (int j = 0; j < i; ++j) s -= Li[j] * w[j];
                    w[i] = s / Li[i];
                }
                // Backward sub: Lᵀ nu = w  →  nu = Q(x̄ - x̂)
                for (int i = n_ - 1; i >= 0; --i)
                {
                    double s = w[i];
                    for (int j = i + 1; j < n_; ++j)
                        s -= L[j * n_ + i] * nu[j];
                    nu[i] = s / L[i * n_ + i];
                }
            }
            else
            {
                // Fallback: nu = Q(x̄ - x̂) via matvec
                const double *GEOQP_RESTRICT Q = Q_;
                for (int i = 0; i < n_; ++i)
                {
                    const double *GEOQP_RESTRICT Qi = Q + i * n_;
                    double s = 0.0;
                    for (int j = 0; j < n_; ++j)
                        s += Qi[j] * (xb[j] - closest_boundary_pt[j]);
                    nu[i] = s;
                }
            }

            // Compute L = ‖x̄ - x̂‖_Q = ‖nu‖ (before normalisation)
            double norm_sq = 0.0;
            for (int i = 0; i < n_; ++i) norm_sq += nu[i] * nu[i];
            double L_val = std::sqrt(norm_sq);

            if (L_val > 1e-14)
            {
                double inv = 1.0 / L_val;
                for (int i = 0; i < n_; ++i) nu[i] *= inv;
            }
            else
            {
                // ν undefined — zero it so the score degenerates to pure Q-distance
                std::fill(nu, nu + n_, 0.0);
            }
            return L_val;
        }

        inline double Initializer::bfs_score(const double *vptr, double two_L) const
        {
            // score(v) = ‖v - x̄‖_Q²  -  two_L · ⟨ν, v⟩
            // Maximising this is equivalent to finding v* per Lemma 2.
            double q_dist = compute_q_weighted_distance_squared_ptr(vptr);
            const double *GEOQP_RESTRICT nu = ws_nu_.data();
            double nu_dot = 0.0;
            for (int i = 0; i < n_; ++i) nu_dot += nu[i] * vptr[i];
            return q_dist - two_L * nu_dot;
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
            geoqp::timing::ScopedTimer _t(geoqp::timing::initializer_ns, geoqp::timing::initializer_count);
            iterations = 0;
            vertices_explored = 0;

            if (is_box_constrained_)
            {
                vertices_explored = 1;
                iterations = 1;
                return find_good_vertex_box_fast();
            }

            // Try a local vertex-search (greedy neighbor exploration) to find a far
            // good vertex without performing full vertex enumeration.
            Polytope polytope(A_, b_, m_, n_);

            std::vector<double> start = initial_vertex;
            if (!polytope.is_vertex(start))
            {
                try
                {
                    geoqp::timing::ScopedTimer __t_init_sub(geoqp::timing::initializer_find_initial_vertex_ns, geoqp::timing::initializer_find_initial_vertex_count);
                    start = polytope.find_initial_vertex();
                }
                catch (...)
                {
                    start = initial_vertex;
                }
            }

            const int reserve_hint = std::min(512, std::max(32, n_ * 8));
            std::vector<double> vertices;
            vertices.reserve(static_cast<size_t>(reserve_hint) * static_cast<size_t>(n_));

            std::vector<int> frontier_idx;
            frontier_idx.reserve(reserve_hint);
            std::vector<int> visited_idx;
            visited_idx.reserve(reserve_hint);

            std::vector<uint64_t> visited_hash;
            visited_hash.reserve(reserve_hint);

            auto push_vertex = [&](const std::vector<double> &v) -> int
            {
                int idx = static_cast<int>(vertices.size() / n_);
                vertices.insert(vertices.end(), v.begin(), v.end());
                return idx;
            };

            auto ptr_of = [&](int idx) -> const double *
            { return vertices.data() + static_cast<size_t>(idx) * static_cast<size_t>(n_); };

            int start_idx = push_vertex(start);
            visited_idx.push_back(start_idx);
            visited_hash.push_back(compute_vertex_fingerprint(start.data(), n_, 1e-8));

            {
                size_t before = vertices.size();
                {
                    geoqp::timing::ScopedTimer __t_nei_sub(geoqp::timing::initializer_neighbors_ns, geoqp::timing::initializer_neighbors_count);
                    polytope.get_neighbors_flat(start, vertices);
                }
                size_t after = vertices.size();
                for (size_t off = before; off + static_cast<size_t>(n_) <= after; off += static_cast<size_t>(n_))
                {
                    int vidx = static_cast<int>(off / static_cast<size_t>(n_));
                    frontier_idx.push_back(vidx);
                }
            }

            // --- Compute separation direction ν and scaling 2·L for Lemma 2 score ---
            // Use the closest-constraint boundary point as the proxy for x̂.
            // This is the point on the most-violated constraint hyperplane
            // closest to x̄, i.e.  x̂_proxy = x̄ - slack_i/‖aᵢ‖² · aᵢ.
            double two_L = 0.0;
            {
                const double *GEOQP_RESTRICT xb = x_bar_.data();
                int worst_i = -1;
                double worst_slack = 0.0;
                for (int i = 0; i < m_; ++i)
                {
                    // slack = aᵢᵀ x̄ - bᵢ  (positive means violated)
                    double s = ws_Axbar_[i] - b_[i];
                    if (s > worst_slack) { worst_slack = s; worst_i = i; }
                }

                if (worst_i >= 0)
                {
                    // Closest point on hyperplane aᵢᵀx = bᵢ  to x̄:
                    //   x̂_proxy = x̄ - (aᵢᵀx̄ - bᵢ)/‖aᵢ‖² · aᵢ
                    const double *GEOQP_RESTRICT ai = A_ + worst_i * n_;
                    double ai_sq = 0.0;
                    for (int j = 0; j < n_; ++j) ai_sq += ai[j] * ai[j];
                    if (ai_sq > 1e-20)
                    {
                        double alpha = worst_slack / ai_sq;
                        // proxy = x̄ - alpha·aᵢ  (stored in ws_direction_ temporarily)
                        double *GEOQP_RESTRICT proxy = ws_direction_.data();
                        for (int j = 0; j < n_; ++j) proxy[j] = xb[j] - alpha * ai[j];
                        two_L = 2.0 * compute_separation_direction(proxy);
                    }
                }
                // If two_L == 0.0, ws_nu_ is all-zeros → score degenerates to
                // pure Q-weighted distance, which is still correct.
            }

            std::vector<double> &best_vertex = ws_candidate_;
            best_vertex.assign(n_, 0.0);
            double best_score = -std::numeric_limits<double>::infinity();

            {
                auto active = polytope.get_active_constraints(start);
                if (is_good_vertex(start, active))
                {
                    best_vertex = start;
                    best_score = bfs_score(ptr_of(start_idx), two_L);
                }
            }

            const int max_visit = std::max(100, config_.max_simplex_iterations);
            while (!frontier_idx.empty() && static_cast<int>(vertices_explored) < max_visit)
            {
                // Pick frontier vertex with the highest Lemma-2 score.
                int pick_i = 0;
                double pick_score = -std::numeric_limits<double>::infinity();
                for (size_t fi = 0; fi < frontier_idx.size(); ++fi)
                {
                    double s = bfs_score(ptr_of(frontier_idx[fi]), two_L);
                    if (s > pick_score)
                    {
                        pick_score = s;
                        pick_i = static_cast<int>(fi);
                    }
                }

                int vidx = frontier_idx[pick_i];
                frontier_idx[pick_i] = frontier_idx.back();
                frontier_idx.pop_back();

                ::geoqp::timing::init_counter_frontier_picks.fetch_add(1, std::memory_order_relaxed);

                const double *vptr = ptr_of(vidx);

                if (is_vertex_visited(vptr, n_, 1e-8, visited_hash, visited_idx, vertices))
                    continue;

                visited_idx.push_back(vidx);
                visited_hash.push_back(compute_vertex_fingerprint(vptr, n_, 1e-8));
                vertices_explored++;
                ::geoqp::timing::init_counter_vertices_explored.fetch_add(1, std::memory_order_relaxed);

                // check goodness
                {
                    std::vector<double> vview(n_);
                    for (int i = 0; i < n_; ++i)
                        vview[i] = vptr[i];
                    {
                        geoqp::timing::ScopedTimer __t_active_sub(geoqp::timing::initializer_active_ns, geoqp::timing::initializer_active_count);
                        auto active = polytope.get_active_constraints(vview);
                        if (is_good_vertex(vview, active))
                        {
                            double s = bfs_score(vptr, two_L);
                            if (s > best_score)
                            {
                                best_score = s;
                                for (int i = 0; i < n_; ++i)
                                    best_vertex[i] = vptr[i];
                            }
                        }
                    }
                }

                // Expand neighbors of the current vertex.
                // get_neighbors_flat needs a std::vector; build a view from the
                // already-stored pointer without extra allocation.
                std::vector<double> vview_expand(vptr, vptr + n_);
                size_t before2 = vertices.size();
                {
                    geoqp::timing::ScopedTimer __t_nei_sub2(geoqp::timing::initializer_neighbors_ns, geoqp::timing::initializer_neighbors_count);
                    polytope.get_neighbors_flat(vview_expand, vertices);
                }
                size_t after2 = vertices.size();

                for (size_t off = before2; off + static_cast<size_t>(n_) <= after2; off += static_cast<size_t>(n_))
                {
                    const double *nvptr = vertices.data() + off;
                    bool seen2 = is_vertex_visited(nvptr, n_, 1e-8, visited_hash, visited_idx, vertices);
                    ::geoqp::timing::init_counter_neighbor_expansions.fetch_add(1, std::memory_order_relaxed);

                    if (!seen2)
                    {
                        int nidx = static_cast<int>(off / static_cast<size_t>(n_));
                        frontier_idx.push_back(nidx);
                        ::geoqp::timing::init_counter_frontier_pushes.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }

            if (best_score > -std::numeric_limits<double>::infinity())
            {
                iterations = 1 + static_cast<int>(visited_idx.size());
                return best_vertex;
            }

            // If local search failed to find a good vertex, fall back to the
            // constraint-ranking + ray casting strategy below.
            // Direction: ν (pointing toward the unconstrained min, away from the
            // polytope) — use ws_nu_ if computed, otherwise fall back to
            // (initial_vertex - x̄) normalised.
            const double *GEOQP_RESTRICT init_ptr = initial_vertex.data();
            const double *GEOQP_RESTRICT xbar_ptr = x_bar_.data();
            double *GEOQP_RESTRICT d_ptr = ws_direction_.data();

            // Use ν as the fallback ray direction when it was successfully computed.
            // ν points from x̂ toward x̄ (outward from the polytope), which by
            // Proposition 1 aligns with the exit face constraints.
            // If ν is zero (degenerate case), fall back to (init_vertex - x̄).
            {
                const double *GEOQP_RESTRICT nu = ws_nu_.data();
                double nu_norm_sq = 0.0;
                for (int j = 0; j < n_; ++j) nu_norm_sq += nu[j] * nu[j];

                if (nu_norm_sq > 0.5) // nu is already unit-length when valid
                {
                    // Cast ray from initial_vertex in direction -ν
                    // (i.e. into the polytope toward x̂, then past it).
                    // We want to move away from x̄, so direction is -ν.
                    for (int j = 0; j < n_; ++j) d_ptr[j] = -nu[j];
                }
                else
                {
                    for (int j = 0; j < n_; ++j)
                        d_ptr[j] = init_ptr[j] - xbar_ptr[j];
                }
            }

            double norm_d = LinearAlgebra::norm(d_ptr, n_);
            if (norm_d > 1e-10)
            {
                double inv = 1.0 / norm_d;
                for (int j = 0; j < n_; ++j)
                    d_ptr[j] *= inv;
            }

            LinearAlgebra::matvec(A_, d_ptr, ws_Ad_.data(), m_, n_);

            std::vector<std::pair<double, int>> ranked_constraints(m_);
            for (int i = 0; i < m_; ++i)
            {
                ranked_constraints[i] = {ws_Ad_[i], i};
            }

            if (m_ >= n_)
            {
                std::partial_sort(ranked_constraints.begin(),
                                  ranked_constraints.begin() + n_,
                                  ranked_constraints.end(),
                                  [](const std::pair<double, int> &a, const std::pair<double, int> &b)
                                  {
                                      return a.first > b.first;
                                  });
            }

            if (m_ >= n_)
            {
                std::vector<double> A_sub(n_ * n_);
                std::vector<double> b_sub(n_);
                std::vector<double> v_candidate(n_);

                for (int i = 0; i < n_; ++i)
                {
                    int original_idx = ranked_constraints[i].second;
                    const double *row_ptr = A_ + original_idx * n_;
                    for (int j = 0; j < n_; ++j)
                    {
                        A_sub[i * n_ + j] = row_ptr[j];
                    }
                    b_sub[i] = b_[original_idx];
                }

                try
                {
                    LinearAlgebra::solve_linear_system(A_sub.data(), b_sub.data(), v_candidate.data(), n_);

                    Polytope poly2(A_, b_, m_, n_);
                    if (poly2.is_vertex(v_candidate))
                    {
                        std::vector<int> active_set;
                        active_set.reserve(n_);
                        for (int i = 0; i < n_; ++i)
                            active_set.push_back(ranked_constraints[i].second);

                        if (is_good_vertex(v_candidate, active_set))
                        {
                            vertices_explored = 1;
                            iterations = 1;
                            return v_candidate;
                        }
                    }
                }
                catch (...)
                {
                }
            }

            LinearAlgebra::matvec(A_, init_ptr, ws_Ax_.data(), m_, n_);

            const double *GEOQP_RESTRICT Ax_ptr = ws_Ax_.data();
            const double *GEOQP_RESTRICT Ad_ptr = ws_Ad_.data();

            double t_max = std::numeric_limits<double>::infinity();

            for (int i = 0; i < m_; ++i)
            {
                if (Ad_ptr[i] > 1e-12)
                {
                    double dist = b_[i] - Ax_ptr[i];
                    if (dist < 0.0)
                        dist = 0.0; // clamp small violations
                    double t = dist / Ad_ptr[i];
                    if (t < t_max)
                        t_max = t;
                }
            }

            if (!std::isfinite(t_max) || t_max < 0.0)
                t_max = 0.0;

            std::vector<double> furthest_v(n_);
            for (int j = 0; j < n_; ++j)
                furthest_v[j] = init_ptr[j] + t_max * d_ptr[j];

            // Try to promote the raycast point to an actual vertex.
            {
                std::vector<int> active_at_furthest = polytope.get_active_constraints(furthest_v);

                if (static_cast<int>(active_at_furthest.size()) < n_)
                {
                    std::vector<std::pair<double, int>> slack_idx;
                    slack_idx.reserve(m_);
                    for (int i = 0; i < m_; ++i)
                    {
                        const double *a_row = A_ + i * n_;
                        double ax = LinearAlgebra::dot(a_row, furthest_v.data(), n_);
                        double slack = std::abs(b_[i] - ax);
                        slack_idx.emplace_back(slack, i);
                    }
                    std::sort(slack_idx.begin(), slack_idx.end());

                    std::vector<char> is_added(m_, 0);
                    for (int idx : active_at_furthest)
                        is_added[idx] = 1;
                    for (const auto &p : slack_idx)
                    {
                        if ((int)active_at_furthest.size() >= n_)
                            break;
                        int ci = p.second;
                        if (!is_added[ci])
                        {
                            active_at_furthest.push_back(ci);
                            is_added[ci] = 1;
                        }
                    }
                }

                if (static_cast<int>(active_at_furthest.size()) >= n_)
                {
                    std::vector<int> try_set(active_at_furthest.begin(), active_at_furthest.begin() + n_);
                    std::vector<double> solved = compute_vertex_from_active_set(try_set);
                    if (!solved.empty())
                    {
                        vertices_explored++;
                        ::geoqp::timing::init_counter_vertices_explored.fetch_add(1, std::memory_order_relaxed);
                        iterations++;
                        ::geoqp::timing::init_counter_raycast_fallback.fetch_add(1, std::memory_order_relaxed);
                        return solved;
                    }
                }
            }

            vertices_explored++;
            ::geoqp::timing::init_counter_vertices_explored.fetch_add(1, std::memory_order_relaxed);
            iterations++;
            ::geoqp::timing::init_counter_raycast_fallback.fetch_add(1, std::memory_order_relaxed);
            return furthest_v;
        }

    } // namespace internal
} // namespace geoqp
