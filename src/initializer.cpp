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
        // Helper: Compute squared Euclidean distance from vertex to x_bar using SIMD unrolling
        inline double compute_distance_squared_ptr(
            const double *vptr, const double *xbar_ptr, int n)
        {
            const double *GEOQP_RESTRICT v = vptr;
            const double *GEOQP_RESTRICT xb = xbar_ptr;
            
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            int i = 0;
            GEOQP_UNROLL_LOOP_4
            for (; i + 3 < n; i += 4)
            {
                double d0 = v[i] - xb[i];
                double d1 = v[i + 1] - xb[i + 1];
                double d2 = v[i + 2] - xb[i + 2];
                double d3 = v[i + 3] - xb[i + 3];
                sum0 += d0 * d0;
                sum1 += d1 * d1;
                sum2 += d2 * d2;
                sum3 += d3 * d3;
            }
            double res = (sum0 + sum1) + (sum2 + sum3);
            for (; i < n; ++i)
            {
                double d = v[i] - xb[i];
                res += d * d;
            }
            return res;
        }

        // Helper: Fast 64-bit fingerprint for a vertex via FNV hashing
        inline uint64_t compute_vertex_fingerprint(const double *vptr, int n, double tol)
        {
            const double use_tol = std::max(tol, 1e-12);
            uint64_t h = 14695981039346656037ULL; // FNV offset
            for (int i = 0; i < n; ++i)
            {
                int64_t q = llround(vptr[i] / use_tol);
                uint64_t x = static_cast<uint64_t>(q);
                h ^= x;
                h *= 1099511628211ULL; // FNV prime
            }
            return h;
        }

        // Helper: Check approximate equality of two vertices
        inline bool vertices_approx_equal(const double *a, const double *b, int n, double tol)
        {
            for (int i = 0; i < n; ++i)
                if (std::abs(a[i] - b[i]) > tol)
                    return false;
            return true;
        }

        // Helper: Check if vertex is already visited (hash + verify pattern)
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
                    // Verify on hash match
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
            return compute_distance_squared_ptr(v.data(), x_bar_.data(), n_);
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
            // good vertex without performing full vertex enumeration. This exploits
            // the combinatorial structure: exploring neighbors typically reaches a
            // far vertex in few hops for well-behaved polytopes.
            Polytope polytope(A_, b_, m_, n_);

            std::vector<double> start = initial_vertex;
            if (!polytope.is_vertex(start))
            {
                try {
                    geoqp::timing::ScopedTimer __t_init_sub(geoqp::timing::initializer_find_initial_vertex_ns, geoqp::timing::initializer_find_initial_vertex_count);
                    start = polytope.find_initial_vertex();
                } catch (...) {
                    // fallback to initial_vertex if we couldn't find a vertex
                    start = initial_vertex;
                }
            }

            // Flat storage for vertices to avoid many small allocations
            const int reserve_hint = std::min(512, std::max(32, n_ * 8));
            std::vector<double> vertices;                // flat: consecutive vertex blocks of length n_
            vertices.reserve(static_cast<size_t>(reserve_hint) * static_cast<size_t>(n_));

            std::vector<int> frontier_idx;               // indices into vertices
            frontier_idx.reserve(reserve_hint);
            std::vector<int> visited_idx;
            visited_idx.reserve(reserve_hint);

            std::vector<uint64_t> visited_hash;
            visited_hash.reserve(reserve_hint);

            // helper: push a vector<double> vertex into flat storage and return its index
            auto push_vertex = [&](const std::vector<double> &v)->int {
                int idx = static_cast<int>(vertices.size() / n_);
                vertices.insert(vertices.end(), v.begin(), v.end());
                return idx;
            };

            // pointer-based push to avoid creating temporaries
            auto push_vertex_ptr = [&](const double *ptr)->int {
                int idx = static_cast<int>(vertices.size() / n_);
                vertices.insert(vertices.end(), ptr, ptr + n_);
                return idx;
            };

            // pointer-based helpers
            auto ptr_of = [&](int idx)->const double* { return vertices.data() + static_cast<size_t>(idx) * static_cast<size_t>(n_); };

            // push start
            int start_idx = push_vertex(start);
            visited_idx.push_back(start_idx);
            visited_hash.push_back(compute_vertex_fingerprint(start.data(), n_, 1e-8));

            // get neighbors of start into frontier using flat API and append directly into vertices
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

            std::vector<double> &best_vertex = ws_candidate_;
            best_vertex.assign(n_, 0.0);
            double best_dist = -1.0;

            // If the start itself is a good vertex, consider it
            {
                auto active = polytope.get_active_constraints(start);
                if (is_good_vertex(start, active))
                {
                    best_vertex = start;
                    best_dist = compute_distance_squared_ptr(ptr_of(start_idx), x_bar_.data(), n_);
                }
            }

            const int max_visit = std::max(100, config_.max_simplex_iterations);
            while (!frontier_idx.empty() && static_cast<int>(vertices_explored) < max_visit)
            {
                // pick frontier index with largest distance
                int pick_i = 0;
                double pick_dist = -1.0;
                for (size_t fi = 0; fi < frontier_idx.size(); ++fi)
                {
                    double d = compute_distance_squared_ptr(ptr_of(frontier_idx[fi]), x_bar_.data(), n_);
                    if (d > pick_dist) { pick_dist = d; pick_i = static_cast<int>(fi); }
                }

                int vidx = frontier_idx[pick_i];
                // swap-pop
                frontier_idx[pick_i] = frontier_idx.back();
                frontier_idx.pop_back();

                // record that we picked a frontier vertex
                ::geoqp::timing::init_counter_frontier_picks.fetch_add(1, std::memory_order_relaxed);

                const double *vptr = ptr_of(vidx);

                // Check if vertex was already visited
                if (is_vertex_visited(vptr, n_, 1e-8, visited_hash, visited_idx, vertices))
                    continue;

                visited_idx.push_back(vidx);
                visited_hash.push_back(compute_vertex_fingerprint(vptr, n_, 1e-8));
                vertices_explored++;
                ::geoqp::timing::init_counter_vertices_explored.fetch_add(1, std::memory_order_relaxed);
                
                // check goodness
                {
                    // create a temporary vector view for polytope API
                    std::vector<double> vview(n_);
                    for (int i = 0; i < n_; ++i) vview[i] = vptr[i];
                    {
                        geoqp::timing::ScopedTimer __t_active_sub(geoqp::timing::initializer_active_ns, geoqp::timing::initializer_active_count);
                        auto active = polytope.get_active_constraints(vview);
                        if (is_good_vertex(vview, active))
                        {
                            double d = compute_distance_squared_ptr(vptr, x_bar_.data(), n_);
                            if (d > best_dist)
                            {
                                best_dist = d;
                                for (int i = 0; i < n_; ++i) best_vertex[i] = vptr[i];
                            }
                        }
                    }
                }

                // expand neighbors by appending directly into vertices via flat API
                ws_candidate_.assign(n_, 0.0);
                for (int ii = 0; ii < n_; ++ii) ws_candidate_[ii] = vptr[ii];
                size_t before2 = vertices.size();
                {
                    geoqp::timing::ScopedTimer __t_nei_sub2(geoqp::timing::initializer_neighbors_ns, geoqp::timing::initializer_neighbors_count);
                    polytope.get_neighbors_flat(ws_candidate_, vertices);
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

            if (best_dist >= 0.0)
            {
                iterations = 1 + static_cast<int>(visited_idx.size());
                return best_vertex;
            }

            // If local search failed to find a good vertex, fall back to the
            // previous constraint-ranking + ray casting strategy below.

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
                std::vector<double> A_sub(n_ * n_);
                std::vector<double> b_sub(n_);
                std::vector<double> v_candidate(n_);

                // Extract rows
                for(int i=0; i<n_; ++i) {
                    int original_idx = ranked_constraints[i].second;
                    const double* row_ptr = A_ + original_idx * n_;
                    for(int j=0; j<n_; ++j) {
                        A_sub[i*n_ + j] = row_ptr[j];
                    }
                    b_sub[i] = b_[original_idx];
                }

                try {
                    LinearAlgebra::solve_linear_system(A_sub.data(), b_sub.data(), v_candidate.data(), n_);

                    Polytope poly2(A_, b_, m_, n_);
                    if (poly2.is_vertex(v_candidate))
                    {
                        std::vector<int> active_set; 
                        active_set.reserve(n_);
                        for(int i=0; i<n_; ++i) active_set.push_back(ranked_constraints[i].second);
                        
                        if (is_good_vertex(v_candidate, active_set)) {
                            vertices_explored = 1;
                            iterations = 1;
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
            
            const double *GEOQP_RESTRICT Ax_ptr = ws_Ax_.data();
            const double *GEOQP_RESTRICT Ad_ptr = ws_Ad_.data();

            double t_max = std::numeric_limits<double>::infinity();
            
            for (int i = 0; i < m_; ++i)
            {
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

            // Try to promote the raycast point to an actual vertex.
            // Strategy: collect currently active (tight) constraints at furthest_v;
            // if fewer than n, add constraints with smallest slack until we have n
            // and attempt to solve the linear system to get a true vertex. If the
            // solve yields a feasible point, return it; otherwise return furthest_v.
            {
                std::vector<int> active_at_furthest = polytope.get_active_constraints(furthest_v);

                if (static_cast<int>(active_at_furthest.size()) < n_)
                {
                    // compute slacks and sort by absolute slack
                    std::vector<std::pair<double,int>> slack_idx;
                    slack_idx.reserve(m_);
                    for (int i = 0; i < m_; ++i)
                    {
                        const double *a_row = A_ + i * n_;
                        double ax = LinearAlgebra::dot(a_row, furthest_v.data(), n_);
                        double slack = std::abs(b_[i] - ax);
                        slack_idx.emplace_back(slack, i);
                    }
                    std::sort(slack_idx.begin(), slack_idx.end());

                    // add smallest-slack constraints until we have n unique indices
                    std::vector<char> is_added(m_, 0);
                    for (int idx : active_at_furthest) is_added[idx] = 1;
                    for (const auto &p : slack_idx)
                    {
                        if ((int)active_at_furthest.size() >= n_) break;
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
                    // try all combinations of n constraints from the set if larger than n
                    // but to keep it cheap, just take the first n (they are sorted by slack)
                    std::vector<int> try_set(active_at_furthest.begin(), active_at_furthest.begin() + n_);
                    std::vector<double> solved = compute_vertex_from_active_set(try_set);
                    if (!solved.empty())
                    {
                        // verified feasible inside compute_vertex_from_active_set
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
