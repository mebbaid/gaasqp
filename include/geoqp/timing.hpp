// Lightweight timing counters for profiling hot paths
#pragma once
#include <atomic>
#include <cstdint>


namespace geoqp {
namespace timing {

inline std::atomic<uint64_t> initializer_ns{0};
inline std::atomic<uint64_t> initializer_count{0};

// finer-grained initializer timers
inline std::atomic<uint64_t> initializer_find_initial_vertex_ns{0};
inline std::atomic<uint64_t> initializer_find_initial_vertex_count{0};

inline std::atomic<uint64_t> initializer_neighbors_ns{0};
inline std::atomic<uint64_t> initializer_neighbors_count{0};

inline std::atomic<uint64_t> initializer_active_ns{0};
inline std::atomic<uint64_t> initializer_active_count{0};

inline std::atomic<uint64_t> initializer_candidate_ns{0};
inline std::atomic<uint64_t> initializer_candidate_count{0};

inline std::atomic<uint64_t> initializer_raycast_ns{0};
inline std::atomic<uint64_t> initializer_raycast_count{0};

// polytope fine-grained
inline std::atomic<uint64_t> poly_find_iter_ns{0};
inline std::atomic<uint64_t> poly_find_iter_count{0};
inline std::atomic<uint64_t> poly_find_solve_ns{0};
inline std::atomic<uint64_t> poly_find_solve_count{0};
inline std::atomic<uint64_t> poly_find_active_ns{0};
inline std::atomic<uint64_t> poly_find_active_count{0};

// polytope neighbors
inline std::atomic<uint64_t> poly_neighbors_solve_ns{0};
inline std::atomic<uint64_t> poly_neighbors_solve_count{0};
inline std::atomic<uint64_t> poly_neighbors_feas_ns{0};
inline std::atomic<uint64_t> poly_neighbors_feas_count{0};

// initializer hash/equality checks
inline std::atomic<uint64_t> init_hash_ns{0};
inline std::atomic<uint64_t> init_hash_count{0};
inline std::atomic<uint64_t> init_eq_ns{0};
inline std::atomic<uint64_t> init_eq_count{0};

// initializer pick & push
inline std::atomic<uint64_t> initializer_pick_ns{0};
inline std::atomic<uint64_t> initializer_pick_count{0};
inline std::atomic<uint64_t> initializer_push_ns{0};
inline std::atomic<uint64_t> initializer_push_count{0};

inline std::atomic<uint64_t> projection_ns{0};
inline std::atomic<uint64_t> projection_count{0};

inline std::atomic<uint64_t> schur_ns{0};
inline std::atomic<uint64_t> schur_count{0};

inline std::atomic<uint64_t> solve_ns{0};
inline std::atomic<uint64_t> solve_count{0};

// linear algebra kernel timers
inline std::atomic<uint64_t> la_matvec_ns{0};
inline std::atomic<uint64_t> la_matvec_count{0};
inline std::atomic<uint64_t> la_dot_ns{0};
inline std::atomic<uint64_t> la_dot_count{0};
// element counters for average lengths
inline std::atomic<uint64_t> la_dot_elems{0};
inline std::atomic<uint64_t> la_matvec_elems{0};
inline std::atomic<uint64_t> la_cholesky_ns{0};
inline std::atomic<uint64_t> la_cholesky_count{0};
inline std::atomic<uint64_t> la_forward_ns{0};
inline std::atomic<uint64_t> la_forward_count{0};
inline std::atomic<uint64_t> la_backward_ns{0};
inline std::atomic<uint64_t> la_backward_count{0};
inline std::atomic<uint64_t> la_solve_cholesky_ns{0};
inline std::atomic<uint64_t> la_solve_cholesky_count{0};
inline std::atomic<uint64_t> la_rank1_ns{0};
inline std::atomic<uint64_t> la_rank1_count{0};
inline std::atomic<uint64_t> la_extend_ns{0};
inline std::atomic<uint64_t> la_extend_count{0};
inline std::atomic<uint64_t> la_shrink_ns{0};
inline std::atomic<uint64_t> la_shrink_count{0};

// lightweight counters (increment-only) for hotspot analysis
inline std::atomic<uint64_t> init_counter_vertices_explored{0};
inline std::atomic<uint64_t> init_counter_frontier_picks{0};
inline std::atomic<uint64_t> init_counter_frontier_pushes{0};
inline std::atomic<uint64_t> init_counter_neighbor_expansions{0};
inline std::atomic<uint64_t> init_counter_hash_checks{0};
inline std::atomic<uint64_t> init_counter_eq_checks{0};
inline std::atomic<uint64_t> init_counter_raycast_fallback{0};
inline std::atomic<uint64_t> face_hops_count{0};
// sampled per-hop timing and stats
inline std::atomic<uint64_t> face_hop_ns_total{0};
inline std::atomic<uint64_t> face_hop_sample_count{0};
inline std::atomic<uint64_t> face_hop_face_size_sum{0};
inline std::atomic<uint64_t> face_hop_projection_calls{0};
inline std::atomic<uint64_t> face_hop_projection_failures{0};

struct ScopedTimer
{
    // No-op timer: keep the same constructor signature so call sites compile,
    // but avoid doing any timing or atomic updates to remove measurement overhead.
    ScopedTimer(std::atomic<uint64_t> & /*ns_*/, std::atomic<uint64_t> & /*count_*/) {}
    // Overload used in some places (no-op)
    ScopedTimer() {}
    ~ScopedTimer() {}
};

inline void print_report()
{
    // Intentionally left empty to avoid printing lightweight counters in normal runs.
    // Use a debugger or enable explicit diagnostics if you need these counters.
    (void)0;
}

// No-op reporter registration (keep symbol but do not register atexit handler)

} // namespace timing
} // namespace geoqp
