// Lightweight timing counters for profiling hot paths
#pragma once
#include <atomic>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cstdlib>

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

struct ScopedTimer
{
    std::atomic<uint64_t> &ns;
    std::atomic<uint64_t> &count;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedTimer(std::atomic<uint64_t> &ns_, std::atomic<uint64_t> &count_) : ns(ns_), count(count_), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer()
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t d = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        ns.fetch_add(d, std::memory_order_relaxed);
        count.fetch_add(1, std::memory_order_relaxed);
    }
};

inline void print_report()
{
    using std::cout; using std::endl;
    cout << "--- GeoQP timing report (nanoseconds / counts) ---\n";
    cout << "Initializer: " << initializer_ns.load() << " ns over " << initializer_count.load() << " calls\n";
    cout << "  find_initial_vertex: " << initializer_find_initial_vertex_ns.load() << " ns over " << initializer_find_initial_vertex_count.load() << " calls\n";
    cout << "  neighbors:           " << initializer_neighbors_ns.load() << " ns over " << initializer_neighbors_count.load() << " calls\n";
    cout << "  active_check:        " << initializer_active_ns.load() << " ns over " << initializer_active_count.load() << " calls\n";
    cout << "  candidate_solve:     " << initializer_candidate_ns.load() << " ns over " << initializer_candidate_count.load() << " calls\n";
    cout << "  raycast:             " << initializer_raycast_ns.load() << " ns over " << initializer_raycast_count.load() << " calls\n";
    cout << "Polytope find_initial_vertex:\n";
    cout << "  iter_loop:           " << poly_find_iter_ns.load() << " ns over " << poly_find_iter_count.load() << " calls\n";
    cout << "  final_solve:         " << poly_find_solve_ns.load() << " ns over " << poly_find_solve_count.load() << " calls\n";
    cout << "  active_check:        " << poly_find_active_ns.load() << " ns over " << poly_find_active_count.load() << " calls\n";
    cout << "Polytope neighbors:\n";
    cout << "  candidate_solve:     " << poly_neighbors_solve_ns.load() << " ns over " << poly_neighbors_solve_count.load() << " calls\n";
    cout << "  feasibility_check:   " << poly_neighbors_feas_ns.load() << " ns over " << poly_neighbors_feas_count.load() << " calls\n";
    cout << "Initializer hash/eq:\n";
    cout << "  hash:                " << init_hash_ns.load() << " ns over " << init_hash_count.load() << " calls\n";
    cout << "  eq:                  " << init_eq_ns.load() << " ns over " << init_eq_count.load() << " calls\n";
    cout << "  pick:                " << initializer_pick_ns.load() << " ns over " << initializer_pick_count.load() << " calls\n";
    cout << "  push/insert:         " << initializer_push_ns.load() << " ns over " << initializer_push_count.load() << " calls\n";
    cout << "Projection:  " << projection_ns.load() << " ns over " << projection_count.load() << " calls\n";
    cout << "Schur ops:   " << schur_ns.load() << " ns over " << schur_count.load() << " calls\n";
    cout << "Solve ops:   " << solve_ns.load() << " ns over " << solve_count.load() << " calls\n";
}

struct ReporterRegister { ReporterRegister() { std::atexit(print_report); } };
static ReporterRegister _reporter_register;

} // namespace timing
} // namespace geoqp
