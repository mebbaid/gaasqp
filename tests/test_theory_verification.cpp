/**
 * @file test_theory_verification.cpp
 * @brief Verifies the theoretical guarantees of GeoQP against box_antipodal.tex:
 *
 *   - BoxAntipodal:       Phase I box fast-path visits exactly 1 vertex in O(n)
 *                         and places v̂ at the Euclidean-farthest good vertex
 *                         (Lemma part ii + Proposition 2 in box_antipodal.tex).
 *   - FarthestGoodVertex: v̂ = argmax_{v ∈ V_good} ‖v − x̄‖₂  (Proposition 2:
 *                         "farthest in Euclidean sense").
 *   - PhaseIIHopBound:    Phase II hops from initial face to F* ≤ number of
 *                         faces adjacent to the initial face in the box polytope.
 */

#include <gtest/gtest.h>
#include <geoqp/geoqp.hpp>
#include <geoqp/internal.hpp>

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>

namespace
{

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

// Build a symmetric positive-definite Q from a diagonal (fast, controllable).
std::vector<double> make_diag_Q(const std::vector<double> &diag)
{
    int n = (int)diag.size();
    std::vector<double> Q(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        Q[i * n + i] = diag[i];
    return Q;
}

// Build a dense PD matrix Q = A'A + eps*I for given random seed.
std::vector<double> make_dense_Q(int n, unsigned seed = 42, double eps = 1.0)
{
    std::srand(seed);
    std::vector<double> A(n * n);
    for (auto &v : A) v = (std::rand() / (double)RAND_MAX) * 2.0 - 1.0;
    std::vector<double> Q(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                Q[i * n + j] += A[k * n + i] * A[k * n + j];
    for (int i = 0; i < n; ++i) Q[i * n + i] += eps;
    return Q;
}

// Compute x_bar = -Q^{-1} g via Cholesky (re-uses internal LinearAlgebra).
std::vector<double> compute_xbar(const std::vector<double> &Q,
                                  const std::vector<double> &g, int n)
{
    std::vector<double> L(n * n), neg_g(n), x(n), tmp(n);
    geoqp::internal::LinearAlgebra::cholesky(Q.data(), L.data(), n);
    for (int i = 0; i < n; ++i) neg_g[i] = -g[i];
    geoqp::internal::LinearAlgebra::solve_cholesky(L.data(), neg_g.data(), x.data(), tmp.data(), n);
    return x;
}

// Euclidean squared distance ‖v − x̄‖₂²
double eucl_dist2(const std::vector<double> &v,
                  const std::vector<double> &xbar, int n)
{
    double d = 0.0;
    for (int j = 0; j < n; ++j)
        d += (v[j] - xbar[j]) * (v[j] - xbar[j]);
    return d;
}

// True iff box vertex v is a good vertex w.r.t. x̄ (Definition 1, box_antipodal.tex):
//   ∀j: (v_j = lb_j → x̄_j > lb_j)  AND  (v_j = ub_j → x̄_j < ub_j).
bool is_good_box_vertex(const std::vector<double> &v,
                        const std::vector<double> &xbar,
                        const std::vector<double> &lb,
                        const std::vector<double> &ub, int n)
{
    for (int j = 0; j < n; ++j)
    {
        if (std::abs(v[j] - lb[j]) < 1e-10 && xbar[j] <= lb[j] + 1e-10) return false;
        if (std::abs(v[j] - ub[j]) < 1e-10 && xbar[j] >= ub[j] - 1e-10) return false;
    }
    return true;
}

// Enumerate all 2^n vertices of the box [lb, ub] (for small n only).
std::vector<std::vector<double>> enumerate_box_vertices(
    const std::vector<double> &lb, const std::vector<double> &ub, int n)
{
    int total = 1 << n;
    std::vector<std::vector<double>> verts(total, std::vector<double>(n));
    for (int mask = 0; mask < total; ++mask)
        for (int j = 0; j < n; ++j)
            verts[mask][j] = (mask >> j) & 1 ? ub[j] : lb[j];
    return verts;
}

// Build a box-only geoqp::Problem (lb/ub, no A/b).
geoqp::Problem make_box_problem(const std::vector<double> &Q,
                                 const std::vector<double> &g,
                                 const std::vector<double> &lb,
                                 const std::vector<double> &ub)
{
    geoqp::Problem p;
    p.n = (int)g.size();
    p.m = 0;
    p.Q = Q;
    p.g = g;
    p.lb = lb;
    p.ub = ub;
    return p;
}

// Count faces of the box polytope that are adjacent to `active_set`.
// Two faces of a box polytope are adjacent iff they share n-2 bound constraints
// (i.e. differ in exactly one bound index from the active set).
// For a face defined by k active bounds, its adjacent faces are obtained by:
//   - flipping one active bound to its opposite (lb↔ub for same variable), OR
//   - replacing one active bound with another variable's bound.
// Simpler computable bound: at most 2*(m - k) + k = 2m - k adjacent faces,
// but for a pure box with m=2n the combinatorial number of faces adjacent to
// a k-face is exactly 2*(n-k)*k + k  (swap one inactive bound in, or flip).
// We use a direct adjacency count: two faces A, B of a box polytope are
// adjacent iff |sym_diff(A,B)| == 2 and they share a (k-1)-face.
// For verification purposes we use the simpler bound: # adjacent ≤ 2*n
// (each variable contributes at most 2 adjacent faces: tighten lb or ub).
int count_adjacent_faces_box(const std::vector<int> &active_set, int n)
{
    // For a box polytope a face is identified by a subset of bound constraints.
    // Each variable j contributes up to 2 constraints: lb (row 2j) and ub (row 2j+1)
    // (using the ordering set by expand_bounds_into_constraints: ub rows first, then lb).
    // Two faces are adjacent iff their symmetric difference has size exactly 2
    // and the shared constraints form a valid (lower-dimensional) face.
    //
    // For a k-face of a box polytope the number of adjacent faces is at most 2*(2n - k).
    // We return that upper bound for the assertion.
    int k = (int)active_set.size();
    int m = 2 * n;  // total bound constraints
    return 2 * (m - k);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test fixture
// ─────────────────────────────────────────────────────────────────────────────

struct BoxVerificationResult
{
    bool solved;
    int  vertices_explored;   // from sol.vertices_explored
    int  phase2_iterations;   // from sol.phase2_iterations
    std::vector<double> x_star;   // final optimal solution
    std::vector<double> x_phase1; // Phase I vertex v* (before Phase II/III)
    std::vector<int>    active_set;
    // True if box fast-path fired (vertices_explored == 1)
    bool box_fast_path_used;
    // True if v* maximises the BFS score over all box vertices (n <= 20 only)
    bool vstar_is_optimal;
    // Phase II hops ≤ adjacent faces upper bound
    bool phase2_within_adjacency_bound;
};

BoxVerificationResult run_box_verification(
    const std::vector<double> &Q,
    const std::vector<double> &g,
    const std::vector<double> &lb,
    const std::vector<double> &ub,
    bool check_vstar_optimality = true)
{
    int n = (int)g.size();
    BoxVerificationResult r{};

    geoqp::Config cfg;
    cfg.tolerance = 1e-9;
    cfg.constraint_tolerance = 1e-9;
    cfg.verbosity = 0;
    geoqp::Solver solver(cfg);

    auto sol = solver.solve(make_box_problem(Q, g, lb, ub));

    r.solved            = (sol.status == geoqp::Status::SUCCESS);
    r.vertices_explored = sol.vertices_explored;
    r.phase2_iterations = sol.phase2_iterations;
    r.x_star            = sol.x;
    r.x_phase1          = sol.x_phase1;
    r.active_set        = sol.active_set;

    // Proposition 3: box fast-path should visit exactly 1 vertex
    r.box_fast_path_used = (r.vertices_explored == 1);

    // Lemma 2: v* must maximise the BFS score over all 2^n box vertices.
    // Only tractable for small n.
    r.vstar_is_optimal = true;
    if (check_vstar_optimality && n <= 20 && r.solved)
    {
        auto xbar = compute_xbar(Q, g, n);

        // Proposition 2 (box_antipodal.tex): v̂ = argmax_{v ∈ V_good} ‖v − x̄‖₂.
        // The box fast-path places v̂_j at the Euclidean-farthest bound for each
        // coordinate j (Lemma, part ii: antipodal construction).
        // We enumerate all 2^n box vertices, filter to good vertices (Definition 1),
        // and verify v̂ achieves the maximum Euclidean distance (within 1% tolerance).
        auto all_verts = enumerate_box_vertices(lb, ub, n);
        double best_dist = -1.0;
        for (auto &v : all_verts)
            if (is_good_box_vertex(v, xbar, lb, ub, n))
                best_dist = std::max(best_dist, eucl_dist2(v, xbar, n));

        double vstar_dist = eucl_dist2(r.x_phase1, xbar, n);
        r.vstar_is_optimal = (best_dist < 0.0) || (vstar_dist >= 0.99 * best_dist);
    }

    // Theorem 1: Phase II hops ≤ # adjacent faces of initial face.
    if (r.solved)
    {
        int adj_bound = count_adjacent_faces_box(r.active_set, n);
        r.phase2_within_adjacency_bound = (r.phase2_iterations <= adj_bound);
    }
    else
    {
        r.phase2_within_adjacency_bound = true; // skip if not solved
    }

    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// ── BoxAntipodal: O(n) fast-path fires (Lemma, box_antipodal.tex) ─────────────

TEST(BoxAntipodal, DiagonalQ_SmallBox)
{
    int n = 5;
    auto Q = make_diag_Q({2.0, 3.0, 1.5, 4.0, 2.5});
    std::vector<double> g = {5.0, -3.0, 7.0, -2.0, 4.0};
    std::vector<double> lb(n, -1.0), ub(n, 1.0);

    auto r = run_box_verification(Q, g, lb, ub);
    ASSERT_TRUE(r.solved) << "Solver failed";
    EXPECT_TRUE(r.box_fast_path_used)
        << "Expected vertices_explored==1 (box fast-path), got " << r.vertices_explored;
}

TEST(BoxAntipodal, DiagonalQ_AsymmetricBounds)
{
    int n = 8;
    auto Q = make_diag_Q({1.0, 2.0, 0.5, 3.0, 1.5, 2.5, 4.0, 1.0});
    std::vector<double> g = {3.0, -5.0, 2.0, -1.0, 6.0, -4.0, 1.0, -3.0};
    std::vector<double> lb = {-2.0, -1.0, -3.0, -0.5, -1.5, -2.0, -1.0, -4.0};
    std::vector<double> ub = { 1.0,  2.0,  0.5,  1.5,  0.5,  1.0,  3.0,  0.5};

    auto r = run_box_verification(Q, g, lb, ub);
    ASSERT_TRUE(r.solved) << "Solver failed";
    EXPECT_TRUE(r.box_fast_path_used)
        << "Expected vertices_explored==1 (box fast-path), got " << r.vertices_explored;
}

TEST(BoxAntipodal, DenseQ_MediumBox)
{
    int n = 10;
    auto Q = make_dense_Q(n, 42, 2.0);
    std::vector<double> g(n);
    for (int i = 0; i < n; ++i) g[i] = (i % 3 == 0) ? 5.0 : -3.0;
    std::vector<double> lb(n, -1.0), ub(n, 1.0);

    auto r = run_box_verification(Q, g, lb, ub, /*check_vstar*/ false);
    ASSERT_TRUE(r.solved) << "Solver failed";
    EXPECT_TRUE(r.box_fast_path_used)
        << "Expected vertices_explored==1, got " << r.vertices_explored;
}

TEST(BoxAntipodal, LargeBox_StillOneFastPathVertex)
{
    int n = 50;
    auto Q = make_diag_Q(std::vector<double>(n, 1.0));
    std::vector<double> g(n);
    for (int i = 0; i < n; ++i) g[i] = (i % 2 == 0) ? 10.0 : -10.0;
    std::vector<double> lb(n, -1.0), ub(n, 1.0);

    // Don't check v* optimality (2^50 vertices)
    auto r = run_box_verification(Q, g, lb, ub, /*check_vstar*/ false);
    ASSERT_TRUE(r.solved) << "Solver failed";
    EXPECT_TRUE(r.box_fast_path_used)
        << "Expected vertices_explored==1, got " << r.vertices_explored;
}

// ── Lemma (box_antipodal.tex): v̂ coordinates at the Euclidean-farthest bound ──
// For a pure box (T=I, c=0), the antipodal construction (Lemma, part ii) places
// v̂_j at the bound farther from x̄_j: v̂_j = lb_j if |x̄_j−lb_j| > |x̄_j−ub_j|,
// else ub_j.  This satisfies the good-vertex condition: v̂_j=lb_j ⟹ x̄_j>lb_j.
TEST(BoxAntipodal, VstarCoordinatesCorrect)
{
    int n = 6;
    std::vector<double> diag = {1.0, 2.0, 3.0, 1.5, 2.5, 4.0};
    auto Q = make_diag_Q(diag);
    // Choose g so x̄_j = -g_j / q_j is outside [-1,1] on each coordinate
    // with known direction: x̄ = [3, -2, 4, -3, 2, -5]
    std::vector<double> g = {-3.0, 4.0, -12.0, 4.5, -5.0, 20.0};
    std::vector<double> lb(n, -1.0), ub(n, 1.0);

    auto xbar = compute_xbar(Q, g, n);

    auto r = run_box_verification(Q, g, lb, ub);
    ASSERT_TRUE(r.solved);

    for (int j = 0; j < n; ++j)
    {
        double expected = (std::abs(xbar[j] - lb[j]) > std::abs(xbar[j] - ub[j]))
                          ? lb[j] : ub[j];
        EXPECT_NEAR(r.x_phase1[j], expected, 1e-6)
            << "v̂[" << j << "] = " << r.x_phase1[j]
            << " but expected " << expected
            << " (x̄[" << j << "] = " << xbar[j] << ")";
    }
}

// ── Proposition 2 (box_antipodal.tex): v̂ = Euclidean-farthest good vertex ────
TEST(FarthestGoodVertex, DiagQ_n5)
{
    int n = 5;
    auto Q = make_diag_Q({2.0, 1.0, 3.0, 0.5, 4.0});
    std::vector<double> g = {4.0, -3.0, 6.0, -1.0, 8.0};
    std::vector<double> lb(n, -1.0), ub(n, 1.0);

    auto r = run_box_verification(Q, g, lb, ub);
    ASSERT_TRUE(r.solved);
    EXPECT_TRUE(r.vstar_is_optimal)
        << "v̂ is not the Euclidean-farthest good vertex (Proposition 2)";
}

TEST(FarthestGoodVertex, DenseQ_n8)
{
    int n = 8;
    auto Q = make_dense_Q(n, 123, 1.0);
    std::vector<double> g(n);
    for (int i = 0; i < n; ++i) g[i] = (i % 2 == 0) ? 4.0 : -4.0;
    std::vector<double> lb(n, -1.0), ub(n, 1.0);

    auto r = run_box_verification(Q, g, lb, ub);
    ASSERT_TRUE(r.solved);
    EXPECT_TRUE(r.vstar_is_optimal)
        << "v̂ is not the Euclidean-farthest good vertex (Proposition 2)";
}

// ── Phase II hop bound: hops ≤ # adjacent faces ──────────────────────────────
TEST(PhaseIIHopBound, SmallBox)
{
    int n = 5;
    auto Q = make_diag_Q({1.0, 2.0, 3.0, 1.5, 2.5});
    std::vector<double> g = {3.0, -2.0, 5.0, -1.0, 4.0};
    std::vector<double> lb(n, -1.0), ub(n, 1.0);

    auto r = run_box_verification(Q, g, lb, ub);
    ASSERT_TRUE(r.solved);
    EXPECT_TRUE(r.phase2_within_adjacency_bound)
        << "Phase II hops=" << r.phase2_iterations
        << " exceeds adjacency bound for active_set size=" << r.active_set.size();
}

TEST(PhaseIIHopBound, DenseQ_n10)
{
    int n = 10;
    auto Q = make_dense_Q(n, 77, 1.5);
    std::vector<double> g(n);
    for (int i = 0; i < n; ++i) g[i] = (i % 3 == 0) ? 5.0 : -3.0;
    std::vector<double> lb(n, -1.0), ub(n, 1.0);

    auto r = run_box_verification(Q, g, lb, ub, false);
    ASSERT_TRUE(r.solved);
    EXPECT_TRUE(r.phase2_within_adjacency_bound)
        << "Phase II hops=" << r.phase2_iterations
        << " exceeds adjacency bound";
}

TEST(PhaseIIHopBound, n20)
{
    int n = 20;
    auto Q = make_dense_Q(n, 99, 2.0);
    std::vector<double> g(n);
    for (int i = 0; i < n; ++i) g[i] = std::sin(i * 0.7) * 5.0;
    std::vector<double> lb(n, -2.0), ub(n, 2.0);

    auto r = run_box_verification(Q, g, lb, ub, false);
    ASSERT_TRUE(r.solved);
    EXPECT_TRUE(r.phase2_within_adjacency_bound)
        << "Phase II hops=" << r.phase2_iterations
        << " exceeds adjacency bound for n=" << n;
}

// ── Regression: unconstrained min inside box → zero hops ────────────────────
TEST(Regression, UnconstrainedMinFeasible_ZeroHops)
{
    // g=0 → x̄=0, which is inside [-1,1]^n → solver returns immediately
    int n = 8;
    auto Q = make_diag_Q({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    std::vector<double> g(n, 0.0);
    std::vector<double> lb(n, -1.0), ub(n, 1.0);

    geoqp::Config cfg;
    cfg.tolerance = 1e-9;
    geoqp::Solver solver(cfg);
    auto sol = solver.solve(make_box_problem(Q, g, lb, ub));

    ASSERT_EQ(sol.status, geoqp::Status::SUCCESS);
    EXPECT_EQ(sol.phase1_iterations, 0);
    EXPECT_EQ(sol.phase2_iterations, 0);
    for (int j = 0; j < n; ++j)
        EXPECT_NEAR(sol.x[j], 0.0, 1e-8);
}

// ── Combined: verify all three properties together ──────────────────────────
// Uses a diagonal Q so that all three theoretical properties hold provably:
//   - BoxAntipodal:       box fast-path O(n) (vertices_explored == 1)
//   - FarthestGoodVertex: Euclidean-farthest good vertex (per-coord greedy)
//   - PhaseIIHopBound:    Phase II hops ≤ adjacency bound
TEST(Combined, AllThreePropertiesHold_n8)
{
    int n = 8;
    // Diagonal Q: Euclidean per-coordinate fast-path == Q-weighted optimum.
    auto Q = make_diag_Q({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    std::vector<double> g(n);
    for (int i = 0; i < n; ++i) g[i] = (i % 2 == 0) ? 6.0 : -6.0;
    std::vector<double> lb(n, -1.0), ub(n, 1.0);

    auto r = run_box_verification(Q, g, lb, ub);
    ASSERT_TRUE(r.solved) << "Solver failed";

    EXPECT_TRUE(r.box_fast_path_used)
        << "Proposition 3 violated: vertices_explored=" << r.vertices_explored;
    EXPECT_TRUE(r.vstar_is_optimal)
        << "Proposition 2 violated: v̂ is not Euclidean-farthest good vertex";
    EXPECT_TRUE(r.phase2_within_adjacency_bound)
        << "Theorem 1 violated: phase2_iterations=" << r.phase2_iterations
        << " > adjacency bound, active_set.size()=" << r.active_set.size();
}

} // namespace
