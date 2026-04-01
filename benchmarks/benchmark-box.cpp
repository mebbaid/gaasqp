/**
 * @file benchmark-box.cpp
 * @brief Theory-focused benchmark for box QPs and affine-image-of-box QPs,
 *        following box_antipodal.tex.
 *
 * Verifies and characterises the three main results of box_antipodal.tex:
 *
 *   Lemma           (§Background): Good vertices of T(B)+c are characterised
 *                   by the antipodal rule  ŷᵢ = ℓᵢ if ȳᵢ > ℓᵢ, else uᵢ.
 *                   For axis-aligned boxes this is O(n) and vertices_explored == 1.
 *
 *   Proposition 1   (prop:halfbox_affine): x* ∈ P₊, the half-box defined by
 *                   the Euclidean projection direction v = x̄ − xₑ.
 *                   Measures margin  v⊤(x* − c₀) ≥ 0.
 *
 *   Proposition 2   (prop:three_halfspaces): x* ∈ P₊ ∩ H₁ ∩ H₂  where
 *                       H₁ = { x : −∇f(v̂)⊤(x−v̂) ≥ 0 }
 *                       H₂ = { x : (x̄−v̂)⊤(x−v̂) ≥ 0 }
 *                   and v̂ is the Euclidean-farthest good vertex.
 *                   Reports all three margins and their normalisation by ‖x*−x̄‖.
 *
 * Additional sections:
 *   Phase II hop analysis — shows how the H₁/H₂-guided initialisation
 *                   (from Proposition 2) constrains the number of face hops.
 *   Affine image    — verifies the theory for diagonal-T and dense-T images
 *                   T(B)+c; shows that the box fast-path fires for diagonal T.
 *   Performance     — GeoQP vs qpOASES timing over varying n and Q type.
 *   Hotstart        — fixed Q, repeated solves with varying g.
 *
 * Code partially written with the help of GitHub Copilot.
 */

#include <geoqp/geoqp.hpp>

#include <qpOASES.hpp>
#include <QpSolversEigen/QpSolversEigen.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "benchmark_utils.hpp"

#ifdef HAVE_OSQP
#include <OsqpEigen/Solver.hpp>
#endif

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <functional>
#include <numeric>
#include <cassert>

using Clock = std::chrono::high_resolution_clock;

// Ensure that x̄ = -Q⁻¹g lies strictly outside the box [lb, ub] by at least
// `min_viol` times the per-dimension half-width from the box centre.
// When randomPD produces eigenvalues O(n), x̄ shrinks as O(1/n) and can fall
// inside the box for large n.  Scaling g uniformly preserves problem structure.
static void ensure_xbar_infeasible(
    const Eigen::MatrixXd &H, Eigen::VectorXd &g,
    const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
    double min_viol = 2.0)
{
    Eigen::VectorXd xbar = -H.ldlt().solve(g);
    Eigen::VectorXd centre   = 0.5 * (lb + ub);
    Eigen::VectorXd halfwidth = 0.5 * (ub - lb);
    double max_rel = 0.0;
    for (int i = 0; i < xbar.size(); ++i)
        max_rel = std::max(max_rel,
            std::abs(xbar(i) - centre(i)) / std::max(halfwidth(i), 1e-12));
    if (max_rel < min_viol)
        g *= (min_viol / std::max(max_rel, 1e-14));
}

// ═══════════════════════════════════════════════════════════════════════════
//  box_antipodal.tex geometric helpers
// ═══════════════════════════════════════════════════════════════════════════

// Euclidean squared distance ‖v − x̄‖₂²
static double eucl_dist_sq(const Eigen::VectorXd &v, const Eigen::VectorXd &xbar)
{
    return (v - xbar).squaredNorm();
}

// Euclidean projection of xbar onto axis-aligned box [lb, ub]
// Valid when T is diagonal (P is an axis-aligned box in x-space)
static Eigen::VectorXd project_onto_box(const Eigen::VectorXd &xbar,
                                         const Eigen::VectorXd &lb,
                                         const Eigen::VectorXd &ub)
{
    return xbar.cwiseMax(lb).cwiseMin(ub);
}

// Antipodal good vertex (Lemma part ii, box_antipodal.tex):
//   ŷᵢ = ℓᵢ  if ȳᵢ > ℓᵢ,  else uᵢ
// For axis-aligned boxes (T=I, c=0): same in x-space.
// Takes the Euclidean-farthest bound per coordinate (Proposition 2).
static Eigen::VectorXd good_vertex_antipodal(const Eigen::VectorXd &xbar,
                                              const Eigen::VectorXd &lb,
                                              const Eigen::VectorXd &ub)
{
    int n = xbar.size();
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i)
    {
        double dl = std::abs(xbar(i) - lb(i));
        double du = std::abs(xbar(i) - ub(i));
        v(i) = (dl > du) ? lb(i) : ub(i);
    }
    return v;
}

// True iff v is a good vertex of the axis-aligned box [lb,ub] w.r.t. x̄
// (Definition 1, box_antipodal.tex):  ∀i: (vᵢ=lbᵢ ⟹ x̄ᵢ>lbᵢ)  and  (vᵢ=ubᵢ ⟹ x̄ᵢ<ubᵢ)
static bool is_good_box_vertex(const Eigen::VectorXd &v,
                                const Eigen::VectorXd &xbar,
                                const Eigen::VectorXd &lb,
                                const Eigen::VectorXd &ub,
                                double tol = 1e-8)
{
    int n = v.size();
    for (int i = 0; i < n; ++i)
    {
        if (std::abs(v(i) - lb(i)) < tol && xbar(i) <= lb(i) + tol) return false;
        if (std::abs(v(i) - ub(i)) < tol && xbar(i) >= ub(i) - tol) return false;
    }
    return true;
}

// Enumerate all 2^n vertices of the axis-aligned box [lb, ub]
static std::vector<Eigen::VectorXd> enumerate_box_vertices(
    const Eigen::VectorXd &lb, const Eigen::VectorXd &ub)
{
    int n = lb.size();
    int total = 1 << n;
    std::vector<Eigen::VectorXd> verts(total);
    for (int mask = 0; mask < total; ++mask)
    {
        verts[mask].resize(n);
        for (int j = 0; j < n; ++j)
            verts[mask](j) = ((mask >> j) & 1) ? ub(j) : lb(j);
    }
    return verts;
}

// Proposition 1 (prop:halfbox_affine): margin of x* in P₊
//   v = x̄ − xₑ  (Euclidean projection direction)
//   c₀ = (lb+ub)/2  (box centre)
//   margin = v⊤(x* − c₀)  ← must be ≥ 0
static double pplus_margin(const Eigen::VectorXd &xstar,
                            const Eigen::VectorXd &xbar,
                            const Eigen::VectorXd &lb,
                            const Eigen::VectorXd &ub)
{
    Eigen::VectorXd xe = project_onto_box(xbar, lb, ub);
    Eigen::VectorXd v  = xbar - xe;
    Eigen::VectorXd c0 = 0.5 * (lb + ub);
    return v.dot(xstar - c0);
}

// Proposition 2 (H₁): margin of x* in H₁
//   H₁ = { x : −∇f(v̂)⊤(x−v̂) ≥ 0 },  −∇f(v̂) = Q(x̄−v̂) (up to sign in gradient)
//   Actually: ∇f(v̂) = Qv̂ + g,  so H₁ condition is −(Qv̂+g)⊤(x*−v̂) ≥ 0
//   margin = −(Qv̂+g)⊤(x*−v̂)  ← must be ≥ 0
static double h1_margin(const Eigen::VectorXd &xstar,
                         const Eigen::VectorXd &vhat,
                         const Eigen::MatrixXd &Q,
                         const Eigen::VectorXd &g)
{
    Eigen::VectorXd grad_vhat = Q * vhat + g;   // ∇f(v̂)
    return -grad_vhat.dot(xstar - vhat);         // −∇f(v̂)⊤(x*−v̂) ≥ 0
}

// Proposition 2 (H₂): margin of x* in H₂
//   H₂ = { x : (x̄−v̂)⊤(x−v̂) ≥ 0 }
//   margin = (x̄−v̂)⊤(x*−v̂)  ← must be ≥ 0
static double h2_margin(const Eigen::VectorXd &xstar,
                         const Eigen::VectorXd &vhat,
                         const Eigen::VectorXd &xbar)
{
    return (xbar - vhat).dot(xstar - vhat);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Problem generators
// ═══════════════════════════════════════════════════════════════════════════
static QPProblem makeBoxQP(int n, std::mt19937 &rng, double scale_g = 5.0,
                            double bound = 1.0)
{
    QPProblem p;
    p.n = n; p.m = 0;
    p.name = "box/denseQ";
    p.H = randomPD(n, rng);
    std::normal_distribution<> d(0.0, 1.0);
    p.g.resize(n);
    for (int i = 0; i < n; ++i) p.g(i) = d(rng) * scale_g;
    p.lbX = -bound * Eigen::VectorXd::Ones(n);
    p.ubX =  bound * Eigen::VectorXd::Ones(n);
    // randomPD eigenvalues are O(n) so x̄ = -Q⁻¹g ≈ O(1/n) for large n;
    // scale g to guarantee x̄ lies outside the box.
    ensure_xbar_infeasible(p.H, p.g, p.lbX, p.ubX);
    p.A = Eigen::MatrixXd(0, n);
    p.b = Eigen::VectorXd(0);
    return p;
}

// Axis-aligned box, diagonal Q
static QPProblem makeDiagBoxQP(int n, std::mt19937 &rng, double scale_g = 5.0,
                                double bound = 1.0)
{
    QPProblem p;
    p.n = n; p.m = 0;
    p.name = "box/diagQ";
    std::uniform_real_distribution<> qd(0.5, 4.0);
    p.H = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) p.H(i, i) = qd(rng);
    std::normal_distribution<> d(0.0, 1.0);
    p.g.resize(n);
    for (int i = 0; i < n; ++i) p.g(i) = d(rng) * scale_g;
    p.lbX = -bound * Eigen::VectorXd::Ones(n);
    p.ubX =  bound * Eigen::VectorXd::Ones(n);
    p.A = Eigen::MatrixXd(0, n);
    p.b = Eigen::VectorXd(0);
    return p;
}

// Axis-aligned box, ill-conditioned dense Q
static QPProblem makeIllCondBoxQP(int n, std::mt19937 &rng, double cond,
                                   double scale_g = 5.0, double bound = 1.0)
{
    QPProblem p;
    p.n = n; p.m = 0;
    p.name = "box/cond=" + std::to_string((int)cond);
    p.H = randomPD_conditioned(n, rng, cond);
    std::normal_distribution<> d(0.0, 1.0);
    p.g.resize(n);
    for (int i = 0; i < n; ++i) p.g(i) = d(rng) * scale_g;
    p.lbX = -bound * Eigen::VectorXd::Ones(n);
    p.ubX =  bound * Eigen::VectorXd::Ones(n);
    p.A = Eigen::MatrixXd(0, n);
    p.b = Eigen::VectorXd(0);
    return p;
}

// Affine image T(B)+c with diagonal T (tᵢ ∈ [0.5, 3]):
//   P = { x : lbX ≤ x ≤ ubX } with lbX = T*ℓ+c, ubX = T*u+c
//   This is still an axis-aligned box in x-space → box fast-path fires.
static QPProblem makeDiagAffineBoxQP(int n, std::mt19937 &rng, double scale_g = 5.0)
{
    QPProblem p;
    p.n = n; p.m = 0;
    p.name = "affine-diag/denseQ";
    p.H = randomPD(n, rng);
    std::normal_distribution<> d(0.0, 1.0);
    std::uniform_real_distribution<> td(0.5, 3.0);
    p.g.resize(n);
    for (int i = 0; i < n; ++i) p.g(i) = d(rng) * scale_g;

    // T = diag(t), c = small offset, ℓ = -1, u = 1
    Eigen::VectorXd t(n), c(n);
    for (int i = 0; i < n; ++i) { t(i) = td(rng); c(i) = d(rng) * 0.5; }
    p.lbX = (-1.0 * t) + c;  // t_i * (-1) + c_i
    p.ubX = (       t) + c;  // t_i *   1  + c_i
    // Same O(n) eigenvalue scaling issue — guarantee x̄ outside the affine box.
    ensure_xbar_infeasible(p.H, p.g, p.lbX, p.ubX);
    p.A = Eigen::MatrixXd(0, n);
    p.b = Eigen::VectorXd(0);
    return p;
}

// Affine image T(B)+c with dense nonsingular T:
//   Uses explicit A/b form → box fast-path does NOT fire.
//   A = [T⁻¹; −T⁻¹], b = [u + d; −ℓ − d]  where d = T⁻¹c
struct DenseAffineInfo
{
    Eigen::MatrixXd T;
    Eigen::VectorXd c;
    Eigen::VectorXd lb_y;  // y-space bounds
    Eigen::VectorXd ub_y;
};

static QPProblem makeDenseAffineBoxQP(int n, std::mt19937 &rng, double scale_g = 5.0,
                                       DenseAffineInfo *info_out = nullptr)
{
    QPProblem p;
    p.n = n; p.m = 2 * n;
    p.name = "affine-dense/denseQ";
    p.H = randomPD(n, rng);

    // Dense T via random orthogonal * random diagonal (guaranteed invertible)
    std::normal_distribution<> d(0.0, 1.0);
    std::uniform_real_distribution<> sd(0.5, 2.0);
    Eigen::MatrixXd R(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            R(i, j) = d(rng);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(R);
    Eigen::MatrixXd Q_orth = qr.householderQ();
    Eigen::VectorXd diag_T(n);
    for (int i = 0; i < n; ++i) diag_T(i) = sd(rng);
    Eigen::MatrixXd T = Q_orth * diag_T.asDiagonal();

    Eigen::VectorXd cv(n);
    for (int i = 0; i < n; ++i) cv(i) = d(rng) * 0.3;

    Eigen::VectorXd lb_y = -Eigen::VectorXd::Ones(n);
    Eigen::VectorXd ub_y =  Eigen::VectorXd::Ones(n);

    if (info_out)
    {
        info_out->T    = T;
        info_out->c    = cv;
        info_out->lb_y = lb_y;
        info_out->ub_y = ub_y;
    }

    // Build A/b from T
    Eigen::MatrixXd M = T.inverse();
    Eigen::VectorXd dv = M * cv;

    Eigen::MatrixXd Amat(2 * n, n);
    Eigen::VectorXd bvec(2 * n);
    Amat.topRows(n)    =  M;
    Amat.bottomRows(n) = -M;
    bvec.head(n) = ub_y + dv;
    bvec.tail(n) = -lb_y - dv;

    p.g.resize(n);
    for (int i = 0; i < n; ++i) p.g(i) = d(rng) * scale_g;
    p.lbX = -1e20 * Eigen::VectorXd::Ones(n);
    p.ubX =  1e20 * Eigen::VectorXd::Ones(n);
    p.A = Amat;
    p.b = bvec;
    return p;
}

static geoqp::Config default_cfg()
{
    geoqp::Config cfg;
    cfg.tolerance = 1e-10;
    cfg.constraint_tolerance = 1e-10;
    cfg.verbosity = -1;
    return cfg;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Reference solver (qpOASES)
// ═══════════════════════════════════════════════════════════════════════════

struct RefSolution
{
    bool success = false;
    Eigen::VectorXd x;
    double obj = 0.0;
};

static RefSolution solveQPOASES(const QPProblem &prob)
{
    RefSolution ref;
    int n = prob.n, m = prob.m;
    std::vector<double> H_vec(n * n), g_vec(n);
    std::vector<double> A_vec(m * n), lbA(std::max(m, 1)), ubA(std::max(m, 1));
    std::vector<double> lbX(n), ubX(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            H_vec[i * n + j] = prob.H(i, j);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A_vec[i * n + j] = prob.A(i, j);
    for (int i = 0; i < n; ++i)
    { g_vec[i] = prob.g(i); lbX[i] = prob.lbX(i); ubX[i] = prob.ubX(i); }
    for (int i = 0; i < m; ++i)
    { lbA[i] = -1e20; ubA[i] = prob.b(i); }

    qpOASES::QProblem qp(n, m, qpOASES::HST_POSDEF);
    qpOASES::Options opts;
    opts.setToMPC();
    opts.printLevel = qpOASES::PL_NONE;
    qp.setOptions(opts);
    int nWSR = 4000;
    auto st = qp.init(H_vec.data(), g_vec.data(),
                      m > 0 ? A_vec.data() : nullptr,
                      lbX.data(), ubX.data(),
                      m > 0 ? lbA.data() : nullptr,
                      m > 0 ? ubA.data() : nullptr, nWSR);
    ref.success = (st == qpOASES::SUCCESSFUL_RETURN);
    if (ref.success)
    {
        std::vector<double> sol(n);
        qp.getPrimalSolution(sol.data());
        ref.x = Eigen::Map<Eigen::VectorXd>(sol.data(), n);
        ref.obj = qp.getObjVal();
    }
    return ref;
}

static double solveQPOASES_time(const QPProblem &prob, bool &success, double &objVal,
                                int *nWSR_out = nullptr)
{
    int n = prob.n, m = prob.m;
    std::vector<double> H_vec(n * n), g_vec(n), A_vec(m * n);
    std::vector<double> lbA(std::max(m,1)), ubA(std::max(m,1)), lbX(n), ubX(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            H_vec[i*n+j] = prob.H(i,j);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A_vec[i*n+j] = prob.A(i,j);
    for (int i = 0; i < n; ++i)
    { g_vec[i] = prob.g(i); lbX[i] = prob.lbX(i); ubX[i] = prob.ubX(i); }
    for (int i = 0; i < m; ++i)
    { lbA[i] = -1e20; ubA[i] = prob.b(i); }

    qpOASES::QProblem qp(n, m, qpOASES::HST_POSDEF);
    qpOASES::Options opts; opts.setToMPC(); opts.printLevel = qpOASES::PL_NONE;
    qp.setOptions(opts);
    int nWSR = 4000;
    auto t0 = Clock::now();
    auto st = qp.init(H_vec.data(), g_vec.data(),
                      m > 0 ? A_vec.data() : nullptr,
                      lbX.data(), ubX.data(),
                      m > 0 ? lbA.data() : nullptr,
                      m > 0 ? ubA.data() : nullptr, nWSR);
    auto t1 = Clock::now();
    success = (st == qpOASES::SUCCESSFUL_RETURN);
    objVal  = success ? qp.getObjVal() : 0.0;
    if (nWSR_out) *nWSR_out = nWSR;
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

#ifdef HAVE_OSQP
// Returns solve time in microseconds.  Sets success=false and returns 0 on
// any setup/solver failure.  objVal is set to the optimal objective.
// osqpIter is set to the number of ADMM iterations taken.
static double solveOSQP_time(const QPProblem &prob, bool &success, double &objVal,
                             int &osqpIter)
{
    int n = prob.n, m = prob.m;
    int total = 0;
    for (int i = 0; i < n; ++i) { if (prob.lbX(i) > -1e19) total++; if (prob.ubX(i) < 1e19) total++; }
    for (int i = 0; i < m; ++i) { if (prob.b(i) < 1e19) total++; }

    Eigen::MatrixXd A_c = Eigen::MatrixXd::Zero(total, n);
    Eigen::VectorXd lb_c(total), ub_c(total);
    int row = 0;
    for (int i = 0; i < n; ++i)
    {
        if (prob.lbX(i) > -1e19)
        { A_c(row, i) = 1.0; lb_c(row) = prob.lbX(i); ub_c(row) = 1e20; row++; }
        if (prob.ubX(i) < 1e19)
        { A_c(row, i) = 1.0; lb_c(row) = -1e20; ub_c(row) = prob.ubX(i); row++; }
    }
    for (int i = 0; i < m; ++i)
    {
        if (prob.b(i) < 1e19)
        { A_c.row(row) = prob.A.row(i); lb_c(row) = -1e20; ub_c(row) = prob.b(i); row++; }
    }

    Eigen::MatrixXd H_upper = prob.H.triangularView<Eigen::Upper>();
    Eigen::SparseMatrix<double> H_sp = H_upper.sparseView();
    Eigen::SparseMatrix<double> A_sp = A_c.sparseView();

    QpSolversEigen::Solver osqp;
    success = osqp.instantiateSolver("osqp");
    if (!success) { osqpIter = 0; objVal = 0.0; return 0.0; }
    osqp.setBooleanParameter("verbose", false);
    osqp.setRealNumberParameter("eps_abs", 1e-10);
    osqp.setRealNumberParameter("eps_rel", 1e-10);
    osqp.data()->setNumberOfVariables(n);
    osqp.data()->setNumberOfInequalityConstraints(total);
    Eigen::VectorXd grad = prob.g;
    success = osqp.data()->setHessianMatrix(H_sp)
           && osqp.data()->setGradient(grad)
           && osqp.data()->setInequalityConstraintsMatrix(A_sp)
           && osqp.data()->setLowerBound(lb_c)
           && osqp.data()->setUpperBound(ub_c);
    if (!success) { osqpIter = 0; objVal = 0.0; return 0.0; }

    auto t0 = Clock::now();
    success = osqp.initSolver();
    if (success)
    {
        auto st = osqp.solveProblem();
        success = (st == QpSolversEigen::ErrorExitFlag::NoError);
    }
    auto t1 = Clock::now();

    // QpSolversEigen wrapper doesn't expose iteration count in a stable way here;
    // set osqpIter=0 for now to avoid calling OsqpEigen internals directly.
    osqpIter = 0;
    objVal = 0.0; // avoid calling getSolution() on OsqpEigen (previously caused heap corruption)
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}
#endif  // HAVE_OSQP

// ═══════════════════════════════════════════════════════════════════════════
//  Print helpers
// ═══════════════════════════════════════════════════════════════════════════

static const int W = 130;

static void print_header(const std::string &s)
{
    std::cout << "\n" << std::string(W, '=') << "\n";
    std::cout << "  " << s << "\n";
    std::cout << std::string(W, '=') << "\n";
}

static void print_subheader(const std::string &s)
{
    std::cout << "\n>>> " << s << "\n";
    std::cout << std::string(W, '-') << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 1:  LEMMA — Antipodal Good Vertex, O(n) Fast-Path
//
//  For each n and Q type the Lemma guarantees:
//    (a) vertices_explored == 1  (O(n) fast-path fired)
//    (b) v̂ is a good vertex  (Definition 1)
//    (c) Each coordinate of v̂ is at the Euclidean-farthest bound from x̄
//        (the antipodal rule; gives the Euclidean-farthest good vertex for boxes)
//    (d) v̂ is the maximum-Euclidean-distance vertex among all good vertices
//        (Proposition 2: v̂ = argmax_{v∈V_good} ‖v−x̄‖₂ for axis-aligned boxes)
// ═══════════════════════════════════════════════════════════════════════════

void run_lemma()
{
    print_header("LEMMA (box_antipodal.tex §Background): Antipodal Good Vertex, O(n) Fast-Path");

    std::cout << std::setw(6)  << "n"
              << std::setw(12) << "Q type"
              << std::setw(8)  << "Verts"
              << std::setw(10) << "FastPath?"
              << std::setw(10) << "IsGood?"
              << std::setw(10) << "CoordOK?"
              << std::setw(12) << "FarthestOK?"
              << std::setw(10) << "Ph2hops"
              << std::setw(12) << "Time(us)"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    std::mt19937 rng(42);
    const int trials = 15;
    bool all_pass = true;

    auto run_one = [&](int n, bool diagonal)
    {
        bool fast_path_ok = true, good_ok = true, coord_ok = true, farthest_ok = true;
        bool solved_ok = true;
        double sum_verts = 0, sum_ph2 = 0, sum_time = 0;
        int ct = 0;

        for (int t = 0; t < trials; ++t)
        {
            QPProblem qp = diagonal ? makeDiagBoxQP(n, rng) : makeBoxQP(n, rng);
            auto gp = to_geoqp(qp);
            geoqp::Solver solver(default_cfg());

            auto t0 = Clock::now();
            auto sol = solver.solve(gp);
            auto t1 = Clock::now();

            if (sol.status != geoqp::Status::SUCCESS)
            { solved_ok = false; continue; }

            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            sum_time += us;
            sum_ph2  += sol.phase2_iterations;

            // Trivial case: xbar is feasible
            if (sol.vertices_explored == 0 && sol.x_phase1.empty()) continue;
            ++ct;

            sum_verts += sol.vertices_explored;

            // (a) Fast-path: exactly 1 vertex explored
            if (sol.vertices_explored != 1) fast_path_ok = false;

            Eigen::VectorXd xbar = compute_xbar(qp.H, qp.g);
            Eigen::VectorXd vhat =
                Eigen::Map<const Eigen::VectorXd>(sol.x_phase1.data(), n);

            // (b) v̂ is a good vertex
            if (!is_good_box_vertex(vhat, xbar, qp.lbX, qp.ubX)) good_ok = false;

            // (c) Each coordinate is at the Euclidean-farthest bound
            for (int j = 0; j < n; ++j)
            {
                double dl = std::abs(xbar(j) - qp.lbX(j));
                double du = std::abs(xbar(j) - qp.ubX(j));
                double expected = (dl >= du) ? qp.lbX(j) : qp.ubX(j);
                if (std::abs(vhat(j) - expected) > 1e-6) coord_ok = false;
            }

            // (d) v̂ is Euclidean-farthest among all good vertices  (n ≤ 20 only)
            if (n <= 20)
            {
                auto all_v = enumerate_box_vertices(qp.lbX, qp.ubX);
                double best = -1.0;
                for (auto &v : all_v)
                    if (is_good_box_vertex(v, xbar, qp.lbX, qp.ubX))
                        best = std::max(best, eucl_dist_sq(v, xbar));
                double vhat_d = eucl_dist_sq(vhat, xbar);
                if (best > 0.0 && vhat_d < 0.99 * best) farthest_ok = false;
            }
        }

        bool row_pass = solved_ok && fast_path_ok && good_ok && coord_ok && farthest_ok;
        if (!row_pass) all_pass = false;

        double inv_t = (double)trials;
        double inv_c = ct > 0 ? 1.0 / ct : 0.0;
        std::cout << std::setw(6) << n
                  << std::setw(12) << (diagonal ? "diagonal" : "dense")
                  << std::setw(8)  << std::fixed << std::setprecision(1)
                     << (sum_verts * inv_c)
                  << std::setw(10) << (fast_path_ok ? "OK" : "FAIL")
                  << std::setw(10) << (good_ok ? "OK" : "FAIL")
                  << std::setw(10) << (coord_ok ? "OK" : "FAIL")
                  << std::setw(12) << (n <= 20 ? (farthest_ok ? "OK" : "FAIL") : "n/a")
                  << std::setw(10) << std::setprecision(1) << (sum_ph2 / inv_t)
                  << std::setw(12) << std::setprecision(1) << (sum_time / inv_t)
                  << "\n";
    };

    for (int n : {4, 6, 8, 10, 12, 16, 20, 30, 50, 100, 200})
    {
        run_one(n, true);
        run_one(n, false);
    }

    std::cout << "\nLemma overall: " << (all_pass ? "PASS" : "** FAIL **") << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 2:  PROPOSITION 1 — x* ∈ P₊  (Half-Box Containment)
//
//  P₊ = { x ∈ P : v⊤(x−c₀) ≥ 0 }
//  v   = x̄ − xₑ  (Euclidean projection direction)
//  c₀  = (lb+ub)/2  (box centre)
//
//  Reports the raw margin v⊤(x*−c₀) and its ratio to ‖v‖·‖x*−c₀‖
//  (should always be ≥ 0 for any x̄ outside P).
// ═══════════════════════════════════════════════════════════════════════════

void run_proposition1()
{
    print_header("PROPOSITION 1 (prop:halfbox_affine): x* ∈ P₊  (v = x̄−xₑ, c₀ = box centre)");

    std::cout << std::setw(6)   << "n"
              << std::setw(12)  << "Q type"
              << std::setw(10)  << "scale_g"
              << std::setw(14)  << "avg P+ margin"
              << std::setw(12)  << "min P+ margin"
              << std::setw(10)  << "avg ‖v‖"
              << std::setw(10)  << "avg L"
              << std::setw(10)  << "#Solved"
              << std::setw(10)  << "Result"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    std::mt19937 rng(77);
    const int trials = 20;
    bool all_pass = true;

    auto check = [&](int n, bool diagonal, double scale_g)
    {
        double sum_mg = 0, min_mg = 1e30, sum_v = 0, sum_L = 0;
        bool ok = true;
        int ct = 0;

        for (int t = 0; t < trials; ++t)
        {
            QPProblem qp = diagonal ? makeDiagBoxQP(n, rng, scale_g)
                                    : makeBoxQP(n, rng, scale_g);
            auto gp  = to_geoqp(qp);
            geoqp::Solver solver(default_cfg());
            auto sol = solver.solve(gp);

            if (sol.status != geoqp::Status::SUCCESS) continue;
            if (sol.vertices_explored == 0 && sol.x_phase1.empty()) continue;
            ++ct;

            Eigen::VectorXd xbar = compute_xbar(qp.H, qp.g);
            Eigen::VectorXd xstar(n);
            for (int j = 0; j < n; ++j) xstar(j) = sol.x[j];

            double mg = pplus_margin(xstar, xbar, qp.lbX, qp.ubX);
            sum_mg += mg;
            min_mg  = std::min(min_mg, mg);
            if (mg < -1e-6) ok = false;

            Eigen::VectorXd xe = project_onto_box(xbar, qp.lbX, qp.ubX);
            sum_v += (xbar - xe).norm();
            sum_L += (xbar - xstar).norm();  // rough ‖L‖ = distance to solution
        }

        if (!ok) all_pass = false;
        double inv = ct > 0 ? 1.0 / ct : 0.0;
        std::cout << std::setw(6)   << n
                  << std::setw(12)  << (diagonal ? "diagonal" : "dense")
                  << std::setw(10)  << std::fixed << std::setprecision(1) << scale_g
                  << std::setw(14)  << std::scientific << std::setprecision(2)
                     << (sum_mg * inv)
                  << std::setw(12)  << min_mg
                  << std::setw(10)  << std::fixed << std::setprecision(2)
                     << (sum_v * inv)
                  << std::setw(10)  << (sum_L * inv)
                  << std::setw(10)  << ct
                  << std::setw(10)  << (ok ? "PASS" : "FAIL")
                  << "\n";
    };

    print_subheader("Axis-aligned box, diagonal Q");
    for (int n : {4, 8, 16, 30, 50, 100})
    {
        check(n, true, 2.0);
        check(n, true, 10.0);
        check(n, true, 50.0);
    }

    print_subheader("Axis-aligned box, dense Q");
    for (int n : {4, 8, 16, 30, 50, 100})
    {
        check(n, false, 2.0);
        check(n, false, 10.0);
        check(n, false, 50.0);
    }

    std::cout << "\nProposition 1 overall: " << (all_pass ? "PASS" : "** FAIL **") << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 3:  PROPOSITION 2 — x* ∈ P₊ ∩ H₁ ∩ H₂
//
//  v̂ = Euclidean-farthest good vertex (from Phase I, sol.x_phase1).
//  H₁ = { x : −∇f(v̂)⊤(x−v̂) ≥ 0 }  ← convexity + optimality of f at v̂
//  H₂ = { x : (x̄−v̂)⊤(x−v̂) ≥ 0 }  ← v̂ is farthest from x̄ in V_good
//
//  All three margins must be ≥ 0. The H₂ margin satisfies the stronger bound
//  (from the proof): (x̄−v̂)⊤(x*−v̂) ≥ ½‖x*−v̂‖².
// ═══════════════════════════════════════════════════════════════════════════

void run_proposition2()
{
    print_header("PROPOSITION 2 (prop:three_halfspaces): x* ∈ P₊ ∩ H₁ ∩ H₂");

    std::cout << std::setw(6)   << "n"
              << std::setw(12)  << "Q type"
              << std::setw(14)  << "P+ margin"
              << std::setw(14)  << "H1 margin"
              << std::setw(14)  << "H2 margin"
              << std::setw(14)  << "H2≥½‖x*−v̂‖²?"
              << std::setw(10)  << "min P+"
              << std::setw(10)  << "min H1"
              << std::setw(10)  << "min H2"
              << std::setw(10)  << "Result"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    std::mt19937 rng(123);
    const int trials = 20;
    bool all_pass = true;

    auto check = [&](int n, bool diagonal)
    {
        double sum_pp = 0, sum_h1 = 0, sum_h2 = 0;
        double min_pp = 1e30, min_h1 = 1e30, min_h2 = 1e30;
        bool h2_bound_ok = true;
        int ct = 0;
        bool row_ok = true;

        for (int t = 0; t < trials; ++t)
        {
            QPProblem qp = diagonal ? makeDiagBoxQP(n, rng, 8.0)
                                    : makeBoxQP(n, rng, 8.0);
            auto gp = to_geoqp(qp);
            geoqp::Solver solver(default_cfg());
            auto sol = solver.solve(gp);

            if (sol.status != geoqp::Status::SUCCESS) continue;
            if (sol.x_phase1.empty()) continue;  // xbar feasible → trivial
            ++ct;

            Eigen::VectorXd xbar = compute_xbar(qp.H, qp.g);
            Eigen::VectorXd xstar(n), vhat(n);
            for (int j = 0; j < n; ++j)
            {
                xstar(j) = sol.x[j];
                vhat(j)  = sol.x_phase1[j];
            }

            double pp = pplus_margin(xstar, xbar, qp.lbX, qp.ubX);
            double h1 = h1_margin(xstar, vhat, qp.H, qp.g);
            double h2 = h2_margin(xstar, vhat, xbar);

            sum_pp += pp; min_pp = std::min(min_pp, pp);
            sum_h1 += h1; min_h1 = std::min(min_h1, h1);
            sum_h2 += h2; min_h2 = std::min(min_h2, h2);

            if (pp < -1e-6) row_ok = false;
            if (h1 < -1e-6) row_ok = false;
            if (h2 < -1e-6) row_ok = false;

            // Verify the stronger H₂ bound from the proof:
            //   (x̄−v̂)⊤(x*−v̂) ≥ ½‖x*−v̂‖²
            double half_sq = 0.5 * (xstar - vhat).squaredNorm();
            if (h2 < half_sq - 1e-6) h2_bound_ok = false;
        }

        if (!row_ok) all_pass = false;
        double inv = ct > 0 ? 1.0 / ct : 0.0;
        std::cout << std::setw(6)   << n
                  << std::setw(12)  << (diagonal ? "diagonal" : "dense")
                  << std::setw(14)  << std::scientific << std::setprecision(2)
                     << (sum_pp * inv)
                  << std::setw(14)  << (sum_h1 * inv)
                  << std::setw(14)  << (sum_h2 * inv)
                  << std::setw(14)  << (h2_bound_ok ? "OK" : "FAIL")
                  << std::setw(10)  << std::setprecision(1) << min_pp
                  << std::setw(10)  << min_h1
                  << std::setw(10)  << min_h2
                  << std::setw(10)  << (row_ok ? "PASS" : "FAIL")
                  << "\n";
    };

    for (int n : {4, 6, 8, 10, 12, 16, 20, 30, 50, 100})
    {
        check(n, true);
        check(n, false);
    }

    std::cout << "\nProposition 2 overall: " << (all_pass ? "PASS" : "** FAIL **") << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 4:  Phase II Hop Analysis
//
//  Since x* ∈ P₊ ∩ H₁ ∩ H₂ (Proposition 2), and Phase I already places v̂
//  at the farthest good vertex, the H₁/H₂ guided violated-constraint filter
//  should result in very few Phase II hops in practice.
//
//  Reports: avg hops, max hops, adjacency bound 2(2n−k), us per hop, and
//  how phase II cost scales with n vs phase I and phase III.
// ═══════════════════════════════════════════════════════════════════════════

void run_hop_analysis()
{
    print_header("PHASE II HOP ANALYSIS: H₁/H₂ Guided Convergence");

    // (a) Hop count vs n and Q type
    print_subheader("(a) Hop count vs n — box QP (diag and dense Q)");
    std::cout << std::setw(6)   << "n"
              << std::setw(12)  << "Q type"
              << std::setw(10)  << "AvgPivot"
              << std::setw(10)  << "MaxPivot"
              << std::setw(10)  << "AvgHops"
              << std::setw(10)  << "MaxHops"
              << std::setw(12)  << "HopBound"
              << std::setw(12)  << "BoundOK?"
              << std::setw(12)  << "Ph1/Ph2%"
              << std::setw(12)  << "Time(us)"
              << std::setw(10)  << "#Solved"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    std::mt19937 rng(99);
    const int trials = 30;
    bool all_pass = true;

    auto check_hops = [&](int n, bool diagonal)
    {
        double sum_pivots = 0, sum_hops = 0, sum_time = 0;
        double sum_ph1us = 0, sum_ph2us = 0;
        int max_pivots = 0, max_hops = 0;
        bool bounded = true;
        int ct = 0;

        for (int t = 0; t < trials; ++t)
        {
            QPProblem qp = diagonal ? makeDiagBoxQP(n, rng) : makeBoxQP(n, rng);
            auto gp = to_geoqp(qp);
            geoqp::Solver solver(default_cfg());

            auto t0 = Clock::now();
            auto sol = solver.solve(gp);
            auto t1 = Clock::now();

            if (sol.status != geoqp::Status::SUCCESS) continue;
            if (sol.active_set.empty()) continue;
            ++ct;

            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            sum_time   += us;
            sum_pivots += sol.phase2_iterations;
            max_pivots  = std::max(max_pivots, sol.phase2_iterations);
            sum_hops   += sol.phase2_facet_hops;
            max_hops    = std::max(max_hops, sol.phase2_facet_hops);
            sum_ph1us  += sol.phase1_time * 1e6;
            sum_ph2us  += sol.phase2_time * 1e6;

            // Corollary bound: ≤ 2 facet-adjacency hops (1 to F₀, ≤1 from F₀ to F★)
            if (sol.phase2_facet_hops > 2) bounded = false;
        }

        if (!bounded) all_pass = false;
        double inv = ct > 0 ? 1.0 / ct : 0.0;
        double ph1pct = (sum_ph1us + sum_ph2us) > 0
            ? 100.0 * sum_ph1us / (sum_ph1us + sum_ph2us) : 0.0;

        std::cout << std::setw(6)   << n
                  << std::setw(12)  << (diagonal ? "diagonal" : "dense")
                  << std::setw(10)  << std::fixed << std::setprecision(1)
                     << (sum_pivots * inv)
                  << std::setw(10)  << max_pivots
                  << std::setw(10)  << std::setprecision(1)
                     << (sum_hops * inv)
                  << std::setw(10)  << max_hops
                  << std::setw(12)  << "<=2"
                  << std::setw(12)  << (bounded ? "OK" : "FAIL")
                  << std::setw(11)  << std::setprecision(0) << ph1pct << "%"
                  << std::setw(12)  << std::setprecision(1) << (sum_time * inv)
                  << std::setw(10)  << ct
                  << "\n";
    };

    for (int n : {4, 6, 8, 10, 16, 20, 30, 50, 75, 100, 150, 200})
    {
        check_hops(n, true);
        check_hops(n, false);
    }

    // (b) Separation (scale_g) sweep — how does L affect hop count?
    print_subheader("(b) Separation sweep (n=10 dense Q): larger g → larger L → fewer hops expected");
    std::cout << std::setw(10)  << "scale_g"
              << std::setw(10)  << "avg ‖v‖"
              << std::setw(10)  << "AvgPivot"
              << std::setw(10)  << "MaxPivot"
              << std::setw(10)  << "AvgHops"
              << std::setw(10)  << "MaxHops"
              << std::setw(10)  << "Ph2(us)"
              << std::setw(10)  << "#Solved"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    for (double sg : {0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0})
    {
        std::mt19937 rng2(42);
        int n = 10;
        double sum_v = 0, sum_p = 0, sum_h = 0, sum_ph2 = 0;
        int mp = 0, mh = 0, ct = 0;
        for (int t = 0; t < 30; ++t)
        {
            QPProblem qp = makeBoxQP(n, rng2, sg);
            auto gp = to_geoqp(qp);
            geoqp::Solver solver(default_cfg());
            auto sol = solver.solve(gp);
            if (sol.status != geoqp::Status::SUCCESS) continue;
            if (sol.x_phase1.empty()) continue;
            ++ct;
            Eigen::VectorXd xbar = compute_xbar(qp.H, qp.g);
            Eigen::VectorXd xe   = project_onto_box(xbar, qp.lbX, qp.ubX);
            sum_v   += (xbar - xe).norm();
            sum_p   += sol.phase2_iterations;
            mp = std::max(mp, sol.phase2_iterations);
            sum_h   += sol.phase2_facet_hops;
            mh = std::max(mh, sol.phase2_facet_hops);
            sum_ph2 += sol.phase2_time * 1e6;
        }
        double inv = ct > 0 ? 1.0 / ct : 0.0;
        std::cout << std::setw(10) << std::fixed << std::setprecision(1) << sg
                  << std::setw(10) << std::setprecision(2) << (sum_v * inv)
                  << std::setw(10) << std::setprecision(1) << (sum_p * inv)
                  << std::setw(10) << mp
                  << std::setw(10) << std::setprecision(1) << (sum_h * inv)
                  << std::setw(10) << mh
                  << std::setw(10) << std::setprecision(1) << (sum_ph2 * inv)
                  << std::setw(10) << ct
                  << "\n";
    }

    // (c) Condition number sweep — ill-conditioned Q
    print_subheader("(c) Q condition number sweep (n=15, dense Q)");
    std::cout << std::setw(12)  << "cond(Q)"
              << std::setw(10)  << "AvgPivot"
              << std::setw(10)  << "MaxPivot"
              << std::setw(10)  << "AvgHops"
              << std::setw(10)  << "MaxHops"
              << std::setw(12)  << "ObjDiff"
              << std::setw(10)  << "Solved"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    for (double cond : {1.0, 10.0, 1e2, 1e3, 1e4, 1e6})
    {
        std::mt19937 rng3(55);
        int n = 15;
        double sp = 0, sh = 0, sod = 0;
        int mp = 0, mh = 0, ct = 0;

        for (int t = 0; t < 20; ++t)
        {
            QPProblem qp = makeIllCondBoxQP(n, rng3, cond, 5.0);
            auto gp = to_geoqp(qp);
            geoqp::Solver solver(default_cfg());
            auto sol = solver.solve(gp);
            if (sol.status != geoqp::Status::SUCCESS) continue;
            ++ct;
            sp += sol.phase2_iterations;
            mp = std::max(mp, sol.phase2_iterations);
            sh += sol.phase2_facet_hops;
            mh = std::max(mh, sol.phase2_facet_hops);

            // Diagnostic: dump problem when hops > 2
            if (sol.phase2_facet_hops > 2)
            {
                std::cout << "\n*** HOP BOUND VIOLATION: cond=" << cond
                          << " trial=" << t
                          << " pivots=" << sol.phase2_iterations
                          << " hops=" << sol.phase2_facet_hops
                          << " active_set_size=" << sol.active_set.size()
                          << "\n";
                std::cout << "    Q (upper triangle):\n    ";
                for (int r = 0; r < n; ++r)
                    for (int c2 = r; c2 < n; ++c2)
                        std::cout << std::setprecision(6) << qp.H(r, c2) << " ";
                std::cout << "\n    g: ";
                for (int i = 0; i < n; ++i) std::cout << qp.g(i) << " ";
                std::cout << "\n    lb: ";
                for (int i = 0; i < n; ++i) std::cout << qp.lbX(i) << " ";
                std::cout << "\n    ub: ";
                for (int i = 0; i < n; ++i) std::cout << qp.ubX(i) << " ";
                std::cout << "\n    x*: ";
                for (int i = 0; i < n; ++i) std::cout << sol.x[i] << " ";
                std::cout << "\n    active_set: ";
                for (int idx : sol.active_set) std::cout << idx << " ";
                std::cout << "\n";
            }

            auto ref = solveQPOASES(qp);
            if (ref.success)
                sod += std::abs(sol.objective_value - ref.obj);
        }
        double inv = ct > 0 ? 1.0 / ct : 0.0;
        std::cout << std::setw(12) << std::scientific << std::setprecision(0) << cond
                  << std::setw(10) << std::fixed << std::setprecision(1) << (sp * inv)
                  << std::setw(10) << mp
                  << std::setw(10) << std::setprecision(1) << (sh * inv)
                  << std::setw(10) << mh
                  << std::setw(12) << std::scientific << std::setprecision(1)
                     << (sod * inv)
                  << std::setw(10) << ct
                  << "\n";
    }

    std::cout << "\nHop analysis overall: " << (all_pass ? "PASS" : "** FAIL **") << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 5:  AFFINE IMAGE T(B)+c
//
//  The paper's theory holds for any nonsingular T.  We verify:
//    (a) Diagonal T  → bounds remain axis-aligned in x-space;
//                      box fast-path fires; Props 1 and 2 hold.
//    (b) Dense T     → explicit A/b form; fast-path does not fire (BFS);
//                      but GeoQP still finds the correct solution (obj check).
// ═══════════════════════════════════════════════════════════════════════════

void run_affine_image()
{
    print_header("AFFINE IMAGE T(B)+c: Theory Verification");

    // (a) Diagonal T
    print_subheader("(a) Diagonal T: fast-path fires, Props 1&2 hold");
    std::cout << std::setw(6)   << "n"
              << std::setw(10)  << "Verts"
              << std::setw(10)  << "FastPath?"
              << std::setw(14)  << "P+ margin"
              << std::setw(14)  << "H1 margin"
              << std::setw(14)  << "H2 margin"
              << std::setw(12)  << "ObjDiff"
              << std::setw(10)  << "Result"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    std::mt19937 rng_a(77);
    const int trials_a = 20;
    bool all_pass = true;

    for (int n : {4, 6, 8, 12, 16, 20, 30, 50, 100})
    {
        double sum_v = 0, sum_pp = 0, sum_h1 = 0, sum_h2 = 0, sum_od = 0;
        bool fp_ok = true, pp_ok = true, h1_ok = true, h2_ok = true;
        int ct = 0;

        for (int t = 0; t < trials_a; ++t)
        {
            // Diagonal-T affine image is still axis-aligned in x-space
            QPProblem qp = makeDiagAffineBoxQP(n, rng_a);
            auto gp = to_geoqp(qp);
            geoqp::Solver solver(default_cfg());
            auto sol = solver.solve(gp);

            if (sol.status != geoqp::Status::SUCCESS) continue;
            if (sol.x_phase1.empty()) continue;
            ++ct;

            sum_v += sol.vertices_explored;
            if (sol.vertices_explored != 1) fp_ok = false;

            Eigen::VectorXd xbar = compute_xbar(qp.H, qp.g);
            Eigen::VectorXd xstar(n), vhat(n);
            for (int j = 0; j < n; ++j)
            { xstar(j) = sol.x[j]; vhat(j) = sol.x_phase1[j]; }

            double pp = pplus_margin(xstar, xbar, qp.lbX, qp.ubX);
            double h1 = h1_margin(xstar, vhat, qp.H, qp.g);
            double h2 = h2_margin(xstar, vhat, xbar);

            sum_pp += pp; sum_h1 += h1; sum_h2 += h2;
            if (pp < -1e-6) pp_ok = false;
            if (h1 < -1e-6) h1_ok = false;
            if (h2 < -1e-6) h2_ok = false;

            auto ref = solveQPOASES(qp);
            if (ref.success)
                sum_od += std::abs(sol.objective_value - ref.obj);
        }

        bool row_ok = fp_ok && pp_ok && h1_ok && h2_ok;
        if (!row_ok) all_pass = false;
        double inv = ct > 0 ? 1.0 / ct : 0.0;
        std::cout << std::setw(6)  << n
                  << std::setw(10) << std::fixed << std::setprecision(1) << (sum_v * inv)
                  << std::setw(10) << (fp_ok ? "OK" : "FAIL")
                  << std::setw(14) << std::scientific << std::setprecision(2) << (sum_pp * inv)
                  << std::setw(14) << (sum_h1 * inv)
                  << std::setw(14) << (sum_h2 * inv)
                  << std::setw(12) << (sum_od * inv)
                  << std::setw(10) << (row_ok ? "PASS" : "FAIL")
                  << "\n";
    }

    // (b) Dense T — no box fast-path, but correct solution
    print_subheader("(b) Dense T: no fast-path (BFS Phase I), correct objective");
    std::cout << std::setw(6)   << "n"
              << std::setw(12)  << "AvgVerts"
              << std::setw(12)  << "Ph1iters"
              << std::setw(10)  << "Pivots"
              << std::setw(10)  << "Hops"
              << std::setw(12)  << "Total(us)"
              << std::setw(12)  << "ObjDiff"
              << std::setw(10)  << "#Solved"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    std::mt19937 rng_b(88);
    const int trials_b = 15;

    // Dense T BFS is O(2^n) worst-case; keep n small
    for (int n : {3, 4, 5, 6, 7, 8, 10, 12})
    {
        double sv = 0, sp1 = 0, sp2 = 0, sh = 0, stime = 0, sod = 0;
        int ct = 0, ne = 0;

        for (int t = 0; t < trials_b; ++t)
        {
            QPProblem qp = makeDenseAffineBoxQP(n, rng_b);
            auto gp = to_geoqp(qp);
            geoqp::Solver solver(default_cfg());

            auto t0 = Clock::now();
            auto sol = solver.solve(gp);
            auto t1 = Clock::now();

            if (sol.status != geoqp::Status::SUCCESS) { ne++; continue; }
            ++ct;
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            sv    += sol.vertices_explored;
            sp1   += sol.phase1_iterations;
            sp2   += sol.phase2_iterations;
            sh    += sol.phase2_facet_hops;
            stime += us;

            auto ref = solveQPOASES(qp);
            if (ref.success)
                sod += std::abs(sol.objective_value - ref.obj);
        }

        double inv = ct > 0 ? 1.0 / ct : 0.0;
        std::cout << std::setw(6)  << n
                  << std::setw(12) << std::fixed << std::setprecision(1) << (sv * inv)
                  << std::setw(12) << (sp1 * inv)
                  << std::setw(10) << (sp2 * inv)
                  << std::setw(10) << (sh * inv)
                  << std::setw(12) << (stime * inv)
                  << std::setw(12) << std::scientific << std::setprecision(1)
                     << (sod * inv)
                  << std::setw(10) << ct
                  << (ne > 0 ? "  (" + std::to_string(ne) + " err)" : "")
                  << "\n";
    }

    std::cout << "\nAffine image overall: " << (all_pass ? "PASS" : "** FAIL **") << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 6:  TIMING BREAKDOWN  — Phase-level cost on box QPs
// ═══════════════════════════════════════════════════════════════════════════

void run_timing_breakdown()
{
    print_header("TIMING BREAKDOWN: Phase-Level Cost on Box QPs");

    std::cout << std::setw(6)   << "n"
              << std::setw(12)  << "Q type"
              << std::setw(12)  << "Total(us)"
              << std::setw(10)  << "Ph1(us)"
              << std::setw(10)  << "Ph2(us)"
              << std::setw(10)  << "Ph3(us)"
              << std::setw(8)   << "Ph1%"
              << std::setw(8)   << "Ph2%"
              << std::setw(8)   << "Ph3%"
              << std::setw(8)   << "Pivots"
              << std::setw(8)   << "Hops"
              << std::setw(10)  << "us/pvt"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    std::mt19937 rng(42);
    const int trials = 20;

    auto profile = [&](int n, bool diagonal)
    {
        double ttot = 0, tph1 = 0, tph2 = 0, tph3 = 0, pivots = 0, hops = 0;
        int solved = 0;

        for (int t = 0; t < trials; ++t)
        {
            QPProblem qp = diagonal ? makeDiagBoxQP(n, rng) : makeBoxQP(n, rng);
            auto gp = to_geoqp(qp);
            geoqp::Solver solver(default_cfg());

            auto t0 = Clock::now();
            auto sol = solver.solve(gp);
            auto t1 = Clock::now();
            if (sol.status != geoqp::Status::SUCCESS) continue;
            solved++;

            double tot = std::chrono::duration<double, std::micro>(t1 - t0).count();
            ttot += tot;
            tph1 += sol.phase1_time * 1e6;
            tph2 += sol.phase2_time * 1e6;
            tph3 += sol.phase3_time * 1e6;
            pivots += sol.phase2_iterations;
            hops   += sol.phase2_facet_hops;
        }
        if (!solved) return;
        double inv = 1.0 / solved;
        double at = ttot * inv, a1 = tph1 * inv, a2 = tph2 * inv, a3 = tph3 * inv;
        double ap = pivots * inv, ah = hops * inv;
        std::cout << std::setw(6)  << n
                  << std::setw(12) << (diagonal ? "diagonal" : "dense")
                  << std::setw(12) << std::fixed << std::setprecision(1) << at
                  << std::setw(10) << a1
                  << std::setw(10) << a2
                  << std::setw(10) << a3
                  << std::setw(7)  << std::setprecision(0)
                     << (at > 0 ? 100.0 * a1 / at : 0) << "%"
                  << std::setw(7)  << (at > 0 ? 100.0 * a2 / at : 0) << "%"
                  << std::setw(7)  << (at > 0 ? 100.0 * a3 / at : 0) << "%"
                  << std::setw(8)  << std::setprecision(1) << ap
                  << std::setw(8)  << ah
                  << std::setw(10) << (ap > 0.01 ? a2 / ap : 0.0)
                  << "\n";
    };

    print_subheader("Diagonal Q (box fast-path + diagonal projection)");
    for (int n : {5, 10, 20, 30, 50, 75, 100, 200})
        profile(n, true);

    print_subheader("Dense Q (box fast-path + dense Schur projection)");
    for (int n : {5, 10, 20, 30, 50, 75, 100, 200})
        profile(n, false);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 7:  PERFORMANCE vs qpOASES
// ═══════════════════════════════════════════════════════════════════════════

void run_performance()
{
    print_header("PERFORMANCE: GeoQP vs qpOASES on Box QPs");

    std::cout << std::setw(6)   << "n"
              << std::setw(16)  << "type"
              << std::setw(10)  << "Geo(us)"
              << std::setw(12)  << "qpOAS(us)"
#ifdef HAVE_OSQP
              << std::setw(12)  << "OSQP(us)"
#endif
              << std::setw(10)  << "Geo/qpO"
              << std::setw(10)  << "AvgPivot"
              << std::setw(10)  << "qpO_iter"
#ifdef HAVE_OSQP
              << std::setw(10)  << "OSQP_iter"
#endif
              << std::setw(10)  << "AvgHops"
              << std::setw(12)  << "ObjDiff"
              << std::setw(10)  << "Result"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    std::mt19937 rng(42);
    const int trials = 10;

    auto bench = [&](int n, const std::string &label,
                      std::function<QPProblem(int, std::mt19937 &)> gen)
    {
        double tg = 0, tq = 0, pivots = 0, hops = 0, max_od = 0;
        int geo_ok = 0, qpo_iters = 0;
#ifdef HAVE_OSQP
        double to = 0;
        int osqp_iters = 0;
#endif

        for (int t = 0; t < trials; ++t)
        {
            QPProblem qp = gen(n, rng);
            auto gp = to_geoqp_explicit(qp);
            geoqp::Solver solver(default_cfg());

            auto t0 = Clock::now();
            auto sol = solver.solve(gp);
            auto t1 = Clock::now();
            tg += std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (sol.status == geoqp::Status::SUCCESS) geo_ok++;
            pivots += sol.phase2_iterations;
            hops   += sol.phase2_facet_hops;

            bool success; double qobj; int nwsr = 0;
            tq += solveQPOASES_time(qp, success, qobj, &nwsr);
            qpo_iters += nwsr;
            if (sol.status == geoqp::Status::SUCCESS && success)
                max_od = std::max(max_od, std::abs(sol.objective_value - qobj));

#ifdef HAVE_OSQP
            bool osqp_ok; double osqp_obj; int osqp_iter_t = 0;
            to += solveOSQP_time(qp, osqp_ok, osqp_obj, osqp_iter_t);
            osqp_iters += osqp_iter_t;
#endif
        }

        double inv = 1.0 / trials;
        double ratio = (tq > 0 && tg > 0) ? tg / tq : 0.0;
        std::string res = std::to_string(geo_ok) + "/" + std::to_string(trials);
        std::cout << std::setw(6)  << n
                  << std::setw(16) << label
                  << std::setw(10) << std::fixed << std::setprecision(1) << (tg * inv)
                  << std::setw(12) << (tq * inv)
#ifdef HAVE_OSQP
                  << std::setw(12) << (to * inv)
#endif
                  << std::setw(9)  << std::setprecision(3) << ratio << "x"
                  << std::setw(10) << std::setprecision(1) << (pivots * inv)
                  << std::setw(10) << std::setprecision(1) << (qpo_iters * inv)
#ifdef HAVE_OSQP
                  << std::setw(10) << std::setprecision(1) << (osqp_iters * inv)
#endif
                  << std::setw(10) << (hops * inv)
                  << std::setw(12) << std::scientific << std::setprecision(1) << max_od
                  << std::setw(10) << res
                  << "\n";
    };

    print_subheader("Axis-aligned, diagonal Q");
    for (int n : {5, 10, 20, 30, 50, 75, 100, 150, 200})
        bench(n, "box/diagQ", [](int n, std::mt19937 &r){ return makeDiagBoxQP(n, r); });

    print_subheader("Axis-aligned, dense Q");
    for (int n : {5, 10, 20, 30, 50, 75, 100, 150, 200})
        bench(n, "box/denseQ", [](int n, std::mt19937 &r){ return makeBoxQP(n, r); });

    print_subheader("Diagonal-T affine image, dense Q");
    for (int n : {5, 10, 20, 30, 50, 75, 100})
        bench(n, "affine-diag/Q", [](int n, std::mt19937 &r){ return makeDiagAffineBoxQP(n, r); });

    print_subheader("Ill-conditioned Q (cond=10^4) — box");
    for (int n : {5, 10, 20, 30, 50})
        bench(n, "box/cond=1e4",
              [](int n, std::mt19937 &r){ return makeIllCondBoxQP(n, r, 1e4); });
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 8:  HOTSTART — Fixed Q, Repeated Solves with Varying g
// ═══════════════════════════════════════════════════════════════════════════

void run_hotstart()
{
    print_header("HOTSTART: Fixed Q, Varying g (MPC-style)");

    std::cout << std::setw(6)   << "n"
              << std::setw(12)  << "Q type"
              << std::setw(12)  << "Geo(us)"
              << std::setw(13)  << "qpOAS(us)"
              << std::setw(10)  << "Geo/qpO"
              << std::setw(10)  << "AvgPivot"
              << std::setw(10)  << "AvgHops"
              << std::setw(12)  << "ObjDiff"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    std::mt19937 rng(42);
    const int hs_trials = 50;

    auto run_hs = [&](int n, bool diagonal)
    {
        QPProblem structure = diagonal ? makeDiagBoxQP(n, rng) : makeBoxQP(n, rng);

        // GeoQP hotstart setup  — use explicit A/b so setup() API can work
        int n_lb = n, n_ub = n;
        int m_exp = n_lb + n_ub;
        Eigen::MatrixXd A_exp = Eigen::MatrixXd::Zero(m_exp, n);
        Eigen::VectorXd b_exp(m_exp);
        for (int i = 0; i < n; ++i)
        { A_exp(i, i) = 1.0; b_exp(i) = structure.ubX(i); }
        for (int i = 0; i < n; ++i)
        { A_exp(n + i, i) = -1.0; b_exp(n + i) = -structure.lbX(i); }

        std::vector<double> Q_flat = eigen_to_flat(structure.H);
        std::vector<double> A_flat = eigen_to_flat(A_exp);
        std::vector<double> b_flat = eigen_to_vec(b_exp);

        geoqp::Config cfg = default_cfg();
        geoqp::Solver geo_solver(cfg);
        geo_solver.setup(n, m_exp, Q_flat.data(), A_flat.data());

        // qpOASES: init with first g, then hotstart
        std::vector<double> H_vec(n * n), g0(n), lbX(n), ubX(n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                H_vec[i*n+j] = structure.H(i,j);
        for (int i = 0; i < n; ++i)
        { g0[i] = structure.g(i); lbX[i] = structure.lbX(i); ubX[i] = structure.ubX(i); }
        qpOASES::QProblem qpoases_qp(n, 0, qpOASES::HST_POSDEF);
        qpOASES::Options opts; opts.setToMPC(); opts.printLevel = qpOASES::PL_NONE;
        qpoases_qp.setOptions(opts);
        int nWSR0 = 2000;
        qpoases_qp.init(H_vec.data(), g0.data(), nullptr,
                        lbX.data(), ubX.data(), nullptr, nullptr, nWSR0);

        double tg = 0, tq = 0, pivots = 0, hops = 0, sod = 0;
        bool geo_ok = true;
        std::normal_distribution<> gdist(0.0, 5.0);
        std::vector<double> g_flat(n);

        for (int t = 0; t < hs_trials; ++t)
        {
            for (int i = 0; i < n; ++i) g_flat[i] = gdist(rng);

            auto t0 = Clock::now();
            auto sol = geo_solver.solve_hotstart(g_flat.data(), b_flat.data());
            auto t1 = Clock::now();
            tg += std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (sol.status != geoqp::Status::SUCCESS) geo_ok = false;
            pivots += sol.phase2_iterations;
            hops   += sol.phase2_facet_hops;
            double geo_obj = sol.objective_value;

            int nWSR_hs = 2000;
            auto t0q = Clock::now();
            qpoases_qp.hotstart(g_flat.data(), lbX.data(), ubX.data(),
                                nullptr, nullptr, nWSR_hs);
            auto t1q = Clock::now();
            tq += std::chrono::duration<double, std::micro>(t1q - t0q).count();
            sod += std::abs(geo_obj - qpoases_qp.getObjVal());
        }

        double inv = 1.0 / hs_trials;
        double ratio = (tq > 0 && tg > 0) ? tg / tq : 0.0;
        std::cout << std::setw(6)  << n
                  << std::setw(12) << (diagonal ? "diagonal" : "dense")
                  << std::setw(12) << std::fixed << std::setprecision(2) << (tg * inv)
                  << std::setw(13) << (tq * inv)
                  << std::setw(9)  << std::setprecision(3) << ratio << "x"
                  << std::setw(10) << std::setprecision(1) << (pivots * inv)
                  << std::setw(10) << (hops * inv)
                  << std::setw(12) << std::scientific << std::setprecision(1)
                     << (sod * inv)
                  << (geo_ok ? "" : "  [ERR]")
                  << "\n";
    };

    for (int n : {5, 10, 20, 30, 50, 75, 100})
    {
        run_hs(n, true);
        run_hs(n, false);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main()
{
    std::cout << std::string(W, '=') << "\n";
    std::cout << "  GeoQP BOX BENCHMARK — following box_antipodal.tex\n";
    std::cout << "  Lemma | Proposition 1 | Proposition 2 | Affine image | Performance\n";
    std::cout << std::string(W, '=') << "\n";

    run_lemma();           // Lemma:   O(n) antipodal construction, V_good ≠ ∅
    run_proposition1();    // Prop 1:  x* ∈ P₊
    run_proposition2();    // Prop 2:  x* ∈ P₊ ∩ H₁ ∩ H₂
    run_hop_analysis();    // Phase II hop count and H₁/H₂ guidance
    run_affine_image();    // Affine image T(B)+c: diagonal and dense T
    run_timing_breakdown();// Phase-level cost breakdown
    run_performance();     // GeoQP vs qpOASES, varying n and Q type
    run_hotstart();        // Hotstart: fixed Q, varying g

    std::cout << "\n" << std::string(W, '=') << "\n";
    std::cout << "  Benchmark complete.\n";
    std::cout << std::string(W, '=') << "\n";
    return 0;
}
