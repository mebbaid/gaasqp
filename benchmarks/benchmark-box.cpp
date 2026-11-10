/**
 * @file benchmark-box.cpp
 * @brief benchmark for box QPs and affine-image-of-box QPs 
 *
 * Code partially written with the help of GitHub Copilot.
 */

#include <geoqp/geoqp.hpp>
#include <geoqp/box_geometry.hpp>
#include "solver_suite.hpp"

#include <qpOASES.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "benchmark_utils.hpp"


#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <limits>
#include <cstdlib>
#include <map>
#include <functional>
#include <numeric>
#include <cassert>

using Clock = std::chrono::high_resolution_clock;

// Ensure that x̄ = -Q⁻¹g lies strictly outside the box [lb, ub] 
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


// Euclidean squared distance ‖v − x̄‖₂²
static double eucl_dist_sq(const Eigen::VectorXd &v, const Eigen::VectorXd &xbar)
{
    return (v - xbar).squaredNorm();
}

// Euclidean projection of xbar onto axis-aligned box [lb, ub]
static Eigen::VectorXd project_onto_box(const Eigen::VectorXd &xbar,
                                         const Eigen::VectorXd &lb,
                                         const Eigen::VectorXd &ub)
{
    return xbar.cwiseMax(lb).cwiseMin(ub);
}

// Antipodal good vertex (Lemma part ii, box_antipodal.tex):
//   ŷᵢ = ℓᵢ  if ȳᵢ > ℓᵢ,  else uᵢ
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


// Corrected cut (visible-facet result, replaces the false half-space theorem
// η⊤(x*−c₀) ≥ 0 with ηᵢ = sᵢ/hᵢ for all i):
//   margin = η⊤(x*−c₀) − (2 − k),  ηᵢ = sᵢ/hᵢ on V = {i : x̄ violates bound i}, 0 off V.
// Axis-aligned box (T = I, c = 0).
static geoqp::geometry::BoxFrame box_frame(const Eigen::VectorXd &xbar,
                                           const Eigen::VectorXd &lb,
                                           const Eigen::VectorXd &ub)
{
    using geoqp::geometry::AffineBox;
    std::vector<double> l(lb.data(), lb.data() + lb.size());
    std::vector<double> u(ub.data(), ub.data() + ub.size());
    std::vector<double> xb(xbar.data(), xbar.data() + xbar.size());
    return geoqp::geometry::make_frame(AffineBox::axis_aligned(l, u), xb);
}

static double pplus_margin(const Eigen::VectorXd &xstar,
                            const Eigen::VectorXd &xbar,
                            const Eigen::VectorXd &lb,
                            const Eigen::VectorXd &ub)
{
    const auto f = box_frame(xbar, lb, ub);
    return geoqp::geometry::corrected_cut_margin(
        f, std::vector<double>(xstar.data(), xstar.data() + xstar.size()));
}

// Number of visible facets {ξᵢ = 1}, i ∈ V, containing x* (claimed ≥ 1).
static int visible_facet_count(const Eigen::VectorXd &xstar,
                               const Eigen::VectorXd &xbar,
                               const Eigen::VectorXd &lb,
                               const Eigen::VectorXd &ub,
                               double tol = 1e-7)
{
    const auto f = box_frame(xbar, lb, ub);
    return static_cast<int>(geoqp::geometry::visible_facets_containing(
        f, std::vector<double>(xstar.data(), xstar.data() + xstar.size()), tol).size());
}


//   margin = −(Qv̂+g)⊤(x*−v̂)  ← must be ≥ 0
static double h1_margin(const Eigen::VectorXd &xstar,
                         const Eigen::VectorXd &vhat,
                         const Eigen::MatrixXd &Q,
                         const Eigen::VectorXd &g)
{
    Eigen::VectorXd grad_vhat = Q * vhat + g;   // ∇f(v̂)
    return -grad_vhat.dot(xstar - vhat);         // −∇f(v̂)⊤(x*−v̂) ≥ 0
}


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

// Affine image T(B)+c with diagonal T (tᵢ ∈ [0.5, 3]): still axis aligned box
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
    Eigen::VectorXd lb_y;
    Eigen::VectorXd ub_y;
};

static QPProblem makeDenseAffineBoxQP(int n, std::mt19937 &rng, double scale_g = 5.0,
                                       DenseAffineInfo *info_out = nullptr)
{
    QPProblem p;
    p.n = n; p.m = 2 * n;
    p.name = "affine-dense/denseQ";
    p.H = randomPD(n, rng);

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


static QPProblem makeDenseAffineBoxQP_reformulated(int n, std::mt19937 &rng,
                                                    double scale_g = 5.0)
{
    DenseAffineInfo info;
    QPProblem orig = makeDenseAffineBoxQP(n, rng, scale_g, &info);

    const Eigen::MatrixXd &T = info.T;
    const Eigen::VectorXd &c = info.c;

    QPProblem p;
    p.n = n; p.m = 0;
    p.name = "affine-dense/reformulated";
    p.H = T.transpose() * orig.H * T;
    p.g = T.transpose() * (orig.H * c + orig.g);
    p.lbX = -Eigen::VectorXd::Ones(n);
    p.ubX =  Eigen::VectorXd::Ones(n);
    p.A = Eigen::MatrixXd(0, n);
    p.b = Eigen::VectorXd(0);
    ensure_xbar_infeasible(p.H, p.g, p.lbX, p.ubX);
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
    const int trials = 10;
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

            if (sol.vertices_explored == 0 && sol.x_phase1.empty()) continue;
            ++ct;

            sum_verts += sol.vertices_explored;

            if (sol.vertices_explored != 1) fast_path_ok = false;

            Eigen::VectorXd xbar = compute_xbar(qp.H, qp.g);
            Eigen::VectorXd vhat =
                Eigen::Map<const Eigen::VectorXd>(sol.x_phase1.data(), n);

            if (!is_good_box_vertex(vhat, xbar, qp.lbX, qp.ubX)) good_ok = false;

            for (int j = 0; j < n; ++j)
            {
                double dl = std::abs(xbar(j) - qp.lbX(j));
                double du = std::abs(xbar(j) - qp.ubX(j));
                double expected = (dl >= du) ? qp.lbX(j) : qp.ubX(j);
                if (std::abs(vhat(j) - expected) > 1e-6) coord_ok = false;
            }

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

//  Visible-facet lemma + corrected cut
//
//  x* lies on a facet {ξᵢ = 1} with i ∈ V, hence
//  P₊ = { x ∈ P : η⊤T⁻¹(x−c₀) ≥ 2 − k },  ηᵢ = sᵢ/hᵢ on V, 0 off V.
//  (Replaces the false half-box claim η⊤(x*−c₀) ≥ 0 with η = s/h everywhere.)

void run_proposition1()
{
    print_header("VISIBLE-FACET LEMMA: x* on a visible facet, x* ∈ P₊ = {Σ_V ξ ≥ 2−k}");

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
            if (visible_facet_count(xstar, xbar, qp.lbX, qp.ubX) < 1) ok = false;

            Eigen::VectorXd xe = project_onto_box(xbar, qp.lbX, qp.ubX);
            sum_v += (xbar - xe).norm();
            sum_L += (xbar - xstar).norm();  
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

    std::cout << "\nVisible-facet lemma overall: " << (all_pass ? "PASS" : "** FAIL **") << "\n";
}

// x* ∈ P₊ ∩ H₁
//  H₁ = { x : −∇f(v̂)⊤(x−v̂) ≥ 0 }  ← convexity + optimality of f at v̂

void run_proposition2()
{
    print_header("COROLLARY (cor:three_halfspaces): x* ∈ P₊ ∩ H₁");

    std::cout << std::setw(6)   << "n"
              << std::setw(12)  << "Q type"
              << std::setw(14)  << "P+ margin"
              << std::setw(14)  << "H1 margin"
              << std::setw(10)  << "min P+"
              << std::setw(10)  << "min H1"
              << std::setw(10)  << "Result"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    std::mt19937 rng(123);
    const int trials = 20;
    bool all_pass = true;

    auto check = [&](int n, bool diagonal)
    {
        double sum_pp = 0, sum_h1 = 0;
        double min_pp = 1e30, min_h1 = 1e30;
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

            sum_pp += pp; min_pp = std::min(min_pp, pp);
            sum_h1 += h1; min_h1 = std::min(min_h1, h1);

            if (pp < -1e-6) row_ok = false;
            if (h1 < -1e-6) row_ok = false;
        }

        if (!row_ok) all_pass = false;
        double inv = ct > 0 ? 1.0 / ct : 0.0;
        std::cout << std::setw(6)   << n
                  << std::setw(12)  << (diagonal ? "diagonal" : "dense")
                  << std::setw(14)  << std::scientific << std::setprecision(2)
                     << (sum_pp * inv)
                  << std::setw(14)  << (sum_h1 * inv)
                  << std::setw(10)  << std::setprecision(1) << min_pp
                  << std::setw(10)  << min_h1
                  << std::setw(10)  << (row_ok ? "PASS" : "FAIL")
                  << "\n";
    };

    for (int n : {4, 6, 8, 10, 12, 16, 20, 30, 50, 100})
    {
        check(n, true);
        check(n, false);
    }

    std::cout << "\nCorollary overall: " << (all_pass ? "PASS" : "** FAIL **") << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  Pivots and Hop Analysis
//
//  Since x* ∈ P₊ ∩ H₁ (Corollary cor:three_halfspaces), and Phase I already
//  places v̂ at the farthest good vertex, the H₁-guided violated-constraint
//  filter results in very few Phase II hops in practice.
//
//  Reports: avg hops, max hops, adjacency bound ≤2, us per hop, and how
//  phase II cost scales with n vs phase I and phase III.
// ═══════════════════════════════════════════════════════════════════════════

void run_hop_analysis()
{
    print_header("PHASE II HOP ANALYSIS: H₁-Guided Convergence");

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


void run_affine_image()
{
    print_header("AFFINE IMAGE T(B)+c: Theory Verification");

    // (a) Diagonal T
    print_subheader("(a) Diagonal T: fast-path fires, Thm+Corollary hold");
    std::cout << std::setw(6)   << "n"
              << std::setw(10)  << "Verts"
              << std::setw(10)  << "FastPath?"
              << std::setw(14)  << "P+ margin"
              << std::setw(14)  << "H1 margin"
              << std::setw(12)  << "ObjDiff"
              << std::setw(10)  << "Result"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    std::mt19937 rng_a(77);
    const int trials_a = 20;
    bool all_pass = true;

    for (int n : {4, 6, 8, 12, 16, 20, 30, 50, 100})
    {
        double sum_v = 0, sum_pp = 0, sum_h1 = 0, sum_od = 0;
        bool fp_ok = true, pp_ok = true, h1_ok = true;
        int ct = 0;

        for (int t = 0; t < trials_a; ++t)
        {
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

            sum_pp += pp; sum_h1 += h1;
            if (pp < -1e-6) pp_ok = false;
            if (h1 < -1e-6) h1_ok = false;

            auto ref = solveQPOASES(qp);
            if (ref.success)
                sum_od += std::abs(sol.objective_value - ref.obj);
        }

        bool row_ok = fp_ok && pp_ok && h1_ok;
        if (!row_ok) all_pass = false;
        double inv = ct > 0 ? 1.0 / ct : 0.0;
        std::cout << std::setw(6)  << n
                  << std::setw(10) << std::fixed << std::setprecision(1) << (sum_v * inv)
                  << std::setw(10) << (fp_ok ? "OK" : "FAIL")
                  << std::setw(14) << std::scientific << std::setprecision(2) << (sum_pp * inv)
                  << std::setw(14) << (sum_h1 * inv)
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
//  PERFORMANCE: GaasQP vs third-party solvers (cold solves, median over trials)
//
//  Every solver gets the same QP and the same tolerance; timings cover setup +
//  solve (see solver_suite.hpp).  Besides the human-readable table, one
//  machine-readable line is printed per (trial, solver):
//    PERF,<label>,<n>,<trial>,<solver>,<ok>,<time_us>,<iters>,<obj_diff>,<viol>,<pivots>,<hops>
//  obj_diff is |f − f_ref| against qpOASES (QProblem); pivots/hops are GaasQP's.
// ═══════════════════════════════════════════════════════════════════════════

static int bench_trials()
{
    const char *env = std::getenv("GEOQP_BENCH_TRIALS");
    const int t = env ? std::atoi(env) : 30;
    return t > 0 ? t : 30;
}

static double median_of(std::vector<double> v)
{
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    const size_t k = v.size() / 2;
    return (v.size() % 2) ? v[k] : 0.5 * (v[k - 1] + v[k]);
}

static bench::QP to_bench_qp(const QPProblem &qp)
{
    bench::QP b;
    b.H = qp.H;
    b.g = qp.g;
    b.A = qp.A;
    b.ubA = qp.b;
    b.lbA = Eigen::VectorXd::Constant(qp.m, -bench::kInf);
    b.lb = qp.lbX;
    b.ub = qp.ubX;
    return b;
}

void run_performance()
{
    print_header("PERFORMANCE: GaasQP vs third-party solvers on box QPs");

    const double tol = 1e-10;
    const int trials = bench_trials();
    const std::vector<bench::SolverId> others = bench::all_solvers();

    std::cout << "# trials=" << trials << "  tol=" << std::scientific << std::setprecision(0) << tol
              << std::defaultfloat
              << "  times: median wall-clock (setup+solve) per cold solve, microseconds\n";
    std::cout << "# PERF,<label>,<n>,<trial>,<solver>,<ok>,<time_us>,<iters>,<obj_diff>,<viol>,<pivots>,<hops>\n";

    auto header = [&]() {
        std::cout << std::setw(6) << "n" << std::setw(18) << "type" << std::setw(11) << "GaasQP";
        for (auto id : others) std::cout << std::setw(11) << bench::name(id);
        std::cout << std::setw(8) << "Pivot" << std::setw(7) << "Hops" << std::setw(11) << "MaxObjDiff"
                  << "  failures\n";
        std::cout << std::string(W, '-') << "\n";
    };

    std::mt19937 rng(42);

    auto bench_row = [&](int n, const std::string &label,
                         std::function<QPProblem(int, std::mt19937 &)> gen)
    {
        std::map<std::string, std::vector<double>> times;
        std::map<std::string, int> fails;
        std::vector<double> pivots, hops;
        double max_od = 0.0;

        for (int t = 0; t < trials; ++t)
        {
            const QPProblem qp = gen(n, rng);
            // Bounds are passed as lb/ub (as to every other solver), not as
            // explicit ±eⱼ rows; general rows, if any, stay in A.
            const geoqp::Problem gp = to_geoqp(qp);
            const bench::QP bq = to_bench_qp(qp);

            // Reference objective: qpOASES (general QProblem).
            const bench::Run ref = bench::run(bench::SolverId::qpOASES, bq, tol);
            const double fref = ref.ok ? ref.obj : std::numeric_limits<double>::quiet_NaN();

            auto emit = [&](const std::string &solver, bool ok, double us, int iters,
                            double obj, double viol, int piv, int hop) {
                const double od = std::abs(obj - fref);
                if (ok)
                {
                    times[solver].push_back(us);
                    if (std::isfinite(od) && solver != "qpOASES") max_od = std::max(max_od, od);
                }
                else
                    ++fails[solver];
                std::cout << "PERF," << label << "," << n << "," << t << "," << solver << ","
                          << (ok ? 1 : 0) << "," << std::setprecision(6) << us << "," << iters << ","
                          << od << "," << viol << "," << piv << "," << hop << "\n";
            };

            // GaasQP
            {
                geoqp::Solver solver(default_cfg());
                const auto t0 = Clock::now();
                const auto sol = solver.solve(gp);
                const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
                const bool ok = (sol.status == geoqp::Status::SUCCESS);
                double obj = std::numeric_limits<double>::quiet_NaN(), viol = obj;
                if (ok)
                {
                    const Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(sol.x.data(), n);
                    obj = bench::objective(bq, x);
                    viol = bench::max_violation(bq, x);
                }
                emit("GaasQP", ok, us, sol.phase2_iterations, obj, viol,
                     sol.phase2_iterations, sol.phase2_facet_hops);
                if (ok)
                {
                    pivots.push_back(sol.phase2_iterations);
                    hops.push_back(sol.phase2_facet_hops);
                }
            }
            // Third-party solvers (qpOASES result reused, not re-run)
            for (auto id : others)
            {
                if (!bench::applicable(id, bq)) continue;
                const bench::Run r = (id == bench::SolverId::qpOASES) ? ref : bench::run(id, bq, tol);
                emit(bench::name(id), r.ok, r.time_us, r.iters, r.obj, r.viol, -1, -1);
            }
        }

        std::cout << std::setw(6) << n << std::setw(18) << label << std::fixed << std::setprecision(1)
                  << std::setw(11) << median_of(times["GaasQP"]);
        for (auto id : others)
        {
            const auto &v = times[bench::name(id)];
            if (v.empty()) std::cout << std::setw(11) << "-";
            else std::cout << std::setw(11) << median_of(v);
        }
        std::cout << std::setw(8) << median_of(pivots) << std::setw(7) << median_of(hops)
                  << std::setw(11) << std::scientific << std::setprecision(1) << max_od << "  ";
        bool any = false;
        for (const auto &kv : fails)
            if (kv.second > 0)
            {
                std::cout << kv.first << ":" << kv.second << "/" << trials << " ";
                any = true;
            }
        std::cout << (any ? "" : "none") << "\n" << std::defaultfloat << std::flush;
    };

    print_subheader("Axis-aligned, diagonal Q");
    header();
    for (int n : {5, 10, 20, 30, 50, 75, 100, 150, 200})
        bench_row(n, "box/diagQ", [](int n, std::mt19937 &r){ return makeDiagBoxQP(n, r); });

    print_subheader("Axis-aligned, dense Q");
    header();
    for (int n : {5, 10, 20, 30, 50, 75, 100, 150, 200})
        bench_row(n, "box/denseQ", [](int n, std::mt19937 &r){ return makeBoxQP(n, r); });

    print_subheader("Dense-T affine image, reformulated (box fast-path)");
    header();
    for (int n : {5, 10, 20, 30, 50, 75, 100, 150, 200})
        bench_row(n, "affine-dense/refo", [](int n, std::mt19937 &r){ return makeDenseAffineBoxQP_reformulated(n, r); });

    print_subheader("Dense-T affine image, explicit A = [T^-1; -T^-1] (general path)");
    header();
    for (int n : {3, 4, 5, 6, 7, 8})
        bench_row(n, "affine-dense/Q", [](int n, std::mt19937 &r){ return makeDenseAffineBoxQP(n, r); });

    print_subheader("Ill-conditioned Q (cond=10^4) — box");
    header();
    for (int n : {5, 10, 20, 30, 50})
        bench_row(n, "box/cond=1e4",
                  [](int n, std::mt19937 &r){ return makeIllCondBoxQP(n, r, 1e4); });
}

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


int main()
{
    std::cout << std::string(W, '=') << "\n";
    std::cout << "  GeoQP BOX BENCHMARK — following box_antipodal.tex\n";
    std::cout << "  Lemma | Proposition 1 | Proposition 2 | Affine image | Performance\n";
    std::cout << std::string(W, '=') << "\n";

    run_lemma();           
    run_proposition1();    
    run_proposition2();    
    run_hop_analysis();    
    run_affine_image();    
    run_timing_breakdown();
    run_performance();     
    run_hotstart();        

    std::cout << "\n" << std::string(W, '=') << "\n";
    std::cout << "  Benchmark complete.\n";
    std::cout << std::string(W, '=') << "\n";
    return 0;
}
