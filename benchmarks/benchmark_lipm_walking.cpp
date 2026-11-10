/**
 * @brief Benchmark: DCM (Divergent Component of Motion) MPC 
 *
 *  (see https://github.com/gbionics/walking-controllers/blob/master/src/SimplifiedModelControllers/src/DCMModelPredictiveController.cpp):
 * 
 * Solvers compared:
 *   1. GeoQP  (Dense active-set, geometric initialisation)
 *   2. qpOASES (Dense active-set, MPC options)
 *   3. OSQP   (Sparse ADMM, dense problem passed via QpSolversEigen)
 *
 * Code written with the help of GitHub Copilot.
 */

#include <geoqp/geoqp.hpp>
#include "solver_suite.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <geoqp/internal.hpp>
#include <geoqp/timing.hpp>
#include <cstdlib>
#include <limits>

// Tolerance applied to every solver in this benchmark (paper: 1e-7).
static constexpr double kLipmTol = 1e-7;

using Clock = std::chrono::high_resolution_clock;


struct Parameters
{
    double com_height   = 0.53;   // [m]  CoM height (LIPM assumption)
    double gravity      = 9.81;   // [m/s²]
    double dt           = 0.01;   // [s]  sampling period T (100 Hz — keeps alpha close to 1)
    int    N            = 15;     // MPC horizon (number of steps)
    double foot_half_x  = 0.10;   // [m]  half-length of each foot support rectangle (x)
    double foot_half_y  = 0.05;   // [m]  half-width  of each foot support rectangle (y)
    double stride_x     = 0.18;   // [m]  forward step length per stride
    double stride_y     = 0.10;   // [m]  lateral   step width  per stride
    double dsp_duration = 0.05;   // [s]  double-support phase  (5 steps at 100 Hz)
    double ssp_duration = 0.35;   // [s]  single-support phase  (35 steps at 100 Hz)

    double q_scale = 1.0;
    double r_scale = 1.0;

    double omega() const { return std::sqrt(gravity / com_height); }
    double alpha() const { return std::exp(omega() * dt); }
};

struct DenseQP
{
    int n = 0;              // 2*N
    int m = 0;              // 0 (no general A matrix — pure box)
    std::vector<double> H;  // n×n, row-major, dense
    std::vector<double> g;  // n
    std::vector<double> lb; // n  (per-component lower bound on U)
    std::vector<double> ub; // n  (per-component upper bound on U)
    std::vector<double> C;
    std::vector<double> lbA;
    std::vector<double> ubA;
    double const_term = 0.0;
};


// Per-step support box: [lx, ux] x [ly, uy]
struct SupportBox { double lx, ux, ly, uy; };

// ─────────────────────────────────────────────────────────────────────────────
//  Per-step support polygon
//
//  During single-support (SS) the ZMP constraint is an axis-aligned rectangle
//  — represented as lb/ub bounds, firing GeoQP's box fast-path.
//
//  During double-support (DS) the ZMP constraint is the convex hull of both
//  feet, computed via Andrews' monotone chain.  The result is a set of
//  halfplane constraints  A_k * u_k <= b_k  (typically 6 rows for two
//  offset rectangles), passed as general A/b rows in the QP.
//
//  is_box == true  →  use lb/ub (SS)
//  is_box == false →  use halfplanes A_rows/b_rows (DS)
// ─────────────────────────────────────────────────────────────────────────────
struct SupportPolygon
{
    bool is_box = true;
    // SS: axis-aligned bounds
    double lx = 0, ux = 0, ly = 0, uy = 0;
    // DS: halfplane rows  A_rows[i] = {a0, a1},  b_rows[i] = b_i
    //     constraint: a0*u_x + a1*u_y <= b_i
    std::vector<std::array<double, 2>> A_rows;
    std::vector<double> b_rows;
};

// Monotone-chain 2-D convex hull of a set of points.
static std::vector<std::array<double,2>> convexHull2D(std::vector<std::array<double,2>> pts)
{
    int n = static_cast<int>(pts.size());
    if (n < 3) return pts;
    std::sort(pts.begin(), pts.end(), [](const auto &a, const auto &b){
        return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
    });
    auto cross = [](const auto &o, const auto &a, const auto &b){
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]);
    };
    std::vector<std::array<double,2>> h(2*n);
    int k = 0;
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross(h[k-2], h[k-1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    for (int i = n-2, t = k+1; i >= 0; --i) {
        while (k >= t && cross(h[k-2], h[k-1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    h.resize(k-1);
    return h;
}

// Build halfplane constraints from a CCW convex hull polygon.
// Each edge (p0→p1) gives:  (p1.y-p0.y)*u_x + (p0.x-p1.x)*u_y <= p0.x*p1.y - p1.x*p0.y
static SupportPolygon polygonFromHull(const std::vector<std::array<double,2>> &hull)
{
    SupportPolygon sp;
    sp.is_box = false;
    int n = static_cast<int>(hull.size());
    for (int i = 0; i < n; ++i) {
        const auto &p0 = hull[i];
        const auto &p1 = hull[(i+1) % n];
        double a0 = p1[1]-p0[1];
        double a1 = p0[0]-p1[0];
        double b  = p0[0]*p1[1] - p1[0]*p0[1];
        // Normalise to unit L2 norm. 
        const double norm = std::sqrt(a0*a0 + a1*a1);
        if (norm > 1e-12) { a0 /= norm; a1 /= norm; b /= norm; }
        sp.A_rows.push_back({a0, a1});
        sp.b_rows.push_back(b);
    }
    return sp;
}

// Build the convex hull of two foot rectangles for a DS step.
static SupportPolygon makeDSPolygon(double lx, double rx,  
                                    double ly, double ry, 
                                    double hx, double hy)
{
    std::vector<std::array<double,2>> pts;
    pts.reserve(8);
    for (double fx : {lx, rx})
        for (double fy : {ly, ry})
            for (auto [dx, dy] : std::array<std::array<double,2>,4>{{{-hx,-hy},{-hx,hy},{hx,-hy},{hx,hy}}})
                pts.push_back({fx+dx, fy+dy});
    return polygonFromHull(convexHull2D(pts));
}

DenseQP buildQP(const Parameters &p,
                const std::vector<double>        &xi0,          // 2D DCM: [xi_x, xi_y]
                const std::vector<double>        &u_prev,       // previous ZMP: [u_x, u_y]
                const std::vector<SupportBox>    &support,      // N support boxes
                const std::vector<double>        &xi_ref)       // 2*N reference DCM trajectory
{
    const int  N    = p.N;
    const int  n    = 2 * N;           // total decision variables
    const double al = p.alpha();       // e^{omega*T}
    const double be = 1.0 - al;        // (1 - alpha) = B_d scalar
    const double Q  = p.q_scale;       // DCM tracking weight (per component)
    const double R  = p.r_scale;       // input-increment weight (per component)

    // S_u[k,j] = alpha^{k-j} * beta  for j <= k
    std::vector<double> Su_s(N * N, 0.0);
    for (int k = 0; k < N; ++k)
    {
        double coeff = be;
        for (int j = k; j >= 0; --j)
        {
            Su_s[k * N + j] = coeff;
            coeff *= al;
        }
    }
    // (S_u^T * Q * S_u)[i,j] = Q * sum_k Su[k,i]*Su[k,j]
    std::vector<double> H_s(N * N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            double s = 0.0;
            for (int k = std::max(i,j); k < N; ++k)
                s += Su_s[k * N + i] * Su_s[k * N + j];
            H_s[i * N + j] = Q * s;
        }

    // Term 2: Theta^T * R * Theta.
    // Theta (N x N): Theta[k,k]=1, Theta[k,k-1]=-1 for k>=1
    // (Theta^T * Theta)[i,j]:
    //   diagonal [i,i]: 1 + (i<N-1 ? 1 : 0)  => 2 for i<N-1, 1 for i=N-1
    //   off-diag [i,i+1] = [i+1,i] = -1  for i < N-1
    for (int i = 0; i < N; ++i)
    {
        H_s[i * N + i] += R * (i < N - 1 ? 2.0 : 1.0);
        if (i < N - 1)
        {
            H_s[i * N + (i + 1)] += -R;
            H_s[(i + 1) * N + i] += -R;
        }
    }
    // Regulariser
    for (int i = 0; i < N; ++i)
        H_s[i * N + i] += 1e-8;

    DenseQP qp;
    qp.n = n;
    qp.m = 0;
    qp.H.assign(n * n, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            double h = H_s[i * N + j];
            // x block
            qp.H[(2*i)   * n + (2*j)  ] = h;
            // y block
            qp.H[(2*i+1) * n + (2*j+1)] = h;
        }


    std::vector<double> alpha_pow(N + 1); 
    alpha_pow[0] = al;
    for (int k = 1; k < N; ++k)
        alpha_pow[k] = alpha_pow[k - 1] * al;

    // Gradient g 
    qp.g.assign(n, 0.0);
    for (int axis = 0; axis < 2; ++axis)
    {
        const double xi0_a   = xi0[axis];
        const double uprev_a = u_prev[axis];

        for (int i = 0; i < N; ++i)
        {
            double s = 0.0;
            for (int k = i; k < N; ++k)
            {
                double e_k = alpha_pow[k] * xi0_a - xi_ref[k * 2 + axis];
                s += Su_s[k * N + i] * e_k;
            }
            qp.g[2 * i + axis] = Q * s;
        }
        // Boundary term from input-increment penalty (Theta * U evaluated at k=0)
        qp.g[axis] += -R * uprev_a;
    }

    //  Constant term 
    for (int axis = 0; axis < 2; ++axis)
        for (int k = 0; k < N; ++k)
        {
            double e = alpha_pow[k] * xi0[axis] - xi_ref[k * 2 + axis];
            qp.const_term += 0.5 * Q * e * e;
        }
    qp.const_term += 0.5 * R * (u_prev[0]*u_prev[0] + u_prev[1]*u_prev[1]);

    // Box constraints: lb <= U <= ub 
    qp.lb.resize(n);
    qp.ub.resize(n);
    for (int k = 0; k < N; ++k)
    {
        qp.lb[2 * k    ] = support[k].lx;
        qp.ub[2 * k    ] = support[k].ux;
        qp.lb[2 * k + 1] = support[k].ly;
        qp.ub[2 * k + 1] = support[k].uy;
    }

    return qp;
}

//  buildQP overload for mixed SS/DS support polygons (GEOQP_ENABLE_DS=1)
DenseQP buildQP(const Parameters                    &p,
                const std::vector<double>            &xi0,
                const std::vector<double>            &u_prev,
                const std::vector<SupportPolygon>    &support,
                const std::vector<double>            &xi_ref)
{
    const int N = p.N;
    const int n = 2 * N;

    int m_total = 0;
    for (const auto &sp : support)
        if (!sp.is_box)
            m_total += static_cast<int>(sp.b_rows.size());

    // Build temporarily with infinite boxes to get H, g, const_term
    std::vector<SupportBox> dummy(N, {-1e20, 1e20, -1e20, 1e20});
    DenseQP qp = buildQP(p, xi0, u_prev, dummy, xi_ref);

    qp.m = m_total;

    // Overwrite per-step bounds; 
    for (int k = 0; k < N; ++k)
    {
        const auto &sp = support[k];
        if (sp.is_box)
        {
            qp.lb[2*k    ] = sp.lx;  qp.ub[2*k    ] = sp.ux;
            qp.lb[2*k + 1] = sp.ly;  qp.ub[2*k + 1] = sp.uy;
        }
        else
        {
            qp.lb[2*k    ] = qp.lb[2*k + 1] = -1e31;
            qp.ub[2*k    ] = qp.ub[2*k + 1] =  1e31;
        }
    }

    if (m_total > 0)
    {
        qp.C.assign(m_total * n, 0.0);
        qp.lbA.assign(m_total, -1e20);
        qp.ubA.resize(m_total);

        int row = 0;
        for (int k = 0; k < N; ++k)
        {
            const auto &sp = support[k];
            if (sp.is_box) continue;
            for (int r = 0; r < (int)sp.b_rows.size(); ++r)
            {
                qp.C[row * n + 2*k    ] = sp.A_rows[r][0];
                qp.C[row * n + 2*k + 1] = sp.A_rows[r][1];
                qp.ubA[row] = sp.b_rows[r];
                ++row;
            }
        }
    }

    return qp;
}


//  DCM walking schedule — pure-box version (default, fast-path)
static std::vector<SupportBox> makeSupportBoxes(const Parameters &p,
                                                 double cx0, double cy0,
                                                 int N)
{
    const double hx = p.foot_half_x;
    const double hy = p.foot_half_y;

    const int nb_dsp = std::max(1, (int)std::round(p.dsp_duration / p.dt));
    const int nb_ssp = std::max(1, (int)std::round(p.ssp_duration / p.dt));
    const int step_len = nb_dsp + nb_ssp;

    std::vector<SupportBox> boxes;
    boxes.reserve(N);

    double foot_x = cx0, foot_y = cy0;
    int foot_sign = 1, phase_idx = 0;

    for (int k = 0; k < N; ++k)
    {
        boxes.push_back({foot_x - hx, foot_x + hx, foot_y - hy, foot_y + hy});
        if (++phase_idx >= step_len)
        {
            phase_idx = 0;
            foot_x += p.stride_x;
            foot_y += foot_sign * p.stride_y;
            foot_sign = -foot_sign;
        }
    }
    return boxes;
}

//  DCM walking schedule — SS/DS version (GEOQP_ENABLE_DS=1)
static std::vector<SupportPolygon> makeSupportPolygons(const Parameters &p,
                                                        double cx0, double cy0,
                                                        int N)
{
    const double hx = p.foot_half_x;
    const double hy = p.foot_half_y;

    const int nb_dsp = std::max(1, (int)std::round(p.dsp_duration / p.dt));
    const int nb_ssp = std::max(1, (int)std::round(p.ssp_duration / p.dt));
    const int step_len = nb_dsp + nb_ssp;

    std::vector<SupportPolygon> polys;
    polys.reserve(N);

    double foot_x = cx0, foot_y = cy0;
    double prev_foot_x = cx0, prev_foot_y = cy0;
    int foot_sign = 1, phase_idx = 0;

    for (int k = 0; k < N; ++k)
    {
        if (phase_idx < nb_dsp)
        {
            polys.push_back(makeDSPolygon(prev_foot_x, foot_x,
                                          prev_foot_y, foot_y, hx, hy));
        }
        else
        {
            SupportPolygon sp;
            sp.is_box = true;
            sp.lx = foot_x - hx; sp.ux = foot_x + hx;
            sp.ly = foot_y - hy; sp.uy = foot_y + hy;
            polys.push_back(sp);
        }
        if (++phase_idx >= step_len)
        {
            phase_idx = 0;
            prev_foot_x = foot_x; prev_foot_y = foot_y;
            foot_x += p.stride_x;
            foot_y += foot_sign * p.stride_y;
            foot_sign = -foot_sign;
        }
    }
    return polys;
}

// Build a DCM reference trajectory
static std::vector<double> makeDCMRef(int N, double goal_x, double goal_y)
{
    std::vector<double> ref(2 * N);
    for (int k = 0; k < N; ++k)
    {
        ref[2 * k    ] = goal_x;
        ref[2 * k + 1] = goal_y;
    }
    return ref;
}


struct Result
{
    std::string name;
    bool success = false;
    double time_ms = 0.0;
    double objective = 0.0;
    int iterations = -1;
    int facet_hops = -1;
    int avg_pivot = -1;
    std::vector<double> solution;
};

static Result runGeoQP(const DenseQP &qp, geoqp::FaceHoppingStrategy strat, const std::string &name)
{
    geoqp::Problem prob;
    prob.n = qp.n;
    prob.Q = qp.H;
    prob.g = qp.g;
    prob.lb = qp.lb;
    prob.ub = qp.ub;

    // Pass general A*x <= b rows (populated for DS steps; empty for pure-box mode)
    prob.m = qp.m;
    prob.A = qp.C;
    prob.b = qp.ubA;

    geoqp::Config cfg;
    cfg.face_hopping_strategy = strat;
    // Plain Phase II by default: on this benchmark it needs far fewer pivots
    // (2–7 instead of 2–50) and is faster than the Schur-caching variant.
    // GEOQP_USE_CACHING=1 selects the caching variant for the general path.
    const char *env = std::getenv("GEOQP_USE_CACHING");
    cfg.use_caching = (env && std::string(env) == "1");
    cfg.warm_start = false;
    cfg.tolerance = kLipmTol;
    cfg.constraint_tolerance = kLipmTol; // same tolerance as every other solver

    if (!geoqp::validate_problem(prob))
    {
        std::cerr << "[ERROR] runGeoQP: invalid problem for solver (n=" << prob.n << " m=" << prob.m << ")\n";
        Result fail; fail.name = name; fail.success = false; return fail;
    }

    geoqp::Solver solver(cfg);
    Result res;
    res.name = name;

    auto t0 = Clock::now();
    auto sol = solver.solve(prob);
    res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    res.success = (sol.status == geoqp::Status::SUCCESS);
    res.objective = sol.objective_value + qp.const_term;

    res.iterations = sol.iterations;
    res.facet_hops = sol.phase2_facet_hops;
    res.avg_pivot = sol.phase2_iterations;
    res.solution = sol.x;
    return res;
}

// Condensed LIPM QP → common benchmark QP (bounds with |v| ≥ 1e20 mean "none").
static bench::QP to_bench_qp(const DenseQP &qp)
{
    const int n = qp.n, m = qp.m;
    bench::QP b;
    b.H = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(qp.H.data(), n, n);
    b.g = Eigen::Map<const Eigen::VectorXd>(qp.g.data(), n);
    b.lb = Eigen::VectorXd::Constant(n, -bench::kInf);
    b.ub = Eigen::VectorXd::Constant(n, bench::kInf);
    if ((int)qp.lb.size() == n && (int)qp.ub.size() == n)
        for (int i = 0; i < n; ++i)
        {
            b.lb(i) = std::max(qp.lb[i], -bench::kInf);
            b.ub(i) = std::min(qp.ub[i], bench::kInf);
        }
    b.A.resize(m, n);
    b.lbA.resize(m);
    b.ubA.resize(m);
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j) b.A(i, j) = qp.C[i * n + j];
        b.lbA(i) = std::max(qp.lbA[i], -bench::kInf);
        b.ubA(i) = std::min(qp.ubA[i], bench::kInf);
    }
    return b;
}


int main()
{
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║  DCM MPC Benchmark                                           ║\n"
              << "║  GeoQP vs qpOASES vs OSQP — pure box constraints            ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n\n";

    Parameters p;

    const char *qenv = std::getenv("MPC_Q_SCALE");
    const char *renv = std::getenv("MPC_R_SCALE");
    if (qenv) p.q_scale = std::atof(qenv);
    if (renv) p.r_scale = std::atof(renv);

    const double xi0_x = 0.0, xi0_y = 0.0;
    const double u_prev_x = 0.0, u_prev_y = 0.0;
    const double cx0 = 0.0, cy0 = 0.0;    

    std::cout << "Parameters:\n"
              << "  com_height = " << p.com_height << " m\n"
              << "  omega      = " << p.omega()    << " rad/s\n"
              << "  alpha      = " << p.alpha()    << "\n"
              << "  dt         = " << p.dt         << " s\n"
              << "  foot_half  = " << p.foot_half_x << " x " << p.foot_half_y << " m\n"
              << "  stride     = " << p.stride_x    << " x " << p.stride_y    << " m\n"
              << "  Q_scale    = " << p.q_scale     << "  R_scale = " << p.r_scale << "\n\n";

    // We will test horizons N = 5, 10, ..., 60
    std::vector<int> horizons;
    for (int h = 5; h <= 60; h += 5)
        horizons.push_back(h);

    geoqp::internal::schur_stats_reset();

    const char *repeats_env = std::getenv("GEOQP_REPEATS");
    int repeats = repeats_env ? std::atoi(repeats_env) : 30;

    const char *caching_env = std::getenv("GEOQP_USE_CACHING");
    const bool caching_on = (caching_env && std::string(caching_env) == "1");

    const char *ds_env = std::getenv("GEOQP_ENABLE_DS");
    const bool enable_ds = (ds_env && std::string(ds_env) == "1");

    std::cout << "\nResults (horizons)\n";
    std::cout << "# repeats=" << repeats
              << " GEOQP_USE_CACHING=" << (caching_on ? "1" : "0")
              << " GEOQP_ENABLE_DS=" << (enable_ds ? "1" : "0")
              << " constraints=" << (enable_ds ? "SS-box+DS-hull" : "pure-box")
              << "\n";
    std::cout << "CASE format: n=<horizon> QP_n=<n> QP_m=<m> mode=<mode> run_idx=<i>"
                 " Geo_ms=<ms> qpO_ms=<ms> OSQP_ms=<ms>"
                 " Geo_it=<it> qpO_it=<it> OSQP_it=<it>"
                 " Geo_hops=<hops> AvgPivot=<pivots>"
                 " AvgObjGeo=<obj> Diff_vs_qpO=<diff>"
                 " OK_geo=<ok> OK_qpO=<ok> OK_osqp=<ok>\n";

    for (int h : horizons)
    {
        Parameters pc = p;
        pc.N = h;

        const std::vector<double> xi0   = {xi0_x, xi0_y};
        const std::vector<double> u_prev = {u_prev_x, u_prev_y};

        // Goal: walk to first foot landing ~N*stride_x ahead
        double goal_x = cx0 + pc.N * pc.stride_x * 0.5;
        auto xi_ref   = makeDCMRef(h, goal_x, cy0);

        DenseQP qp;
        if (enable_ds)
        {
            auto support = makeSupportPolygons(pc, cx0, cy0, h);
            qp = buildQP(pc, xi0, u_prev, support, xi_ref);
        }
        else
        {
            auto support = makeSupportBoxes(pc, cx0, cy0, h);
            qp = buildQP(pc, xi0, u_prev, support, xi_ref);
        }

        const char *mode_str = enable_ds ? "DS" : "box";

        for (int run_idx = 0; run_idx < repeats; ++run_idx)
        {
            auto geo_res  = runGeoQP(qp, geoqp::FaceHoppingStrategy::STANDARD, "GeoQP");
            // Third-party solvers through the common runners (cold solve each).
            const bench::QP bq = to_bench_qp(qp);
            std::vector<bench::Run> others;
            for (bench::SolverId id : bench::all_solvers())
                if (bench::applicable(id, bq))
                    others.push_back(bench::run(id, bq, kLipmTol));
            auto find = [&](bench::SolverId id) -> const bench::Run * {
                for (const auto &r : others)
                    if (r.id == id) return &r;
                return nullptr;
            };
            const bench::Run *qpo = find(bench::SolverId::qpOASES);
            const bench::Run *osqp = find(bench::SolverId::OSQP);
            const double f_ref = (qpo && qpo->ok) ? qpo->obj + qp.const_term
                                                  : std::numeric_limits<double>::quiet_NaN();

            double diff_geo = 0.0;
            if (qpo && qpo->ok && geo_res.success)
                diff_geo = std::abs(geo_res.objective - f_ref);

            std::cout << "CASE "
                      << "n=" << h << " "
                      << "QP_n=" << qp.n << " "
                      << "QP_m=" << qp.m << " "
                      << "mode=" << mode_str << " "
                      << "run_idx=" << run_idx << " "
                      << "Geo_ms=" << geo_res.time_ms << " "
                      << "qpO_ms=" << (qpo ? qpo->time_us * 1e-3 : -1.0) << " "
                      << "OSQP_ms=" << (osqp ? osqp->time_us * 1e-3 : -1.0) << " "
                      << "Geo_it=" << geo_res.iterations << " "
                      << "qpO_it=" << (qpo ? qpo->iters : -1) << " "
                      << "OSQP_it=" << (osqp ? osqp->iters : -1) << " "
                      << "Geo_hops=" << geo_res.facet_hops << " "
                      << "AvgPivot=" << geo_res.avg_pivot << " "
                      << "AvgObjGeo=" << geo_res.objective << " "
                      << "Diff_vs_qpO=" << diff_geo << " "
                      << "OK_geo=" << (geo_res.success ? 1 : 0) << " "
                      << "OK_qpO=" << ((qpo && qpo->ok) ? 1 : 0) << " "
                      << "OK_osqp=" << ((osqp && osqp->ok) ? 1 : 0);
            // One group per third-party solver: S_<name>_{ms,it,ok,df}
            for (const auto &r : others)
            {
                const std::string nm = bench::name(r.id);
                const double df = r.ok ? std::abs(r.obj + qp.const_term - f_ref)
                                       : std::numeric_limits<double>::quiet_NaN();
                std::cout << " S_" << nm << "_ms=" << r.time_us * 1e-3
                          << " S_" << nm << "_it=" << r.iters
                          << " S_" << nm << "_ok=" << (r.ok ? 1 : 0)
                          << " S_" << nm << "_df=" << df;
            }
            std::cout << "\n";
        }
    }

    std::cout << "\nFinished horizon cases.\n";
    return 0;
}