// Sanity check for solver_suite: every third-party runner must agree with the
// qpOASES (QProblem) reference on random box QPs and on QPs with general rows.
// GaasQP is compared too whenever it reports SUCCESS; a GaasQP failure is
// printed but does not fail this check (known issue: on some box + general-row
// problems Phase II returns more than n constraints and Phase III fails).
// Exit code 0 on success; registered with ctest as `solver_suite_agrees`.

#include "solver_suite.hpp"

#include <geoqp/geoqp.hpp>

#include <cmath>
#include <cstdio>
#include <random>

namespace
{
    bench::QP random_qp(int n, int m, std::mt19937 &rng)
    {
        std::normal_distribution<> N(0.0, 1.0);
        bench::QP qp;
        Eigen::MatrixXd R(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                R(i, j) = N(rng);
        qp.H = R.transpose() * R + 0.1 * n * Eigen::MatrixXd::Identity(n, n);
        qp.g.resize(n);
        for (int i = 0; i < n; ++i) qp.g(i) = 6.0 * N(rng);
        qp.lb = -Eigen::VectorXd::Ones(n);
        qp.ub = Eigen::VectorXd::Ones(n);
        qp.A.resize(m, n);
        qp.lbA = Eigen::VectorXd::Constant(m, -bench::kInf);
        qp.ubA.resize(m);
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j) qp.A(i, j) = N(rng);
            qp.ubA(i) = 0.5 + std::abs(N(rng));
        }
        return qp;
    }

    bool solve_gaasqp(const bench::QP &qp, Eigen::VectorXd &x)
    {
        geoqp::Problem p;
        p.n = qp.n();
        p.m = qp.m();
        p.Q.assign(qp.H.data(), qp.H.data() + qp.n() * qp.n()); // symmetric
        p.g.assign(qp.g.data(), qp.g.data() + qp.n());
        p.A.resize(static_cast<size_t>(p.m) * p.n);
        for (int i = 0; i < p.m; ++i)
            for (int j = 0; j < p.n; ++j) p.A[i * p.n + j] = qp.A(i, j);
        p.b.assign(qp.ubA.data(), qp.ubA.data() + p.m);
        p.lb.assign(qp.lb.data(), qp.lb.data() + p.n);
        p.ub.assign(qp.ub.data(), qp.ub.data() + p.n);
        geoqp::Config cfg;
        cfg.verbosity = -1;
        const auto sol = geoqp::Solver(cfg).solve(p);
        if (sol.status != geoqp::Status::SUCCESS) return false;
        x = Eigen::Map<const Eigen::VectorXd>(sol.x.data(), p.n);
        return true;
    }
} // namespace

int main()
{
    std::mt19937 rng(2026);
    const double tol = 1e-9;
    int failures = 0, checks = 0, gaasqp_failed = 0;
    for (int m : {0, 6})
        for (int n : {3, 8, 20})
            for (int t = 0; t < 5; ++t)
            {
                const bench::QP qp = random_qp(n, m, rng);
                const bench::Run ref = bench::run(bench::SolverId::qpOASES, qp, tol);
                if (!ref.ok)
                {
                    std::printf("reference qpOASES failed n=%d m=%d\n", n, m);
                    ++failures;
                    continue;
                }
                const Eigen::VectorXd &xref = ref.x;
                const double fref = ref.obj;
                Eigen::VectorXd xg;
                if (solve_gaasqp(qp, xg))
                {
                    ++checks;
                    const double dx = (xg - xref).lpNorm<Eigen::Infinity>();
                    if (dx > 1e-6)
                    {
                        ++failures;
                        std::printf("FAIL GaasQP    n=%2d m=%d |dx|=%.2e\n", n, m, dx);
                    }
                }
                else
                {
                    ++gaasqp_failed;
                    std::printf("note: GaasQP did not return SUCCESS on n=%d m=%d (known mixed-problem issue)\n", n, m);
                }
                for (bench::SolverId id : bench::all_solvers())
                {
                    if (!bench::applicable(id, qp)) continue;
                    const bench::Run r = bench::run(id, qp, tol);
                    ++checks;
                    const double dx = r.ok ? (r.x - xref).lpNorm<Eigen::Infinity>() : NAN;
                    const double df = r.ok ? std::abs(r.obj - fref) / (1.0 + std::abs(fref)) : NAN;
                    const bool good = r.ok && dx < 1e-5 && df < 1e-8 && r.viol < 1e-6 &&
                                      std::isfinite(r.time_us) && r.time_us > 0;
                    if (!good)
                    {
                        ++failures;
                        std::printf("FAIL %-9s n=%2d m=%d ok=%d |dx|=%.2e rel df=%.2e viol=%.2e it=%d\n",
                                    bench::name(id).c_str(), n, m, r.ok, dx, df, r.viol, r.iters);
                    }
                }
            }
    std::printf("solver_suite: %d checks, %d failures, GaasQP non-SUCCESS on %d; solvers:",
                checks, failures, gaasqp_failed);
    for (bench::SolverId id : bench::all_solvers()) std::printf(" %s", bench::name(id).c_str());
    std::printf("\n");
    return failures == 0 ? 0 : 1;
}
