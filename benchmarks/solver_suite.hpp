#pragma once
// Uniform runners for the third-party QP solvers used in the benchmarks.
//
// Every runner solves one dense, condensed QP
//
//     min ½ xᵀHx + gᵀx   s.t.   lb ≤ x ≤ ub,   lbA ≤ A x ≤ ubA
//
// from a cold start, with the same conventions for all solvers:
//   * data is converted to the solver's own format OUTSIDE the timer;
//   * the timer covers setup + solve (what a fresh call in a controller costs),
//     like qpOASES' init();
//   * the objective and the constraint violation are recomputed from the
//     returned x, so they are comparable across solvers;
//   * a bound with |value| ≥ bench::kInf means "no bound" and is mapped to each
//     solver's own infinity.

#include <Eigen/Dense>
#include <limits>
#include <string>
#include <vector>

namespace bench
{
    constexpr double kInf = 1e20;

    struct QP
    {
        Eigen::MatrixXd H;
        Eigen::VectorXd g;
        Eigen::MatrixXd A; // m×n, m may be 0
        Eigen::VectorXd lbA, ubA;
        Eigen::VectorXd lb, ub; // size n

        int n() const { return static_cast<int>(g.size()); }
        int m() const { return static_cast<int>(A.rows()); }
        bool box_only() const { return m() == 0; }
    };

    enum class SolverId
    {
        qpOASES,   // QProblem (general constraints), options setToMPC()
        qpOASES_B, // QProblemB, box-only QPs (Ferreau 2011, Sec. 4.2.1)
        OSQP,      // ADMM, polishing on
        DAQP,      // dual active set
        PIQP,      // proximal interior point, dense backend
        Clarabel,  // interior point (conic), QP as nonnegative-cone problem
    };

    struct Run
    {
        SolverId id{};
        bool ok = false;      // solver reported an optimal solution
        double time_us = std::numeric_limits<double>::quiet_NaN();
        int iters = -1;       // solver's own iteration count
        Eigen::VectorXd x;
        double obj = std::numeric_limits<double>::quiet_NaN();  // ½xᵀHx + gᵀx
        double viol = std::numeric_limits<double>::quiet_NaN(); // max constraint violation
    };

    std::string name(SolverId id);
    // Compiled in and able to take this QP (QProblemB needs a box-only QP).
    bool available(SolverId id);
    bool applicable(SolverId id, const QP &qp);
    // All compiled-in third-party solvers, in a fixed reporting order.
    std::vector<SolverId> all_solvers();

    // tol: absolute = relative optimality/feasibility tolerance where the solver
    // has one (qpOASES is run with its MPC defaults, as in the paper).
    Run run(SolverId id, const QP &qp, double tol);

    double objective(const QP &qp, const Eigen::VectorXd &x);
    double max_violation(const QP &qp, const Eigen::VectorXd &x);

} // namespace bench
