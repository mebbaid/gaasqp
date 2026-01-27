#ifndef GEOQP_GEOQP_HPP
#define GEOQP_GEOQP_HPP

#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>

namespace geoqp
{

    enum class Status
    {
        SUCCESS = 0,
        UNBOUNDED = 1,
        INFEASIBLE = 2,
        MAX_ITER = 3,
        NUMERICAL_ERROR = 4,
        INVALID_INPUT = 5
    };

    struct Config
    {
        double tolerance = 1e-8;
        int max_simplex_iterations = 1000;
        int max_iterations = 1000;
        bool warm_start = false;
        bool check_optimality = true;
        int verbosity = 0;
    };

    struct Solution
    {
        std::vector<double> x;      // Primal solution
        std::vector<double> lambda; // Dual multipliers
        std::vector<int> active_set;
        double objective_value;
        Status status;
        int iterations;
        int phase1_iterations;
        double solve_time;
    };

    struct Problem
    {
        int n;
        int m;

        // (1/2) x^T Q x + g^T x
        std::vector<double> Q; // n x n symmetric positive definite (row-major)
        std::vector<double> g; // n vector

        // A x <= b
        std::vector<double> A; // m x n matrix (row-major)
        std::vector<double> b; // m vector

        // Optional: bounds on variables
        std::vector<double> lb; // Lower bounds (size n, -inf if empty)
        std::vector<double> ub; // Upper bounds (size n, +inf if empty)
    };

    class SolverImpl;

    class Solver
    {
    public:
        Solver();
        explicit Solver(const Config &config);
        ~Solver();

        Solution solve(const Problem &problem);
        Solution solve(const Problem &problem, const Solution &warm_start);

        const Config &config() const;
        void set_config(const Config &config);

    private:
        std::unique_ptr<SolverImpl> impl_;
    };

    bool validate_problem(const Problem &problem);

    bool is_positive_definite(const std::vector<double> &Q, int n);

    std::vector<double> compute_unconstrained_minimum(
        const std::vector<double> &Q,
        const std::vector<double> &g,
        int n);

    bool is_feasible(
        const std::vector<double> &x,
        const std::vector<double> &A,
        const std::vector<double> &b,
        int m, int n,
        double tol = 1e-8);

    double compute_objective(
        const std::vector<double> &x,
        const std::vector<double> &Q,
        const std::vector<double> &g,
        int n);

    class GeoQPException : public std::runtime_error
    {
    public:
        explicit GeoQPException(const std::string &msg) : std::runtime_error(msg) {}
    };

    class InvalidProblemException : public GeoQPException
    {
    public:
        explicit InvalidProblemException(const std::string &msg)
            : GeoQPException("Invalid problem: " + msg) {}
    };

    class NumericalException : public GeoQPException
    {
    public:
        explicit NumericalException(const std::string &msg)
            : GeoQPException("Numerical error: " + msg) {}
    };

} // namespace geoqp

#endif // GEOQP_GEOQP_HPP
