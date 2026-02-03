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

    enum class FaceHoppingStrategy
    {
        STANDARD = 0,      // Standard Dantzig pivoting (drop most negative lambda)
        BATCH = 1,         // Batch deactivation (drop all negative lambdas at once)
        SENSITIVITY = 2,   // Sensitivity analysis (drop constraint with best improvement)
        NEIGHBORS = 3      // Face neighbor exploration (evaluate all adjacent faces)
    };

    struct Config
    {
        double tolerance = 1e-8;
        int max_simplex_iterations = 1000;
        int max_iterations = 1000;
        bool warm_start = false;
        bool check_optimality = true;
        int verbosity = 0;
        // Timing verbosity: 0 = off, 1 = print aggregate timings
        int timing_verbosity = 0;
        // Use incremental Schur complement caching (faster for large k)
        bool use_caching = true;
        // Face hopping strategy for Phase II
        FaceHoppingStrategy face_hopping_strategy = FaceHoppingStrategy::STANDARD;
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
        int phase2_iterations;      // Face exploration iterations
        int num_active_constraints; // Number of active constraints at solution
        int vertices_explored;      // Number of vertices explored in Phase 1
        double solve_time;
        double phase1_time;
        double phase2_time;
        double phase3_time;
    };

    struct Problem
    {
        int n;
        int m;

        // (1/2) x^T Q x + g^T x
        std::vector<double> Q; 
        std::vector<double> g; 

        // A x <= b
        std::vector<double> A; 
        std::vector<double> b; 

        // Optional: bounds on variables
        std::vector<double> lb; 
        std::vector<double> ub; 
    };

    class SolverImpl;

    class Solver
    {
    public:
        Solver();
        explicit Solver(const Config &config);
        ~Solver();

        // Original API (involves vector copies)
        Solution solve(const Problem &problem);
        Solution solve(const Problem &problem, const Solution &warm_start);

        // Zero-Copy API for MPC hot-starting
        // Call setup() once with problem structure (Q, A matrices)
        // Then call solve_hotstart() repeatedly with only changing data (g, b vectors)
        void setup(int n, int m, const double* Q, const double* A);
        Solution solve_hotstart(const double* g, const double* b);
        
        // Check if solver has been set up
        bool is_setup() const;
        
        // Clear setup state
        void clear_setup();

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
