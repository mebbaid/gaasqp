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
        STANDARD = 0,      // Dantzig pivoting (drop most negative lambda)
        BATCH = 1,         // Batch deactivation
        SENSITIVITY = 2,   // Sensitivity analysis (drop constraint with best improvement)
        NEIGHBORS = 3      // Face neighbor exploration (TODO)
    };

    struct Config
    {
        double tolerance = 1e-12;
        double constraint_tolerance = 1e-8;  // Looser tolerance for constraint feasibility
        int max_simplex_iterations = 1000;
        int max_iterations = 1000;
        bool warm_start = false;
        bool check_optimality = true;
        int verbosity = 0;
        int timing_verbosity = 0;
        // NOTE: Schur complement caching has known numerical issues on ill-conditioned
        // problems; disabled by default.
        bool use_caching = false;
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
        int phase2_iterations;      
        int phase2_facet_hops;      // Facet-adjacency hops (add→drop transitions) in Phase II
        int num_active_constraints; // Number of active constraints at solution
        int vertices_explored;      // Number of vertices explored in Phase 1
        std::vector<double> x_phase1; // Phase I vertex v* (BFS result, before Phase II/III)
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

        Solution solve(const Problem &problem);
        Solution solve(const Problem &problem, const Solution &warm_start);

        // Zero-Copy API for MPC hot-starting
        void setup(int n, int m, const double* Q, const double* A);
        Solution solve_hotstart(const double* g, const double* b);
        bool is_setup() const;
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
