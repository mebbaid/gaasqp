#include "geoqp/c_api.h"
#include "geoqp/geoqp.hpp"
#include <cstring>
#include <new>

extern "C" {

// Convert C++ solver to opaque handle
struct geoqp_solver_t {
    geoqp::Solver* cpp_solver;
};

geoqp_solver_t* geoqp_solver_create() {
    try {
        geoqp_solver_t* solver = new geoqp_solver_t;
        solver->cpp_solver = new geoqp::Solver();
        return solver;
    } catch (...) {
        return nullptr;
    }
}

geoqp_solver_t* geoqp_solver_create_with_config(const geoqp_config_t* config) {
    try {
        geoqp::Config cpp_config;
        cpp_config.tolerance = config->tolerance;
        cpp_config.max_simplex_iterations = config->max_simplex_iterations;
        cpp_config.max_iterations = config->max_iterations;
        cpp_config.warm_start = config->warm_start;
        cpp_config.check_optimality = config->check_optimality;
        cpp_config.verbosity = config->verbosity;
        
        geoqp_solver_t* solver = new geoqp_solver_t;
        solver->cpp_solver = new geoqp::Solver(cpp_config);
        return solver;
    } catch (...) {
        return nullptr;
    }
}

void geoqp_solver_destroy(geoqp_solver_t* solver) {
    if (solver) {
        delete solver->cpp_solver;
        delete solver;
    }
}

geoqp_status_t geoqp_solve(
    geoqp_solver_t* solver,
    const geoqp_problem_t* problem,
    geoqp_solution_t* solution
) {
    if (!solver || !problem || !solution) {
        return GEOQP_INVALID_INPUT;
    }
    
    try {
        // Convert C problem to C++ problem
        geoqp::Problem cpp_problem;
        cpp_problem.n = problem->n;
        cpp_problem.m = problem->m;
        
        cpp_problem.Q.assign(problem->Q, problem->Q + problem->n * problem->n);
        cpp_problem.g.assign(problem->g, problem->g + problem->n);
        cpp_problem.A.assign(problem->A, problem->A + problem->m * problem->n);
        cpp_problem.b.assign(problem->b, problem->b + problem->m);
        
        if (problem->lb) {
            cpp_problem.lb.assign(problem->lb, problem->lb + problem->n);
        }
        if (problem->ub) {
            cpp_problem.ub.assign(problem->ub, problem->ub + problem->n);
        }
        
        // Solve
        geoqp::Solution cpp_solution = solver->cpp_solver->solve(cpp_problem);
        
        // Convert solution back to C
        if (cpp_solution.x.size() == static_cast<size_t>(problem->n)) {
            std::memcpy(solution->x, cpp_solution.x.data(), problem->n * sizeof(double));
        }
        
        if (cpp_solution.lambda.size() == static_cast<size_t>(problem->m)) {
            std::memcpy(solution->lambda, cpp_solution.lambda.data(), problem->m * sizeof(double));
        }
        
        solution->active_set_size = static_cast<int>(cpp_solution.active_set.size());
        if (solution->active_set && solution->active_set_size > 0) {
            std::memcpy(solution->active_set, cpp_solution.active_set.data(), 
                       solution->active_set_size * sizeof(int));
        }
        
        solution->objective_value = cpp_solution.objective_value;
        solution->iterations = cpp_solution.iterations;
        solution->phase1_iterations = cpp_solution.phase1_iterations;
        solution->solve_time = cpp_solution.solve_time;
        
        // Convert status
        switch (cpp_solution.status) {
            case geoqp::Status::SUCCESS:
                solution->status = GEOQP_SUCCESS;
                break;
            case geoqp::Status::UNBOUNDED:
                solution->status = GEOQP_UNBOUNDED;
                break;
            case geoqp::Status::INFEASIBLE:
                solution->status = GEOQP_INFEASIBLE;
                break;
            case geoqp::Status::MAX_ITER:
                solution->status = GEOQP_MAX_ITER;
                break;
            case geoqp::Status::NUMERICAL_ERROR:
                solution->status = GEOQP_NUMERICAL_ERROR;
                break;
            case geoqp::Status::INVALID_INPUT:
                solution->status = GEOQP_INVALID_INPUT;
                break;
            default:
                solution->status = GEOQP_NUMERICAL_ERROR;
        }
        
        return solution->status;
        
    } catch (...) {
        solution->status = GEOQP_NUMERICAL_ERROR;
        return GEOQP_NUMERICAL_ERROR;
    }
}

geoqp_solution_t* geoqp_solution_create(int n, int m) {
    try {
        geoqp_solution_t* solution = new geoqp_solution_t;
        solution->x = new double[n];
        solution->lambda = new double[m];
        solution->active_set = new int[m];  // At most m active constraints
        solution->active_set_size = 0;
        solution->objective_value = 0.0;
        solution->status = GEOQP_SUCCESS;
        solution->iterations = 0;
        solution->phase1_iterations = 0;
        solution->solve_time = 0.0;
        return solution;
    } catch (...) {
        return nullptr;
    }
}

void geoqp_solution_destroy(geoqp_solution_t* solution) {
    if (solution) {
        delete[] solution->x;
        delete[] solution->lambda;
        delete[] solution->active_set;
        delete solution;
    }
}

geoqp_config_t geoqp_get_default_config() {
    geoqp::Config cpp_config;
    geoqp_config_t config;
    config.tolerance = cpp_config.tolerance;
    config.max_simplex_iterations = cpp_config.max_simplex_iterations;
    config.max_iterations = cpp_config.max_iterations;
    config.warm_start = cpp_config.warm_start;
    config.check_optimality = cpp_config.check_optimality;
    config.verbosity = cpp_config.verbosity;
    return config;
}

bool geoqp_validate_problem(const geoqp_problem_t* problem) {
    if (!problem) return false;
    
    try {
        geoqp::Problem cpp_problem;
        cpp_problem.n = problem->n;
        cpp_problem.m = problem->m;
        cpp_problem.Q.assign(problem->Q, problem->Q + problem->n * problem->n);
        cpp_problem.g.assign(problem->g, problem->g + problem->n);
        cpp_problem.A.assign(problem->A, problem->A + problem->m * problem->n);
        cpp_problem.b.assign(problem->b, problem->b + problem->m);
        
        return geoqp::validate_problem(cpp_problem);
    } catch (...) {
        return false;
    }
}

double geoqp_compute_objective(
    const double* x,
    const double* Q,
    const double* g,
    int n
) {
    try {
        std::vector<double> x_vec(x, x + n);
        std::vector<double> Q_vec(Q, Q + n * n);
        std::vector<double> g_vec(g, g + n);
        return geoqp::compute_objective(x_vec, Q_vec, g_vec, n);
    } catch (...) {
        return 0.0;
    }
}

bool geoqp_is_feasible(
    const double* x,
    const double* A,
    const double* b,
    int m, int n,
    double tol
) {
    try {
        std::vector<double> x_vec(x, x + n);
        std::vector<double> A_vec(A, A + m * n);
        std::vector<double> b_vec(b, b + m);
        return geoqp::is_feasible(x_vec, A_vec, b_vec, m, n, tol);
    } catch (...) {
        return false;
    }
}

} // extern "C"
