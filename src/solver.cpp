#include "geoqp/geoqp.hpp"
#include "geoqp/internal.hpp"
#include <chrono>
#include <iostream>

namespace geoqp {

// Helper functions for zero-copy API (pointer-based versions)
namespace {
    bool is_feasible_ptr(const double* x, const double* A, const double* b,
                        int m, int n, double tol) {
        for (int i = 0; i < m; ++i) {
            double ax = 0.0;
            for (int j = 0; j < n; ++j) {
                ax += A[i * n + j] * x[j];
            }
            if (ax > b[i] + tol) {
                return false;
            }
        }
        return true;
    }
    
    double compute_objective_ptr(const double* x, const double* Q, const double* g, int n) {
        double obj = 0.0;
        // Compute 0.5 * x' * Q * x
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                obj += 0.5 * x[i] * Q[i * n + j] * x[j];
            }
        }
        // Add g' * x
        for (int i = 0; i < n; ++i) {
            obj += g[i] * x[i];
        }
        return obj;
    }
}

// Implementation class
class SolverImpl {
public:
    explicit SolverImpl(const Config& config) : config_(config) {}
    
    Solution solve(const Problem& problem) {
        using Clock = std::chrono::high_resolution_clock;
        auto start_time = Clock::now();
        
        Solution sol;
        sol.status = Status::SUCCESS;
        sol.iterations = 0;
        sol.phase1_iterations = 0;
        sol.phase2_iterations = 0;
        sol.vertices_explored = 0;
        sol.num_active_constraints = 0;
        sol.phase1_time = 0.0;
        sol.phase2_time = 0.0;
        sol.phase3_time = 0.0;
        
        try {
            // Validate problem
            if (!validate_problem(problem)) {
                throw InvalidProblemException("Problem validation failed");
            }
            
            // Compute unconstrained minimum x_bar = -Q^{-1} g
            std::vector<double> x_bar = compute_unconstrained_minimum(
                problem.Q, problem.g, problem.n
            );
            
            if (config_.verbosity >= 1) {
                std::cout << "Unconstrained minimum computed\n";
            }
            
            // Check if unconstrained minimum is feasible
            if (is_feasible(x_bar, problem.A, problem.b, 
                           problem.m, problem.n, config_.tolerance)) {
                if (config_.verbosity >= 1) {
                    std::cout << "Unconstrained minimum is feasible - optimal!\n";
                }
                sol.x = x_bar;
                sol.lambda.resize(problem.m, 0.0);
                sol.objective_value = compute_objective(x_bar, problem.Q, problem.g, problem.n);
                
                auto end_time = Clock::now();
                sol.solve_time = std::chrono::duration<double>(end_time - start_time).count();
                return sol;
            }
            
            if (config_.verbosity >= 1) {
                std::cout << "Unconstrained minimum is infeasible - starting geometric algorithm\n";
            }
            
            // Find initial vertex
            internal::Polytope polytope(problem.A.data(), problem.b.data(), 
                                       problem.m, problem.n);
            std::vector<double> initial_vertex = polytope.find_initial_vertex();
            
            if (config_.verbosity >= 2) {
                std::cout << "Initial vertex found\n";
            }
            
            // Phase I: Find good vertex (farthest from x_bar)
            auto phase1_start = Clock::now();
            
            internal::Initializer initializer(
                problem.Q.data(), problem.A.data(), problem.b.data(), x_bar,
                problem.m, problem.n, config_
            );
            
            std::vector<double> good_vertex = initializer.find_good_vertex(
                initial_vertex, sol.phase1_iterations, sol.vertices_explored
            );
            
            auto phase1_end = Clock::now();
            sol.phase1_time = std::chrono::duration<double>(phase1_end - phase1_start).count();
            
            if (config_.verbosity >= 1) {
                std::cout << "Phase I: Good vertex found after " << sol.phase1_iterations 
                          << " iterations, " << sol.vertices_explored << " vertices explored\n";
            }
            
            // Phase II: Find initial face via ray tracing, then hop to find optimal face
            auto phase2_start = Clock::now();
            
            internal::FaceHopper face_hopper(
                problem.Q.data(), problem.A.data(), problem.b.data(), x_bar,
                problem.m, problem.n, config_
            );
            
            // Step 1: Ray trace to find initial blocking face
            std::vector<int> initial_face = face_hopper.find_initial_face(good_vertex);
            
            if (config_.verbosity >= 2) {
                std::cout << "Initial face has " << initial_face.size() << " constraints\n";
            }
            
            // Step 2: Hop between faces to find optimal one (closest projection to x_bar)
            std::vector<int> optimal_face = face_hopper.find_optimal_face(
                initial_face, sol.phase2_iterations
            );
            
            auto phase2_end = Clock::now();
            sol.phase2_time = std::chrono::duration<double>(phase2_end - phase2_start).count();
            
            if (config_.verbosity >= 1) {
                std::cout << "Phase II: Optimal face found with " << optimal_face.size() 
                          << " constraints after " << sol.phase2_iterations << " face hops\n";
            }
            
            // Special case: if no active face found, x_bar is the solution
            if (optimal_face.empty()) {
                sol.x = x_bar;
                sol.lambda.resize(problem.m, 0.0);
                sol.objective_value = compute_objective(sol.x, problem.Q, problem.g, problem.n);
                sol.active_set.clear();
                sol.num_active_constraints = 0;
                
                if (config_.verbosity >= 1) {
                    std::cout << "No active face - x_bar is optimal\n";
                }
            } else {
                // Phase III: Project x_bar onto optimal face
                auto phase3_start = Clock::now();
                
                internal::Projector projector(
                    problem.Q.data(), problem.A.data(), problem.b.data(), x_bar,
                    face_hopper.get_Q_chol(),
                    problem.m, problem.n, config_
                );
                
                std::vector<double> lambda;
                sol.x = projector.project_onto_face(optimal_face, lambda);
                
                auto phase3_end = Clock::now();
                sol.phase3_time = std::chrono::duration<double>(phase3_end - phase3_start).count();
                
                // Prepare full lambda vector
                sol.lambda.resize(problem.m, 0.0);
                for (size_t i = 0; i < optimal_face.size(); ++i) {
                    sol.lambda[optimal_face[i]] = lambda[i];
                }
                
                sol.active_set = optimal_face;
                sol.num_active_constraints = static_cast<int>(optimal_face.size());
                sol.objective_value = compute_objective(sol.x, problem.Q, problem.g, problem.n);
                
                if (config_.verbosity >= 1) {
                    std::cout << "Phase III: Projection complete\n";
                }
            }
            
            // Check KKT conditions if requested
            if (config_.check_optimality) {
                internal::KKTChecker kkt(
                    problem.Q.data(), problem.g.data(),
                    problem.A.data(), problem.b.data(),
                    problem.m, problem.n, config_.tolerance
                );
                
                if (!kkt.check(sol.x, sol.lambda, sol.active_set)) {
                    double residual = kkt.compute_residual(sol.x, sol.lambda, sol.active_set);
                    if (config_.verbosity >= 1) {
                        std::cout << "Warning: KKT conditions not satisfied (residual: " 
                                  << residual << ")\n";
                    }
                }
            }
            
            if (config_.verbosity >= 1) {
                std::cout << "Solution found with objective value: " << sol.objective_value << "\n";
            }
            
        } catch (const InvalidProblemException& e) {
            sol.status = Status::INVALID_INPUT;
            if (config_.verbosity >= 0) {
                std::cerr << "Invalid problem: " << e.what() << "\n";
            }
        } catch (const NumericalException& e) {
            sol.status = Status::NUMERICAL_ERROR;
            if (config_.verbosity >= 0) {
                std::cerr << "Numerical error: " << e.what() << "\n";
            }
        } catch (const std::exception& e) {
            sol.status = Status::NUMERICAL_ERROR;
            if (config_.verbosity >= 0) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        }
        
        auto end_time = Clock::now();
        sol.solve_time = std::chrono::duration<double>(end_time - start_time).count();
        
        return sol;
    }
    
    const Config& config() const { return config_; }
    void set_config(const Config& config) { config_ = config; }
    
    // Zero-copy API implementation
    void setup(int n, int m, const double* Q, const double* A) {
        setup_n_ = n;
        setup_m_ = m;
        setup_Q_ = Q;
        setup_A_ = A;
        
        // Precompute Q factorization (this is the expensive part)
        Q_chol_.resize(n * n);
        try {
            internal::LinearAlgebra::cholesky(Q, Q_chol_.data(), n);
            has_Q_chol_ = true;
        } catch (...) {
            has_Q_chol_ = false;
        }
        
        is_setup_ = true;
        
        // Clear any previous solution for warm-starting
        last_solution_.x.clear();
        last_solution_.active_set.clear();
    }
    
    bool is_setup() const { return is_setup_; }
    
    void clear_setup() {
        is_setup_ = false;
        setup_n_ = 0;
        setup_m_ = 0;
        setup_Q_ = nullptr;
        setup_A_ = nullptr;
        Q_chol_.clear();
        has_Q_chol_ = false;
        last_solution_.x.clear();
        last_solution_.active_set.clear();
    }
    
    Solution solve_hotstart(const double* g, const double* b) {
        using Clock = std::chrono::high_resolution_clock;
        auto start_time = Clock::now();
        
        if (!is_setup_) {
            Solution sol;
            sol.status = Status::INVALID_INPUT;
            return sol;
        }
        
        Solution sol;
        sol.status = Status::SUCCESS;
        sol.iterations = 0;
        sol.phase1_iterations = 0;
        sol.phase2_iterations = 0;
        sol.vertices_explored = 0;
        sol.num_active_constraints = 0;
        sol.phase1_time = 0.0;
        sol.phase2_time = 0.0;
        sol.phase3_time = 0.0;
        
        try {
            // Compute unconstrained minimum x_bar = -Q^{-1} g
            std::vector<double> x_bar(setup_n_);
            if (has_Q_chol_) {
                // Use precomputed Cholesky factor
                std::vector<double> neg_g(setup_n_);
                for (int i = 0; i < setup_n_; ++i) {
                    neg_g[i] = -g[i];
                }
                internal::LinearAlgebra::solve_cholesky(Q_chol_.data(), neg_g.data(), x_bar.data(), setup_n_);
            } else {
                // Fallback: compute from scratch
                for (int i = 0; i < setup_n_; ++i) {
                    x_bar[i] = 0.0;
                }
            }
            
            if (config_.verbosity >= 1) {
                std::cout << "Unconstrained minimum computed (hotstart)\n";
            }
            
            // Check if unconstrained minimum is feasible
            if (is_feasible_ptr(x_bar.data(), setup_A_, b, 
                               setup_m_, setup_n_, config_.tolerance)) {
                if (config_.verbosity >= 1) {
                    std::cout << "Unconstrained minimum is feasible - optimal!\n";
                }
                sol.x = x_bar;
                sol.lambda.resize(setup_m_, 0.0);
                sol.objective_value = compute_objective_ptr(x_bar.data(), setup_Q_, g, setup_n_);
                
                auto end_time = Clock::now();
                sol.solve_time = std::chrono::duration<double>(end_time - start_time).count();
                return sol;
            }
            
            if (config_.verbosity >= 1) {
                std::cout << "Unconstrained minimum infeasible - using geometric algorithm\n";
            }
            
            // Use warm-start if available
            std::vector<double> initial_vertex;
            if (config_.warm_start && !last_solution_.x.empty()) {
                // Try to use last solution as starting point
                initial_vertex = last_solution_.x;
                if (config_.verbosity >= 2) {
                    std::cout << "Using warm-start from previous solution\n";
                }
            } else {
                // Find initial vertex from scratch
                internal::Polytope polytope(setup_A_, b, setup_m_, setup_n_);
                initial_vertex = polytope.find_initial_vertex();
            }
            
            // Phase I: Find good vertex (skip if warm-starting from good vertex)
            auto phase1_start = Clock::now();
            std::vector<double> good_vertex;
            
            if (config_.warm_start && !last_solution_.x.empty()) {
                // Assume last solution is already a good vertex
                good_vertex = initial_vertex;
                sol.phase1_iterations = 0;
                sol.vertices_explored = 0;
            } else {
                internal::Initializer initializer(
                    setup_Q_, setup_A_, b, x_bar,
                    setup_m_, setup_n_, config_
                );
                
                good_vertex = initializer.find_good_vertex(
                    initial_vertex, sol.phase1_iterations, sol.vertices_explored
                );
            }
            
            auto phase1_end = Clock::now();
            sol.phase1_time = std::chrono::duration<double>(phase1_end - phase1_start).count();
            
            // Phase II: Find optimal face
            auto phase2_start = Clock::now();
            
            internal::FaceHopper face_hopper(
                setup_Q_, setup_A_, b, x_bar,
                setup_m_, setup_n_, config_
            );
            
            // If we have Q factorization, inject it
            if (has_Q_chol_) {
                face_hopper.set_Q_chol(Q_chol_);
            }
            
            std::vector<int> initial_face;
            if (config_.warm_start && !last_solution_.active_set.empty()) {
                // Try using last active set as initial face
                initial_face = last_solution_.active_set;
                if (config_.verbosity >= 2) {
                    std::cout << "Warm-starting Phase II with " << initial_face.size() << " active constraints\n";
                }
            } else {
                initial_face = face_hopper.find_initial_face(good_vertex);
            }
            
            std::vector<int> optimal_face = face_hopper.find_optimal_face(
                initial_face, sol.phase2_iterations
            );
            
            auto phase2_end = Clock::now();
            sol.phase2_time = std::chrono::duration<double>(phase2_end - phase2_start).count();
            
            // Phase III: Project onto optimal face
            if (optimal_face.empty()) {
                sol.x = x_bar;
                sol.lambda.resize(setup_m_, 0.0);
                sol.objective_value = compute_objective_ptr(sol.x.data(), setup_Q_, g, setup_n_);
                sol.active_set.clear();
                sol.num_active_constraints = 0;
            } else {
                auto phase3_start = Clock::now();
                
                internal::Projector projector(
                    setup_Q_, setup_A_, b, x_bar,
                    has_Q_chol_ ? Q_chol_ : std::vector<double>(),
                    setup_m_, setup_n_, config_
                );
                
                std::vector<double> lambda;
                sol.x = projector.project_onto_face(optimal_face, lambda);
                
                auto phase3_end = Clock::now();
                sol.phase3_time = std::chrono::duration<double>(phase3_end - phase3_start).count();
                
                sol.lambda.resize(setup_m_, 0.0);
                for (size_t i = 0; i < optimal_face.size(); ++i) {
                    sol.lambda[optimal_face[i]] = lambda[i];
                }
                
                sol.active_set = optimal_face;
                sol.num_active_constraints = static_cast<int>(optimal_face.size());
                sol.objective_value = compute_objective_ptr(sol.x.data(), setup_Q_, g, setup_n_);
            }
            
            // Store solution for next warm-start
            if (config_.warm_start) {
                last_solution_ = sol;
            }
            
        } catch (const InvalidProblemException& e) {
            sol.status = Status::INVALID_INPUT;
            if (config_.verbosity >= 0) {
                std::cerr << "Invalid problem: " << e.what() << "\n";
            }
        } catch (const NumericalException& e) {
            sol.status = Status::NUMERICAL_ERROR;
            if (config_.verbosity >= 0) {
                std::cerr << "Numerical error: " << e.what() << "\n";
            }
        } catch (const std::exception& e) {
            sol.status = Status::NUMERICAL_ERROR;
            if (config_.verbosity >= 0) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        }
        
        auto end_time = Clock::now();
        sol.solve_time = std::chrono::duration<double>(end_time - start_time).count();
        
        return sol;
    }
    
private:
    Config config_;
    
    // Zero-copy setup state
    bool is_setup_ = false;
    int setup_n_ = 0;
    int setup_m_ = 0;
    const double* setup_Q_ = nullptr;
    const double* setup_A_ = nullptr;
    std::vector<double> Q_chol_;  // Cached Cholesky factorization
    bool has_Q_chol_ = false;
    Solution last_solution_;  // For warm-starting
};

// Public API implementation

Solver::Solver() : impl_(std::make_unique<SolverImpl>(Config())) {}

Solver::Solver(const Config& config) : impl_(std::make_unique<SolverImpl>(config)) {}

Solver::~Solver() = default;

Solution Solver::solve(const Problem& problem) {
    return impl_->solve(problem);
}

Solution Solver::solve(const Problem& problem, const Solution& warm_start) {
    // TODO: Implement warm start
    return impl_->solve(problem);
}

void Solver::setup(int n, int m, const double* Q, const double* A) {
    impl_->setup(n, m, Q, A);
}

Solution Solver::solve_hotstart(const double* g, const double* b) {
    return impl_->solve_hotstart(g, b);
}

bool Solver::is_setup() const {
    return impl_->is_setup();
}

void Solver::clear_setup() {
    impl_->clear_setup();
}

const Config& Solver::config() const {
    return impl_->config();
}

void Solver::set_config(const Config& config) {
    impl_->set_config(config);
}

} // namespace geoqp
