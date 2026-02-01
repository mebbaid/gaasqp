#include "geoqp/geoqp.hpp"
#include "geoqp/internal.hpp"
#include <chrono>
#include <iostream>

namespace geoqp {

// Implementation class
class SolverImpl {
public:
    explicit SolverImpl(const Config& config) : config_(config) {}
    
    Solution solve(const Problem& problem) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Solution sol;
        sol.status = Status::SUCCESS;
        sol.iterations = 0;
        sol.phase1_iterations = 0;
        
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
                
                auto end_time = std::chrono::high_resolution_clock::now();
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
            
            // Phase I: Find good vertex
            internal::Phase1Solver phase1(
                problem.Q.data(), problem.A.data(), problem.b.data(), x_bar,
                problem.m, problem.n, config_
            );
            
            std::vector<double> good_vertex = phase1.find_good_vertex(
                initial_vertex, sol.phase1_iterations
            );
            
            if (config_.verbosity >= 1) {
                std::cout << "Good vertex found after " << sol.phase1_iterations << " iterations\n";
            }
            
            // Phase II: Find active face
            internal::Phase2Solver phase2(
                problem.A.data(), problem.b.data(), x_bar,
                problem.m, problem.n, config_
            );
            
            std::vector<int> active_face = phase2.find_active_face(good_vertex);
            
            if (config_.verbosity >= 1) {
                std::cout << "Active face identified with " << active_face.size() << " constraints\n";
            }
            
            // Special case: if no active face found, the good vertex itself might be optimal
            // This can happen in low-dimensional cases
            if (active_face.empty()) {
                // Use the good vertex as the solution
                sol.x = good_vertex;
                sol.lambda.resize(problem.m, 0.0);
                sol.objective_value = compute_objective(sol.x, problem.Q, problem.g, problem.n);
                sol.active_set.clear();
                
                if (config_.verbosity >= 1) {
                    std::cout << "No active face found - using good vertex as solution\n";
                }
            } else {
                // Phase III: Project onto face
                internal::Phase3Solver phase3(
                    problem.A.data(), problem.b.data(), x_bar,
                    problem.m, problem.n, config_
                );
                
                std::vector<double> lambda;
                sol.x = phase3.project_onto_face(active_face, lambda);
                
                // Prepare full lambda vector
                sol.lambda.resize(problem.m, 0.0);
                for (size_t i = 0; i < active_face.size(); ++i) {
                    sol.lambda[active_face[i]] = lambda[i];
                }
                
                sol.active_set = active_face;
                sol.objective_value = compute_objective(sol.x, problem.Q, problem.g, problem.n);
            }
            
            // Check KKT conditions if requested
            if (config_.check_optimality) {
                internal::KKTChecker kkt(
                    problem.Q.data(), problem.g.data(),
                    problem.A.data(), problem.b.data(),
                    problem.m, problem.n, config_.tolerance
                );
                
                if (!kkt.check(sol.x, sol.lambda, sol.active_set)) {
                    if (config_.verbosity >= 1) {
                        std::cout << "Warning: KKT conditions not satisfied to tolerance\n";
                    }
                    // Could implement fallback to active-set refinement here
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
        
        auto end_time = std::chrono::high_resolution_clock::now();
        sol.solve_time = std::chrono::duration<double>(end_time - start_time).count();
        
        return sol;
    }
    
    const Config& config() const { return config_; }
    void set_config(const Config& config) { config_ = config; }
    
private:
    Config config_;
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

const Config& Solver::config() const {
    return impl_->config();
}

void Solver::set_config(const Config& config) {
    impl_->set_config(config);
}

} // namespace geoqp
