#include "geoqp/geoqp.hpp"
#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include <chrono>
#include <iostream>
#include <sstream>

namespace geoqp
{

    namespace
    {
        bool is_feasible_ptr(const double *GEOQP_RESTRICT x, const double *GEOQP_RESTRICT A,
                             const double *GEOQP_RESTRICT b, int m, int n, double tol)
        {
            for (int i = 0; i < m; ++i)
            {
                const double *GEOQP_RESTRICT row = A + i * n;

                double sum0 = 0.0;
                double sum1 = 0.0;
                double sum2 = 0.0;
                double sum3 = 0.0;

                int j = 0;
                for (; j <= n - 4; j += 4)
                {
                    sum0 += row[j] * x[j];
                    sum1 += row[j + 1] * x[j + 1];
                    sum2 += row[j + 2] * x[j + 2];
                    sum3 += row[j + 3] * x[j + 3];
                }
                double ax = (sum0 + sum1) + (sum2 + sum3);
                for (; j < n; ++j)
                {
                    ax += row[j] * x[j];
                }

                if (ax > b[i] + tol)
                {
                    return false;
                }
            }
            return true;
        }

        double compute_objective_ptr(const double *GEOQP_RESTRICT x,
                                     const double *GEOQP_RESTRICT Q,
                                     const double *GEOQP_RESTRICT g,
                                     int n)
        {

            double lin_sum0 = 0.0;
            double lin_sum1 = 0.0;
            double lin_sum2 = 0.0;
            double lin_sum3 = 0.0;

            int i = 0;
            for (; i <= n - 4; i += 4)
            {
                lin_sum0 += g[i] * x[i];
                lin_sum1 += g[i + 1] * x[i + 1];
                lin_sum2 += g[i + 2] * x[i + 2];
                lin_sum3 += g[i + 3] * x[i + 3];
            }
            double lin_obj = (lin_sum0 + lin_sum1) + (lin_sum2 + lin_sum3);
            for (; i < n; ++i)
            {
                lin_obj += g[i] * x[i];
            }

            double quad_obj = 0.0;

            for (i = 0; i < n; ++i)
            {
                const double *GEOQP_RESTRICT Q_row = Q + i * n;
                const double xi = x[i];

                double row_sum0 = 0.0;
                double row_sum1 = 0.0;
                double row_sum2 = 0.0;
                double row_sum3 = 0.0;

                int j = 0;
                for (; j <= n - 4; j += 4)
                {
                    row_sum0 += Q_row[j] * x[j];
                    row_sum1 += Q_row[j + 1] * x[j + 1];
                    row_sum2 += Q_row[j + 2] * x[j + 2];
                    row_sum3 += Q_row[j + 3] * x[j + 3];
                }

                double row_dot = (row_sum0 + row_sum1) + (row_sum2 + row_sum3);

                for (; j < n; ++j)
                {
                    row_dot += Q_row[j] * x[j];
                }

                quad_obj += xi * row_dot;
            }

            return 0.5 * quad_obj + lin_obj;
        }
    }

    class SolverImpl
    {
    public:
        explicit SolverImpl(const Config &config) : config_(config) {}

        Solution solve(const Problem &problem)
        {
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

            try
            {
                if (!validate_problem(problem))
                {
                    throw InvalidProblemException("Problem validation failed");
                }

                std::vector<double> x_bar = compute_unconstrained_minimum(
                    problem.Q, problem.g, problem.n);

                if (config_.verbosity >= 1)
                {
                    std::cout << "Unconstrained minimum computed\n";
                }

                if (is_feasible(x_bar, problem.A, problem.b,
                                problem.m, problem.n, config_.constraint_tolerance))
                {
                    if (config_.verbosity >= 1)
                    {
                        std::cout << "Unconstrained minimum is feasible - optimal!\n";
                    }
                    sol.x = x_bar;
                    sol.lambda.resize(problem.m, 0.0);
                    sol.objective_value = compute_objective(x_bar, problem.Q, problem.g, problem.n);

                    auto end_time = Clock::now();
                    sol.solve_time = std::chrono::duration<double>(end_time - start_time).count();
                    return sol;
                }

                if (config_.verbosity >= 1)
                {
                    std::cout << "Unconstrained minimum is infeasible - starting geometric algorithm\n";
                }

                internal::Polytope polytope(problem.A.data(), problem.b.data(),
                                            problem.m, problem.n);
                std::vector<double> initial_vertex = polytope.find_initial_vertex();

                if (config_.verbosity >= 2)
                {
                    std::cout << "Initial vertex found\n";
                }

                auto phase1_start = Clock::now();

                internal::Initializer initializer(
                    problem.Q.data(), problem.A.data(), problem.b.data(), x_bar,
                    problem.m, problem.n, config_);

                std::vector<double> good_vertex = initializer.find_good_vertex(
                    initial_vertex, sol.phase1_iterations, sol.vertices_explored);

                auto phase1_end = Clock::now();
                sol.phase1_time = std::chrono::duration<double>(phase1_end - phase1_start).count();

                if (config_.verbosity >= 1)
                {
                    std::cout << "Phase I: Good vertex found after " << sol.phase1_iterations
                              << " iterations, " << sol.vertices_explored << " vertices explored\n";
                }

                auto phase2_start = Clock::now();

                internal::FaceHopper face_hopper(
                    problem.Q.data(), problem.A.data(), problem.b.data(), x_bar,
                    problem.m, problem.n, config_);

                std::vector<int> initial_face = face_hopper.find_initial_face(good_vertex);

                if (config_.verbosity >= 2)
                {
                    std::cout << "Initial face has " << initial_face.size() << " constraints\n";
                }

                std::vector<int> optimal_face = face_hopper.find_optimal_face(
                    initial_face, sol.phase2_iterations);

                // Debug: per-solve Phase II details when requested
                const char *dbg_phase2 = std::getenv("GEOQP_DBG_PHASE2");
                if (dbg_phase2 && std::string(dbg_phase2) == "1")
                {
                    std::ostringstream oss;
                    oss << "[DBG] Phase2 iterations=" << sol.phase2_iterations
                        << ", initial_face_size=" << initial_face.size() << "\n";
                    oss << "[DBG] initial_face: ";
                    for (int c : initial_face) oss << c << " ";
                    oss << "\n";
                    oss << "[DBG] good_vertex: ";
                    for (double xi : good_vertex) oss << xi << " ";
                    oss << "\n";
                    // Print to both stderr and stdout to ensure visibility
                    std::cerr << oss.str();
                    std::cout << oss.str();
                    std::cout.flush();
                    std::cerr.flush();
                }

                // If phase2_iterations is unexpectedly large, always print a short diagnostic
                if (sol.phase2_iterations > 3)
                {
                    std::cerr << "[WARN] Large Phase II hops: " << sol.phase2_iterations
                              << " (initial_face_size=" << initial_face.size() << ")\n";
                }

                auto phase2_end = Clock::now();
                sol.phase2_time = std::chrono::duration<double>(phase2_end - phase2_start).count();

                // Debug: per-solve Phase II details when requested (hotstart path)
                const char *dbg_phase2_hs = std::getenv("GEOQP_DBG_PHASE2");
                if (dbg_phase2_hs && std::string(dbg_phase2_hs) == "1")
                {
                    std::ostringstream oss_hs;
                    oss_hs << "[DBG] Phase2 iterations=" << sol.phase2_iterations
                           << ", initial_face_size=" << initial_face.size() << "\n";
                    oss_hs << "[DBG] initial_face: ";
                    for (int c : initial_face) oss_hs << c << " ";
                    oss_hs << "\n";
                    oss_hs << "[DBG] good_vertex: ";
                    for (double xi : good_vertex) oss_hs << xi << " ";
                    oss_hs << "\n";
                    std::cerr << oss_hs.str();
                    std::cout << oss_hs.str();
                    std::cout.flush();
                    std::cerr.flush();
                }

                // If phase2_iterations is unexpectedly large, always print a short diagnostic
                if (sol.phase2_iterations > 3)
                {
                    std::cerr << "[WARN] Large Phase II hops: " << sol.phase2_iterations
                              << " (initial_face_size=" << initial_face.size() << ")\n";
                }

                if (config_.verbosity >= 1)
                {
                    std::cout << "Phase II: Optimal face found with " << optimal_face.size()
                              << " constraints after " << sol.phase2_iterations << " face hops\n";
                }

                if (optimal_face.empty())
                {
                    sol.x = x_bar;
                    sol.lambda.resize(problem.m, 0.0);
                    sol.objective_value = compute_objective(sol.x, problem.Q, problem.g, problem.n);
                    sol.active_set.clear();
                    sol.num_active_constraints = 0;

                    if (config_.verbosity >= 1)
                    {
                        std::cout << "No active face - x_bar is optimal\n";
                    }
                }
                else
                {
                    auto phase3_start = Clock::now();

                    internal::Projector projector(
                        problem.Q.data(), problem.A.data(), problem.b.data(), x_bar,
                        face_hopper.get_Q_chol(),
                        problem.m, problem.n, config_,
                        &face_hopper.get_Q_inv());

                    std::vector<double> lambda;
                    sol.x.resize(problem.n);
                    
                    // Use ProjectionSolver directly instead of Projector to leverage Q_inv
                    bool success = internal::ProjectionSolver::solve(
                        optimal_face, problem.Q.data(), problem.A.data(), problem.b.data(), 
                        x_bar, face_hopper.get_Q_chol(),
                        problem.m, problem.n, sol.x, lambda,
                        face_hopper.get_Q_inv(),
                        face_hopper.q_is_diagonal(), face_hopper.get_Q_diag());

                    if (!success) {
                        throw NumericalException("Phase III projection failed");
                    }

                    auto phase3_end = Clock::now();
                    sol.phase3_time = std::chrono::duration<double>(phase3_end - phase3_start).count();

                    sol.lambda.resize(problem.m, 0.0);
                    for (size_t i = 0; i < optimal_face.size(); ++i)
                    {
                        sol.lambda[optimal_face[i]] = lambda[i];
                    }

                    sol.active_set = optimal_face;
                    sol.num_active_constraints = static_cast<int>(optimal_face.size());
                    sol.objective_value = compute_objective(sol.x, problem.Q, problem.g, problem.n);

                    if (config_.verbosity >= 1)
                    {
                        std::cout << "Phase III: Projection complete\n";
                    }
                }

                if (config_.check_optimality)
                {
                    internal::KKTChecker kkt(
                        problem.Q.data(), problem.g.data(),
                        problem.A.data(), problem.b.data(),
                        problem.m, problem.n, config_.tolerance);

                    if (!kkt.check(sol.x, sol.lambda, sol.active_set))
                    {
                        double residual = kkt.compute_residual(sol.x, sol.lambda, sol.active_set);
                        if (config_.verbosity >= 1)
                        {
                            std::cout << "Warning: KKT conditions not satisfied (residual: "
                                      << residual << ")\n";
                        }
                    }
                }

                if (config_.verbosity >= 1)
                {
                    std::cout << "Solution found with objective value: " << sol.objective_value << "\n";
                }
            }
            catch (const InvalidProblemException &e)
            {
                sol.status = Status::INVALID_INPUT;
                if (config_.verbosity >= 0)
                {
                    std::cerr << "Invalid problem: " << e.what() << "\n";
                }
            }
            catch (const NumericalException &e)
            {
                sol.status = Status::NUMERICAL_ERROR;
                if (config_.verbosity >= 0)
                {
                    std::cerr << "Numerical error: " << e.what() << "\n";
                }
            }
            catch (const std::exception &e)
            {
                sol.status = Status::NUMERICAL_ERROR;
                if (config_.verbosity >= 0)
                {
                    std::cerr << "Error: " << e.what() << "\n";
                }
            }

            auto end_time = Clock::now();
            sol.solve_time = std::chrono::duration<double>(end_time - start_time).count();

            return sol;
        }

        const Config &config() const { return config_; }
        void set_config(const Config &config) { config_ = config; }

        void setup(int n, int m, const double *Q, const double *A)
        {
            setup_n_ = n;
            setup_m_ = m;
            setup_Q_ = Q;
            setup_A_ = A;

            Q_chol_.resize(n * n);
            try
            {
                internal::LinearAlgebra::cholesky(Q, Q_chol_.data(), n);
                has_Q_chol_ = true;
            }
            catch (...)
            {
                has_Q_chol_ = false;
            }

            is_setup_ = true;

            ws_x_bar_.resize(n);
            ws_neg_g_.resize(n);
            ws_lambda_.resize(m);
            ws_cholesky_temp_.resize(n);

            std::vector<double> dummy_b(m, 0.0);
            std::vector<double> dummy_x_bar(n, 0.0);

            cached_polytope_ = std::make_unique<internal::Polytope>(A, dummy_b.data(), m, n);
            cached_initializer_ = std::make_unique<internal::Initializer>(
                Q, A, dummy_b.data(), dummy_x_bar, m, n, config_);
            cached_face_hopper_ = std::make_unique<internal::FaceHopper>(
                Q, A, dummy_b.data(), dummy_x_bar, m, n, config_);

            if (has_Q_chol_)
            {
                cached_face_hopper_->set_Q_chol(Q_chol_);
            }

            cached_projector_ = std::make_unique<internal::Projector>(
                Q, A, dummy_b.data(), dummy_x_bar,
                has_Q_chol_ ? Q_chol_ : std::vector<double>(), m, n, config_,
                &cached_face_hopper_->get_Q_inv());

            last_solution_.x.clear();
            last_solution_.active_set.clear();
        }

        bool is_setup() const { return is_setup_; }

        void clear_setup()
        {
            is_setup_ = false;
            setup_n_ = 0;
            setup_m_ = 0;
            setup_Q_ = nullptr;
            setup_A_ = nullptr;
            Q_chol_.clear();
            has_Q_chol_ = false;
            last_solution_.x.clear();
            last_solution_.active_set.clear();
            ws_x_bar_.clear();
            ws_neg_g_.clear();
            ws_lambda_.clear();
            ws_cholesky_temp_.clear();

            cached_polytope_.reset();
            cached_initializer_.reset();
            cached_face_hopper_.reset();
            cached_projector_.reset();
        }

        Solution solve_hotstart(const double *g, const double *b)
        {
            using Clock = std::chrono::high_resolution_clock;
            auto start_time = Clock::now();

            if (!is_setup_)
            {
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

            try
            {
                if (has_Q_chol_)
                {
                    for (int i = 0; i < setup_n_; ++i)
                    {
                        ws_neg_g_[i] = -g[i];
                    }
                    internal::LinearAlgebra::solve_cholesky(Q_chol_.data(), ws_neg_g_.data(),
                                                            ws_x_bar_.data(), ws_cholesky_temp_.data(), setup_n_);
                }
                else
                {
                    for (int i = 0; i < setup_n_; ++i)
                    {
                        ws_x_bar_[i] = 0.0;
                    }
                }

                if (config_.verbosity >= 1)
                {
                    std::cout << "Unconstrained minimum computed (hotstart)\n";
                }

                if (is_feasible_ptr(ws_x_bar_.data(), setup_A_, b,
                                    setup_m_, setup_n_, config_.tolerance))
                {
                    if (config_.verbosity >= 1)
                    {
                        std::cout << "Unconstrained minimum is feasible - optimal!\n";
                    }
                    sol.x.resize(setup_n_);
                    std::copy(ws_x_bar_.begin(), ws_x_bar_.end(), sol.x.begin());
                    sol.lambda.resize(setup_m_, 0.0);
                    sol.objective_value = compute_objective_ptr(ws_x_bar_.data(), setup_Q_, g, setup_n_);

                    auto end_time = Clock::now();
                    sol.solve_time = std::chrono::duration<double>(end_time - start_time).count();
                    return sol;
                }

                if (config_.verbosity >= 1)
                {
                    std::cout << "Unconstrained minimum infeasible - using geometric algorithm\n";
                }

                cached_polytope_->update_bounds(b);
                cached_initializer_->update(b, ws_x_bar_);
                cached_face_hopper_->update(b, ws_x_bar_);
                cached_projector_->update(b, ws_x_bar_);

                std::vector<double> initial_vertex;
                if (config_.warm_start && !last_solution_.x.empty())
                {
                    initial_vertex = last_solution_.x;
                    if (config_.verbosity >= 2)
                    {
                        std::cout << "Using warm-start from previous solution\n";
                    }
                }
                else
                {
                    initial_vertex = cached_polytope_->find_initial_vertex();
                }

                auto phase1_start = Clock::now();
                std::vector<double> good_vertex;

                if (config_.warm_start && !last_solution_.x.empty())
                {
                    good_vertex = initial_vertex;
                    sol.phase1_iterations = 0;
                    sol.vertices_explored = 0;
                }
                else
                {
                    good_vertex = cached_initializer_->find_good_vertex(
                        initial_vertex, sol.phase1_iterations, sol.vertices_explored);
                }

                auto phase1_end = Clock::now();
                sol.phase1_time = std::chrono::duration<double>(phase1_end - phase1_start).count();

                auto phase2_start = Clock::now();

                std::vector<int> initial_face;
                if (config_.warm_start && !last_solution_.active_set.empty())
                {
                    initial_face = last_solution_.active_set;
                    if (config_.verbosity >= 2)
                    {
                        std::cout << "Warm-starting Phase II with " << initial_face.size() << " active constraints\n";
                    }
                }
                else
                {
                    initial_face = cached_face_hopper_->find_initial_face(good_vertex);
                }

                std::vector<int> optimal_face = cached_face_hopper_->find_optimal_face(
                    initial_face, sol.phase2_iterations);

                auto phase2_end = Clock::now();
                sol.phase2_time = std::chrono::duration<double>(phase2_end - phase2_start).count();

                if (optimal_face.empty())
                {
                    sol.x.resize(setup_n_);
                    std::copy(ws_x_bar_.begin(), ws_x_bar_.end(), sol.x.begin());
                    sol.lambda.resize(setup_m_, 0.0);
                    sol.objective_value = compute_objective_ptr(sol.x.data(), setup_Q_, g, setup_n_);
                    sol.active_set.clear();
                    sol.num_active_constraints = 0;
                }
                else
                {
                    auto phase3_start = Clock::now();

                    // If the face is the same as the one from Phase II, we can reuse the result
                    sol.x = cached_projector_->project_onto_face(optimal_face, ws_lambda_);

                    auto phase3_end = Clock::now();
                    sol.phase3_time = std::chrono::duration<double>(phase3_end - phase3_start).count();

                    sol.lambda.resize(setup_m_, 0.0);
                    for (size_t i = 0; i < optimal_face.size(); ++i)
                    {
                        sol.lambda[optimal_face[i]] = ws_lambda_[i];
                    }

                    sol.active_set = optimal_face;
                    sol.num_active_constraints = static_cast<int>(optimal_face.size());
                    sol.objective_value = compute_objective_ptr(sol.x.data(), setup_Q_, g, setup_n_);
                }

                if (config_.warm_start)
                {
                    last_solution_ = sol;
                }
            }
            catch (const InvalidProblemException &e)
            {
                sol.status = Status::INVALID_INPUT;
                if (config_.verbosity >= 0)
                {
                    std::cerr << "Invalid problem: " << e.what() << "\n";
                }
            }
            catch (const NumericalException &e)
            {
                sol.status = Status::NUMERICAL_ERROR;
                if (config_.verbosity >= 0)
                {
                    std::cerr << "Numerical error: " << e.what() << "\n";
                }
            }
            catch (const std::exception &e)
            {
                sol.status = Status::NUMERICAL_ERROR;
                if (config_.verbosity >= 0)
                {
                    std::cerr << "Error: " << e.what() << "\n";
                }
            }

            auto end_time = Clock::now();
            sol.solve_time = std::chrono::duration<double>(end_time - start_time).count();

            return sol;
        }

    private:
        Config config_;

        bool is_setup_ = false;
        int setup_n_ = 0;
        int setup_m_ = 0;
        const double *setup_Q_ = nullptr;
        const double *setup_A_ = nullptr;
        std::vector<double> Q_chol_;
        bool has_Q_chol_ = false;
        Solution last_solution_;

        std::vector<double> ws_x_bar_;
        std::vector<double> ws_neg_g_;
        std::vector<double> ws_lambda_;
        std::vector<double> ws_cholesky_temp_;

        std::unique_ptr<internal::Polytope> cached_polytope_;
        std::unique_ptr<internal::Initializer> cached_initializer_;
        std::unique_ptr<internal::FaceHopper> cached_face_hopper_;
        std::unique_ptr<internal::Projector> cached_projector_;
    };

    Solver::Solver() : impl_(std::make_unique<SolverImpl>(Config())) {}

    Solver::Solver(const Config &config) : impl_(std::make_unique<SolverImpl>(config)) {}

    Solver::~Solver() = default;

    Solution Solver::solve(const Problem &problem)
    {
        return impl_->solve(problem);
    }

    Solution Solver::solve(const Problem &problem, const Solution &warm_start)
    {
        // TODO: Implement warm start
        return impl_->solve(problem);
    }

    void Solver::setup(int n, int m, const double *Q, const double *A)
    {
        impl_->setup(n, m, Q, A);
    }

    Solution Solver::solve_hotstart(const double *g, const double *b)
    {
        return impl_->solve_hotstart(g, b);
    }

    bool Solver::is_setup() const
    {
        return impl_->is_setup();
    }

    void Solver::clear_setup()
    {
        impl_->clear_setup();
    }

    const Config &Solver::config() const
    {
        return impl_->config();
    }

    void Solver::set_config(const Config &config)
    {
        impl_->set_config(config);
    }

} // namespace geoqp
