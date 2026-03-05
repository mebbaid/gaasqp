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
#ifdef GEOQP_DEBUG
        inline void print_phase2_debug(int phase2_iterations, int initial_face_size,
                                       const std::vector<int> &initial_face,
                                       const std::vector<double> &good_vertex,
                                       int verbosity)
        {
            if (verbosity < 3)
                return;

            std::ostringstream oss;
            oss << "[DBG] Phase2 iterations=" << phase2_iterations
                << ", initial_face_size=" << initial_face_size << "\n";
            oss << "[DBG] initial_face: ";
            for (int c : initial_face)
                oss << c << " ";
            oss << "\n";
            oss << "[DBG] good_vertex: ";
            for (double xi : good_vertex)
                oss << xi << " ";
            oss << "\n";
            std::cerr << oss.str();
            std::cout << oss.str();
            std::cout.flush();
            std::cerr.flush();
        }

        // Helper: Print warning if Phase II hops are unexpectedly large
        inline void print_large_phase2_warning(int phase2_iterations, int initial_face_size,
                                               int verbosity)
        {
            if (verbosity >= 3 && phase2_iterations > 3)
            {
                std::cerr << "[WARN] Large Phase II hops: " << phase2_iterations
                          << " (initial_face_size=" << initial_face_size << ")\n";
            }
        }

        // Helper: Print ray intersection trace
        inline void print_ray_trace(int best_t_min_idx, double best_t_min, int best_t_max_idx,
                                    double best_t_max, const std::vector<int> &initial_face,
                                    const std::vector<double> &t_vals, int verbosity)
        {
            std::ostringstream oss;
            oss << "[TRACE] Computed ray t in (0,1): t_min_idx=" << best_t_min_idx
                << " t_min=" << best_t_min
                << " , t_max_idx=" << best_t_max_idx << " t_max=" << best_t_max << "\n";
            oss << "[TRACE] initial_face returned: ";
            for (int c : initial_face)
                oss << c << " ";
            oss << "\n";
            std::cerr << oss.str();
            std::cout << oss.str();

            if (verbosity >= 3)
            {
                std::ostringstream oss2;
                oss2 << "[TRACE] constraints with 0<t<1: ";
                for (size_t ci = 0; ci < t_vals.size(); ++ci)
                {
                    double t = t_vals[ci];
                    if (t > 0.0 && t < 1.0)
                        oss2 << ci << ":" << t << " ";
                }
                oss2 << "\n";
                std::cerr << oss2.str();
                std::cout << oss2.str();
            }
        }
#endif

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

        // Workspace members to eliminate per-solve heap allocations
        std::vector<double> ws_good_vertex_;
        std::vector<double> ws_initial_vertex_;
        std::vector<int> ws_initial_face_;
        std::vector<int> ws_optimal_face_;
        std::vector<double> ws_lambda_local_;
        std::vector<double> ws_Ax_bar_;
        std::vector<double> ws_temp_n_; // For Cholesky-based objective computation
        std::vector<double> ws_d_;      // For ray computation in debug paths

        // Workspace for bound-constraint expansion (lb/ub → A/b rows)
        std::vector<double> ws_A_exp_;
        std::vector<double> ws_b_exp_;

        // Expand lb/ub from `prob` into ws_A_exp_ / ws_b_exp_.
        // Returns the expanded constraint count m_exp (>= prob.m).
        // When neither lb nor ub is non-empty, simply aliases prob.A / prob.b
        // with m_exp == prob.m (no copy needed — caller must distinguish).
        int expand_bounds_into_constraints(const Problem &prob)
        {
            const int n   = prob.n;
            const int m   = prob.m;
            const bool has_lb = !prob.lb.empty();
            const bool has_ub = !prob.ub.empty();

            if (!has_lb && !has_ub)
            {
                // No expansion needed — signal to caller via returned m == prob.m
                return m;
            }

            // Count how many bound rows we will add
            int n_lb = 0, n_ub = 0;
            if (has_lb)
                for (int j = 0; j < n; ++j)
                    if (prob.lb[j] > -1e30) ++n_lb;   // finite lower bound
            if (has_ub)
                for (int j = 0; j < n; ++j)
                    if (prob.ub[j] <  1e30) ++n_ub;   // finite upper bound

            const int m_exp = m + n_lb + n_ub;

            ws_A_exp_.assign(static_cast<size_t>(m_exp) * n, 0.0);
            ws_b_exp_.resize(static_cast<size_t>(m_exp));

            // Copy original A, b
            std::copy(prob.A.begin(), prob.A.end(), ws_A_exp_.begin());
            std::copy(prob.b.begin(), prob.b.end(), ws_b_exp_.begin());

            int row = m;

            // Upper bounds:  +eⱼ x ≤ ub[j]
            if (has_ub)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (prob.ub[j] >= 1e30) continue;
                    ws_A_exp_[static_cast<size_t>(row) * n + j] = 1.0;
                    ws_b_exp_[row] = prob.ub[j];
                    ++row;
                }
            }

            // Lower bounds:  -eⱼ x ≤ -lb[j]
            if (has_lb)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (prob.lb[j] <= -1e30) continue;
                    ws_A_exp_[static_cast<size_t>(row) * n + j] = -1.0;
                    ws_b_exp_[row] = -prob.lb[j];
                    ++row;
                }
            }

            return m_exp;
        }

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

                // Expand lb/ub into extra A/b rows (no-op if both are empty).
                const int m_exp = expand_bounds_into_constraints(problem);
                const bool has_bounds = (m_exp > problem.m);
                const double *A_ptr = has_bounds ? ws_A_exp_.data() : problem.A.data();
                const double *b_ptr = has_bounds ? ws_b_exp_.data() : problem.b.data();

                std::vector<double> x_bar = compute_unconstrained_minimum(
                    problem.Q, problem.g, problem.n);

                if (config_.verbosity >= 1)
                {
                    std::cout << "Unconstrained minimum computed\n";
                }

                // Feasibility check against the full (expanded) constraint set.
                if (is_feasible_ptr(x_bar.data(), A_ptr, b_ptr,
                                    m_exp, problem.n, config_.constraint_tolerance))
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

                internal::Polytope polytope(A_ptr, b_ptr, m_exp, problem.n);
                ws_initial_vertex_ = polytope.find_initial_vertex();

                if (config_.verbosity >= 2)
                {
                    std::cout << "Initial vertex found\n";
                }

                auto phase1_start = Clock::now();

                // Factor Q once here so both Initializer (Q-weighted distances / ν)
                // and FaceHopper can share the same Cholesky factor.
                std::vector<double> Q_chol_local(problem.n * problem.n);
                bool has_Q_chol_local = false;
                try
                {
                    internal::LinearAlgebra::cholesky(problem.Q.data(),
                                                       Q_chol_local.data(), problem.n);
                    has_Q_chol_local = true;
                }
                catch (...) {}

                internal::Initializer initializer(
                    problem.Q.data(), A_ptr, b_ptr, x_bar,
                    m_exp, problem.n, config_);

                if (has_Q_chol_local)
                    initializer.set_Q_chol(Q_chol_local);

                ws_good_vertex_ = initializer.find_good_vertex(
                    ws_initial_vertex_, sol.phase1_iterations, sol.vertices_explored);

                // Expose Phase I vertex (v*) for diagnostics / theory verification.
                sol.x_phase1 = ws_good_vertex_;

                auto phase1_end = Clock::now();
                sol.phase1_time = std::chrono::duration<double>(phase1_end - phase1_start).count();

                if (config_.verbosity >= 1)
                {
                    std::cout << "Phase I: Good vertex found after " << sol.phase1_iterations
                              << " iterations, " << sol.vertices_explored << " vertices explored\n";
                }

                auto phase2_start = Clock::now();

                internal::FaceHopper face_hopper(
                    problem.Q.data(), A_ptr, b_ptr, x_bar,
                    m_exp, problem.n, config_);

                // Re-use the Cholesky factor already computed for Phase I.
                if (has_Q_chol_local)
                    face_hopper.set_Q_chol(Q_chol_local);

                ws_initial_face_ = face_hopper.find_initial_face(ws_good_vertex_);

                if (config_.verbosity >= 2)
                {
                    std::cout << "Initial face has " << ws_initial_face_.size() << " constraints\n";
                }

// Debug: When verbose, compute ray intersection parameter t
#ifdef GEOQP_DEBUG
                if (config_.verbosity >= 3)
                {
                    compute_ax_bar(A_ptr, x_bar.data(), m_exp, problem.n);
                    compute_and_print_ray_trace(A_ptr, x_bar, ws_good_vertex_,
                                                m_exp, problem.n);
                }
#endif

                // Phase II: Find optimal face
                ws_optimal_face_ = face_hopper.find_optimal_face(
                    ws_initial_face_, sol.phase2_iterations);

#ifdef GEOQP_DEBUG
                print_phase2_debug(sol.phase2_iterations, static_cast<int>(ws_initial_face_.size()),
                                   ws_initial_face_, ws_good_vertex_, config_.verbosity);
                print_large_phase2_warning(sol.phase2_iterations, static_cast<int>(ws_initial_face_.size()),
                                           config_.verbosity);
#endif

                auto phase2_end = Clock::now();
                sol.phase2_time = std::chrono::duration<double>(phase2_end - phase2_start).count();

                if (config_.verbosity >= 1)
                {
                    std::cout << "Phase II: Optimal face found with " << ws_optimal_face_.size()
                              << " constraints after " << sol.phase2_iterations << " face hops\n";
                }

                if (ws_optimal_face_.empty())
                {
                    sol.x = x_bar;
                    sol.lambda.resize(problem.m, 0.0);
                    sol.objective_value = compute_objective_fast(x_bar.data(), problem.Q.data(),
                                                                 problem.g.data(), problem.n);
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

                    // Use ProjectionSolver directly to leverage Q_inv if available
                    ws_lambda_local_.resize(ws_optimal_face_.size());
                    sol.x.resize(problem.n);

                    bool success = internal::ProjectionSolver::solve(
                        ws_optimal_face_, problem.Q.data(), A_ptr, b_ptr,
                        x_bar, face_hopper.get_Q_chol(),
                        m_exp, problem.n, sol.x, ws_lambda_local_,
                        face_hopper.get_Q_inv(),
                        face_hopper.q_is_diagonal(),
                        face_hopper.get_Q_diag());

                    if (!success)
                    {
                        throw NumericalException("Phase III projection failed");
                    }

                    auto phase3_end = Clock::now();
                    sol.phase3_time = std::chrono::duration<double>(phase3_end - phase3_start).count();

                    // sol.lambda has size problem.m (original constraint count only).
                    // Duals for bound-expansion rows (indices >= problem.m) are internal.
                    sol.lambda.resize(problem.m, 0.0);
                    for (size_t i = 0; i < ws_optimal_face_.size(); ++i)
                    {
                        const int idx = ws_optimal_face_[i];
                        if (idx < problem.m)
                            sol.lambda[idx] = ws_lambda_local_[i];
                    }

                    sol.active_set = ws_optimal_face_;
                    sol.num_active_constraints = static_cast<int>(ws_optimal_face_.size());
                    sol.objective_value = compute_objective_fast(sol.x.data(), problem.Q.data(),
                                                                problem.g.data(), problem.n);

                    if (config_.verbosity >= 1)
                    {
                        std::cout << "Phase III: Projection complete\n";
                    }
                }

                if (config_.check_optimality)
                {
                    // Build a full-size lambda for the KKT checker — it indexes
                    // by active_set[i] which can contain bound-expansion row
                    // indices in [problem.m, m_exp).  sol.lambda stays sized to
                    // problem.m for the public API.
                    std::vector<double> lambda_kkt(m_exp, 0.0);
                    for (size_t i = 0; i < ws_optimal_face_.size(); ++i)
                        lambda_kkt[ws_optimal_face_[i]] = ws_lambda_local_[i];

                    internal::KKTChecker kkt(
                        problem.Q.data(), problem.g.data(),
                        A_ptr, b_ptr,
                        m_exp, problem.n, config_.tolerance);

                    if (!kkt.check(sol.x, lambda_kkt, sol.active_set))
                    {
                        double residual = kkt.compute_residual(sol.x, lambda_kkt, sol.active_set);
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
        }        const Config &config() const { return config_; }
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

            // Preallocate all workspace vectors to avoid per-solve allocations
            ws_x_bar_.resize(n);
            ws_neg_g_.resize(n);
            ws_lambda_.resize(m);
            ws_cholesky_temp_.resize(n);

            ws_good_vertex_.resize(n);
            ws_initial_vertex_.resize(n);
            ws_initial_face_.reserve(m);
            ws_optimal_face_.reserve(m);
            ws_lambda_local_.reserve(m);
            ws_Ax_bar_.resize(m);
            ws_temp_n_.resize(n);
            ws_d_.resize(n);

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
                cached_initializer_->set_Q_chol(Q_chol_);  // enable Q-metric BFS
            }

            cached_projector_ = std::make_unique<internal::Projector>(
                Q, A, dummy_b.data(), dummy_x_bar,
                has_Q_chol_ ? Q_chol_ : std::vector<double>(), m, n, config_,
                &cached_face_hopper_->get_Q_inv());

            last_solution_.x.clear();
            last_solution_.active_set.clear();
        }

        void setup_solve_objects(int n, int m, const double *Q, const double *A)
        {
            // Setup cached objects if not already done, or if dimensions changed
            if (!is_setup_ || setup_n_ != n || setup_m_ != m)
            {
                setup(n, m, Q, A);
            }
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

            ws_good_vertex_.clear();
            ws_initial_vertex_.clear();
            ws_initial_face_.clear();
            ws_optimal_face_.clear();
            ws_lambda_local_.clear();
            ws_Ax_bar_.clear();
            ws_temp_n_.clear();
            ws_d_.clear();

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
                                    setup_m_, setup_n_, config_.constraint_tolerance))
                {
                    if (config_.verbosity >= 1)
                    {
                        std::cout << "Unconstrained minimum is feasible - optimal!\n";
                    }
                    sol.x = ws_x_bar_;
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

                // Compute A*x_bar for reuse
                compute_ax_bar(setup_A_, ws_x_bar_.data(), setup_m_, setup_n_);

                // Update all cached objects with new bounds and x_bar
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

                if (config_.warm_start && !last_solution_.x.empty())
                {
                    ws_good_vertex_ = initial_vertex;
                    sol.phase1_iterations = 0;
                    sol.vertices_explored = 0;
                }
                else
                {
                    ws_good_vertex_ = cached_initializer_->find_good_vertex(
                        initial_vertex, sol.phase1_iterations, sol.vertices_explored);
                }

                auto phase1_end = Clock::now();
                sol.phase1_time = std::chrono::duration<double>(phase1_end - phase1_start).count();

                auto phase2_start = Clock::now();

                if (config_.warm_start && !last_solution_.active_set.empty())
                {
                    ws_initial_face_ = last_solution_.active_set;
                    if (config_.verbosity >= 2)
                    {
                        std::cout << "Warm-starting Phase II with " << ws_initial_face_.size() << " active constraints\n";
                    }
                }
                else
                {
                    ws_initial_face_ = cached_face_hopper_->find_initial_face(ws_good_vertex_);
                }

#ifdef GEOQP_DEBUG
                if (config_.verbosity >= 3)
                {
                    compute_and_print_ray_trace(setup_A_, ws_x_bar_, ws_good_vertex_,
                                                setup_m_, setup_n_);
                }
#endif

                ws_optimal_face_ = cached_face_hopper_->find_optimal_face(
                    ws_initial_face_, sol.phase2_iterations);

                auto phase2_end = Clock::now();
                sol.phase2_time = std::chrono::duration<double>(phase2_end - phase2_start).count();

                if (ws_optimal_face_.empty())
                {
                    sol.x = ws_x_bar_;
                    sol.lambda.resize(setup_m_, 0.0);
                    sol.objective_value = compute_objective_ptr(sol.x.data(), setup_Q_, g, setup_n_);
                    sol.active_set.clear();
                    sol.num_active_constraints = 0;
                }
                else
                {
                    auto phase3_start = Clock::now();

                    // Use cached projector to project onto the optimal face
                    sol.x = cached_projector_->project_onto_face(ws_optimal_face_, ws_lambda_);

                    auto phase3_end = Clock::now();
                    sol.phase3_time = std::chrono::duration<double>(phase3_end - phase3_start).count();

                    sol.lambda.resize(setup_m_, 0.0);
                    for (size_t i = 0; i < ws_optimal_face_.size(); ++i)
                    {
                        sol.lambda[ws_optimal_face_[i]] = ws_lambda_[i];
                    }

                    sol.active_set = ws_optimal_face_;
                    sol.num_active_constraints = static_cast<int>(ws_optimal_face_.size());
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
        // Helper: Compute A*x_bar once for reuse in feasibility checks
        inline void compute_ax_bar(const double *GEOQP_RESTRICT A,
                                   const double *GEOQP_RESTRICT x_bar,
                                   int m, int n)
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
                    sum0 += row[j] * x_bar[j];
                    sum1 += row[j + 1] * x_bar[j + 1];
                    sum2 += row[j + 2] * x_bar[j + 2];
                    sum3 += row[j + 3] * x_bar[j + 3];
                }
                double ax = (sum0 + sum1) + (sum2 + sum3);
                for (; j < n; ++j)
                {
                    ax += row[j] * x_bar[j];
                }
                ws_Ax_bar_[i] = ax;
            }
        }

        // Fast objective computation using Cholesky factor if available
        inline double compute_objective_fast(const double *GEOQP_RESTRICT x,
                                             const double *GEOQP_RESTRICT Q,
                                             const double *GEOQP_RESTRICT g,
                                             int n)
        {
            // Linear part
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

            // Quadratic part - use Cholesky if available
            double quad_obj = 0.0;

            if (has_Q_chol_ && !Q_chol_.empty())
            {
                // Compute y = L * x
                for (int j = 0; j < n; ++j)
                {
                    const double *L_row = Q_chol_.data() + j * n;
                    double y_j = 0.0;
                    for (int k = 0; k <= j; ++k)
                    {
                        y_j += L_row[k] * x[k];
                    }
                    ws_temp_n_[j] = y_j;
                }
                // Compute quad_obj = y^T * y
                for (int j = 0; j < n; ++j)
                {
                    quad_obj += ws_temp_n_[j] * ws_temp_n_[j];
                }
            }
            else
            {
                // Fallback to full quadratic computation
                for (i = 0; i < n; ++i)
                {
                    const double *Q_row = Q + i * n;
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
            }

            return 0.5 * quad_obj + lin_obj;
        }

#ifdef GEOQP_DEBUG
        // Debug helper: Compute and print ray intersection trace
        inline void compute_and_print_ray_trace(const double *GEOQP_RESTRICT A,
                                                const std::vector<double> &x_bar,
                                                const std::vector<double> &good_vertex,
                                                int m, int n)
        {
            // Compute direction d = x_bar - good_vertex
            for (int i = 0; i < n; ++i)
                ws_d_[i] = x_bar[i] - good_vertex[i];

            int best_t_min_idx = -1, best_t_max_idx = -1;
            double best_t_min = 1e300, best_t_max = -1e300;

            std::vector<double> t_vals(m);
            for (int ci = 0; ci < m; ++ci)
            {
                const double *row = A + ci * n;
                double dot_g = 0.0;
                double dot_d = 0.0;
                for (int j = 0; j < n; ++j)
                {
                    dot_g += row[j] * good_vertex[j];
                    dot_d += row[j] * ws_d_[j];
                }
                double numer = ws_Ax_bar_[ci] - dot_g;
                double denom = dot_d;
                double t = std::numeric_limits<double>::infinity();
                if (std::abs(denom) > 1e-18)
                    t = numer / denom;
                t_vals[ci] = t;
                if (t > 0.0 && t < 1.0)
                {
                    if (t < best_t_min)
                    {
                        best_t_min = t;
                        best_t_min_idx = ci;
                    }
                    if (t > best_t_max)
                    {
                        best_t_max = t;
                        best_t_max_idx = ci;
                    }
                }
            }

            print_ray_trace(best_t_min_idx, best_t_min, best_t_max_idx, best_t_max,
                            ws_initial_face_, t_vals, config_.verbosity);
        }
#endif

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
