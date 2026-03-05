#ifndef GEOQP_INTERNAL_HPP
#define GEOQP_INTERNAL_HPP

#include "geoqp/geoqp.hpp"
#include "geoqp/compiler_hints.hpp"
#include <vector>
#include <set>
#include <chrono>
#include <iostream>

namespace geoqp
{
    namespace internal
    {
        struct Timer
        {
            using Clock = std::chrono::high_resolution_clock;
            using TimePoint = Clock::time_point;

            TimePoint start;

            Timer() : start(Clock::now()) {}

            double elapsed_ms() const
            {
                auto end = Clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                return elapsed.count();
            }

            void reset()
            {
                start = Clock::now();
            }
        };

        class LinearAlgebra
        {
        public:
            static void matvec(
                const double *A, const double *x, double *y,
                int m, int n);
            static double dot(const double *x, const double *y, int n);
            static double norm(const double *x, int n);

            static void solve_linear_system(
                const double *A, const double *b, double *x,
                int n);
            static void cholesky(const double *A, double *L, int n);

            static void forward_sub(
                const double *L, const double *b, double *x,
                int n);
            static void backward_sub(
                const double *L, const double *b, double *x,
                int n);
            static void solve_cholesky(
                const double *L, const double *b, double *x,
                int n);
            static void solve_cholesky(
                const double *L, const double *b, double *x,
                double *workspace, int n);
            static void solve_pd_system(
                const double *Q, const double *b, double *x,
                int n);
            static bool cholesky_rank1_update(double *L, const double *v, int k);
            static bool cholesky_extend(
                const double *L, int k,
                const double *s, double sigma,
                double *L_new);
            static bool cholesky_shrink(
                const double *L, int k, int idx,
                double *L_new);
        };

        class Polytope
        {
        public:
            Polytope(const double *A, const double *b, int m, int n);
            void update_bounds(const double *b) { b_ = b; }
            std::vector<double> find_initial_vertex() const;
            std::vector<std::vector<double>> enumerate_vertices() const;
            bool is_vertex(const std::vector<double> &x, double tol = 1e-8) const;
            std::vector<int> get_active_constraints(
                const std::vector<double> &x,
                double tol = 1e-8) const;

            std::vector<std::vector<double>> get_neighbors(
                const std::vector<double> &vertex) const;

            // Append neighbor vertices as contiguous blocks (each block length = n) into out_flat
            void get_neighbors_flat(const std::vector<double> &vertex, std::vector<double> &out_flat) const;

        private:
            const double *A_;
            const double *b_;
            int m_;
            int n_;
        };

        class Initializer
        {
        public:
            Initializer(
                const double *Q, const double *A, const double *b,
                const std::vector<double> &x_bar,
                int m, int n,
                const Config &config);

            void update(const double *b, const std::vector<double> &x_bar);

            // Supply the Cholesky factor L of Q (Q = L Lᵀ) to enable
            // Q-weighted distances and the ν-biased BFS, as required by
            // Theorem 1 / Lemma 2 in the theory. No-op if not called.
            void set_Q_chol(const std::vector<double> &Q_chol);

            const std::vector<double> &get_Axbar() const { return ws_Axbar_; }

            std::vector<double> find_good_vertex(
                const std::vector<double> &initial_vertex,
                int &iterations,
                int &vertices_explored);

        private:
            void compute_axbar();
            void detect_box_constraints();

            bool is_good_vertex(
                const std::vector<double> &v,
                const std::vector<int> &active_set) const;

            // Q-weighted squared distance ‖v - x̄‖_Q² = (v-x̄)ᵀ Q (v-x̄).
            // Uses the Cholesky factor when available (‖L(v-x̄)‖₂²) to avoid
            // re-factoring Q on every call.
            double compute_q_weighted_distance_squared_ptr(const double *vptr) const;

            std::vector<double> compute_vertex_from_active_set(
                const std::vector<int> &active_indices) const;

            std::vector<double> find_good_vertex_box_fast() const;

            // Compute the separation direction ν ∝ Q(x̄ - x̂) once per
            // find_good_vertex call and store in ws_nu_.  Returns the
            // distance L = ‖x̄ - x̂‖_Q used as the linear-proxy scaling.
            double compute_separation_direction(const double *closest_boundary_pt) const;

            // BFS vertex score from Lemma 2:
            //   score(v) = ‖v - x̄‖_Q²  -  2·L·⟨ν, v⟩
            // Maximising this finds v* (up to constants independent of v).
            inline double bfs_score(const double *vptr, double two_L) const;

            const double *Q_;
            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
            int m_;
            int n_;
            Config config_;

            // Cholesky factor L of Q (lower-triangular), set externally.
            // When present enables O(n²) Q-weighted distances via ‖L·d‖₂².
            std::vector<double> Q_chol_;
            bool has_Q_chol_ = false;

            bool is_box_constrained_ = false;
            std::vector<double> box_lb_;
            std::vector<double> box_ub_;
            std::vector<bool> has_lb_;
            std::vector<bool> has_ub_;

            mutable std::vector<double> ws_Axbar_;
            mutable std::vector<double> ws_direction_;
            mutable std::vector<double> ws_Ax_;
            mutable std::vector<double> ws_Ad_;
            mutable std::vector<double> ws_candidate_;
            // Workspace for Q-weighted distance (L·diff scratch) and ν.
            mutable std::vector<double> ws_Ldiff_;  // size n
            mutable std::vector<double> ws_nu_;     // separation direction ν, size n
        };

        class ConstraintCache
        {
        public:
            ConstraintCache(int m, int n) : m_(m), n_(n), cache_(m) {}

            const std::vector<double> &get_z(
                int constraint_idx,
                const double *A,
                const double *L_Q);

            void clear()
            {
                for (auto &entry : cache_)
                {
                    entry.computed = false;
                }
            }

        private:
            struct Entry
            {
                std::vector<double> z;
                bool computed = false;
            };

            int m_, n_;
            std::vector<Entry> cache_;
        };

        class SchurComplementManager
        {
        public:
            SchurComplementManager() : k_(0) {}

            bool initialize(
                const std::vector<int> &face,
                const double *A,
                const double *L_Q,
                int n,
                ConstraintCache &cache);

            bool add_constraint(
                int constraint_idx,
                const double *A,
                const double *L_Q,
                int n,
                ConstraintCache &cache);
            bool remove_constraint(int face_index);

            bool solve(const std::vector<double> &rhs, std::vector<double> &lambda) const;

            double get_L_diagonal(int i) const
            {
                if (i >= k_)
                    return 0.0;
                return L_S_[i * k_ + i];
            }

            const std::vector<int> &get_active_constraints() const { return active_constraints_; }
            int size() const { return k_; }

        private:
            std::vector<double> L_S_;
            std::vector<int> active_constraints_;
            int k_;
            const double *A_ptr_ = nullptr;
            const double *L_Q_ptr_ = nullptr;
            int n_ = 0;
            ConstraintCache *cache_ptr_ = nullptr;
            struct SchurStats
            {
                int initialize_calls = 0;
                int extend_success = 0;
                int extend_fallback = 0;
                int shrink_success = 0;
                int shrink_fallback = 0;
                int solve_calls = 0;
            } mutable stats_;
        };

        struct SchurStatsGlobal
        {
            int initialize_calls = 0;
            int extend_success = 0;
            int extend_fallback = 0;
            int shrink_success = 0;
            int shrink_fallback = 0;
            int solve_calls = 0;
        };

        void schur_stats_reset();
        SchurStatsGlobal &schur_stats_get();

        class FaceHopper
        {
        public:
            FaceHopper(
                const double *Q, const double *A, const double *b,
                const std::vector<double> &x_bar,
                int m, int n,
                const Config &config);

            // Overload: accept a pre-computed Cholesky factor of Q to skip
            // the O(n³) factorization inside the constructor.
            FaceHopper(
                const double *Q, const double *A, const double *b,
                const std::vector<double> &x_bar,
                int m, int n,
                const Config &config,
                const std::vector<double> &Q_chol);

            void update(const double *b, const std::vector<double> &x_bar)
            {
                b_ = b;
                x_bar_ = x_bar;
            }

            std::vector<int> find_initial_face(
                const std::vector<double> &v);

            std::vector<int> find_optimal_face(
                const std::vector<int> &initial_face,
                int &iterations);

            const std::vector<double> &get_last_x() const { return ws_x_star_; }
            const std::vector<double> &get_last_lambda() const { return ws_lambda_; }

            const std::vector<double> &get_Q_chol() const { return Q_chol_; }
            const std::vector<double> &get_Q_inv() const { return Q_inv_; }
            const std::vector<double> &get_Q_diag() const { return Q_diag_; }
            bool q_is_diagonal() const { return q_is_diagonal_; }
            bool has_diagonal_constraints() const { return has_diagonal_constraints_; }
            const std::vector<int>    &get_constraint_var()  const { return constraint_var_; }
            const std::vector<double> &get_constraint_sign() const { return constraint_sign_; }

            void set_Q_chol(const std::vector<double> &Q_chol) { Q_chol_ = Q_chol; }

            // Accept a pre-computed A·x̄ vector (avoids redundant O(m·n) matvec).
            void set_Axbar(const std::vector<double> &Axbar) { ws_Axbar_ = Axbar; }

        private:
            double compute_intersection_parameter(
                const std::vector<double> &v,
                int constraint_idx) const;

            bool compute_projection_and_lambda(
                const std::vector<int> &face,
                std::vector<double> &x_star,
                std::vector<double> &lambda) const;

            bool compute_projection_cached(
                const std::vector<int> &face,
                std::vector<double> &x_star,
                std::vector<double> &lambda) const;

            std::vector<int> find_optimal_face_cached(
                const std::vector<int> &initial_face,
                int &iterations,
                Timer &t) const;

            int select_single_constraint(
                const std::vector<int> &violated_constraints,
                const std::vector<double> &x_proj,
                const std::vector<int> &initial_face,
                bool force_geom_adj) const;

            // Shared initialisation called by both constructors.
            void init_common();

            const double *Q_;
            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
            int m_;
            int n_;
            Config config_;

            std::vector<double> Q_chol_;
            mutable std::vector<double> Q_inv_;
            bool has_diagonal_constraints_ = false;
            bool q_is_dense_ = false;

            mutable ConstraintCache constraint_cache_;
            mutable SchurComplementManager schur_manager_;

            mutable std::vector<double> ws_residual_;
            mutable std::vector<double> ws_w_;
            mutable std::vector<double> ws_correction_;
            mutable std::vector<double> ws_x_star_;
            mutable std::vector<double> ws_lambda_;
            mutable std::vector<double> ws_Axbar_;
            mutable std::vector<double> ws_is_active_;

            mutable std::vector<int> ws_constrained_vars_;
            mutable std::vector<double> ws_signs_;
            mutable std::vector<double> ws_Q_inv_cols_;
            mutable std::vector<double> ws_S_;
            mutable std::vector<double> ws_t_values_;  // cached t values for find_initial_face

            mutable std::vector<double> ws_opt_x_star_;
            mutable std::vector<double> ws_opt_lambda_;
            mutable std::vector<int> ws_opt_violated_;
            mutable std::vector<int> ws_opt_smaller_face_;
            mutable std::vector<char> ws_opt_is_active_set_; 

            bool q_is_diagonal_ = false;
            std::vector<double> Q_diag_;

            // Per-constraint metadata for diagonal/box constraints (has_diagonal_constraints_==true).
            // constraint_var_[i]  = column index j such that A[i,:] = sign * e_j
            // constraint_sign_[i] = ±1 (the sign of that single nonzero entry)
            // Populated in constructor; allows O(1) feasibility check per constraint.
            std::vector<int>    constraint_var_;
            std::vector<double> constraint_sign_;

            struct TimingStats
            {
                double compute_intersection_total = 0.0;
                int compute_intersection_calls = 0;

                double find_initial_face_total = 0.0;
                int find_initial_face_calls = 0;

                double compute_projection_total = 0.0;
                int compute_projection_calls = 0;

                double find_optimal_face_total = 0.0;
                int find_optimal_face_calls = 0;
            } mutable timing_stats_;
        };

        class ProjectionSolver
        {
        public:
            static bool solve(
                const std::vector<int> &active_set,
                const double *Q,
                const double *A,
                const double *b,
                const std::vector<double> &x_bar,
                const std::vector<double> &Q_chol,
                int m,
                int n,
                std::vector<double> &x_star,
                std::vector<double> &lambda,
                const std::vector<double> &Q_inv = std::vector<double>(),
                bool q_is_diagonal = false,
                const std::vector<double> &Q_diag = std::vector<double>());
        };

        class Projector
        {
        public:
            Projector(
                const double *Q, const double *A, const double *b,
                const std::vector<double> &x_bar,
                const std::vector<double> &Q_chol,
                int m, int n,
                const Config &config,
                const std::vector<double> *Q_inv = nullptr,
                bool q_is_diagonal = false,
                const std::vector<double> *Q_diag = nullptr,
                bool has_diagonal_constraints = false,
                const std::vector<int> *constraint_var = nullptr,
                const std::vector<double> *constraint_sign = nullptr);

            void update(const double *b, const std::vector<double> &x_bar)
            {
                b_ = b;
                x_bar_ = x_bar;
            }

            std::vector<double> project_onto_face(
                const std::vector<int> &active_set,
                std::vector<double> &lambda_out);

        private:
            const double *Q_;
            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
            std::vector<double> Q_chol_;
            const std::vector<double> *Q_inv_;
            int m_;
            int n_;
            Config config_;
            // Diagonal fast-path metadata (owned by FaceHopper/Solver, non-owning pointers)
            bool q_is_diagonal_                       = false;
            const std::vector<double> *Q_diag_        = nullptr;
            bool has_diagonal_constraints_            = false;
            const std::vector<int>    *constraint_var_  = nullptr;
            const std::vector<double> *constraint_sign_ = nullptr;
        };

        class KKTChecker
        {
        public:
            KKTChecker(
                const double *Q, const double *g,
                const double *A, const double *b,
                int m, int n,
                double tol);

            bool check(
                const std::vector<double> &x,
                const std::vector<double> &lambda,
                const std::vector<int> &active_set) const;

            double compute_residual(
                const std::vector<double> &x,
                const std::vector<double> &lambda,
                const std::vector<int> &active_set) const;

        private:
            const double *Q_;
            const double *g_;
            const double *A_;
            const double *b_;
            int m_;
            int n_;
            double tol_;
        };

        std::vector<int> hopping_strategy_batch(
            const std::vector<int> &initial_face,
            int &iterations,
            Timer &t,
            const double *Q,
            const double *A,
            const double *b,
            const std::vector<double> &x_bar,
            int m,
            int n,
            const Config &config,
            ConstraintCache &constraint_cache,
            SchurComplementManager &schur_manager,
            const double *Q_chol);

        std::vector<int> hopping_strategy_sensitivity(
            const std::vector<int> &initial_face,
            int &iterations,
            Timer &t,
            const double *Q,
            const double *A,
            const double *b,
            const std::vector<double> &x_bar,
            int m,
            int n,
            const Config &config,
            ConstraintCache &constraint_cache,
            SchurComplementManager &schur_manager,
            const double *Q_chol);

        std::vector<int> hopping_strategy_neighbors(
            const std::vector<int> &initial_face,
            int &iterations,
            Timer &t,
            const double *Q,
            const double *A,
            const double *b,
            const std::vector<double> &x_bar,
            int m,
            int n,
            const Config &config,
            ConstraintCache &constraint_cache,
            SchurComplementManager &schur_manager,
            const double *Q_chol);

    } // namespace internal
} // namespace geoqp

#endif // GEOQP_INTERNAL_HPP
