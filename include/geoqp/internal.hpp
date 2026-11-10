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
            // Same elimination, overwriting A and b; false if singular.
            static bool solve_linear_system_inplace(double *A, double *b, double *x, int n);
            static void cholesky(const double *A, double *L, int n);

            static void forward_sub(
                const double *L, const double *b, double *x,
                int n);
            // L x = e_v, bit-identical to forward_sub on the unit vector.
            static void forward_sub_unit(const double *L, int v, double *x, int n);
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

        // Antipodal edge frame of an axis-aligned box  ℓ ≤ x ≤ u  relative to x̄.
        //
        //   sᵢ = +1 if x̄ᵢ ≥ μᵢ else −1,   hᵢ = ½(uᵢ − ℓᵢ),   μ = ½(ℓ + u)
        //   v̂ = μ − s∘h                    (antipodal good vertex)
        //   ζ = s∘(x − v̂)/h                (box ↦ [0,2]ⁿ, v̂ ↦ 0, ζ̄ ≥ 1)
        //   V = { i : x̄ᵢ ∉ [ℓᵢ, uᵢ] } = { i : ζ̄ᵢ > 2 }   (visible facets {ζᵢ = 2})
        //   i₀ = argmax ζ̄ᵢ                  (exit facet F₀ of the ray v̂ → x̄)
        //
        // The visible-facet lemma puts x★ on some {ζᵢ = 2}, i ∈ V, and that facet
        // is F₀ or shares a ridge with it.
        struct AntipodalFrame
        {
            bool valid = false;
            bool pure_box = false;      // every constraint row is a box row
            std::vector<double> s;
            std::vector<double> h;      // half-widths
            std::vector<double> zeta_bar;
            std::vector<double> v_hat;
            std::vector<int> plus_row;  // per coordinate: row index of the {ζᵢ = 2} wall
            std::vector<int> violated;  // V, ascending
            int i0 = -1;                // argmax ζ̄ (smallest index on ties)
            int exit_row = -1;          // row of F₀, or −1 if V is empty
        };

        // ─────────────────────────────────────────────────────────────────────
        // Pure-box kernel:  min ½xᵀQx + gᵀx  s.t.  ℓ ≤ x ≤ u,  x̄ = −Q⁻¹g ∉ box.
        //
        // Same algorithm and pivot rule as Initializer (antipodal frame) +
        // FaceHopper (plain Phase II, no caching) on a box, but working on the
        // bound arrays directly: no constraint matrix is built, scanned or
        // multiplied, no helper objects are constructed, and nothing is
        // allocated per iteration.  Rows of Q⁻¹ are computed on demand.
        //
        // Constraints are addressed by position p ∈ [0, 2n) in *row order* (the
        // order the equivalent explicit rows have), which fixes tie-breaking.
        // ─────────────────────────────────────────────────────────────────────
        class BoxKernel
        {
        public:
            // Fills the per-position tables.  Returns false unless every variable
            // has a finite lower and upper bound with ℓ < u and every explicit
            // row of A is ±eⱼ (then that row's bound is merged, tightest wins).
            //   rows 0..m−1 : explicit A x ≤ b;  then finite ub;  then finite lb
            //   (the solver's bound-expansion order).
            bool setup(int n, int m, const double *A, const double *b,
                       const std::vector<double> &lb, const std::vector<double> &ub);

            // Solve.  L is the Cholesky factor of Q (unused when Q is diagonal).
            // Returns false only on a numerical failure of the final projection.
            bool solve(const double *Q, const double *L, const double *x_bar, const Config &config);

            int n() const { return n_; }
            // x̄ within the bounds up to tol?
            bool contains(const double *x, double tol) const;

            // Results of the last solve().
            const double *x() const { return x_; }
            const double *v_hat() const { return v_hat_; }
            int num_active() const { return W_size_; }
            const int *active_rows() const { return active_rows_; } // ascending
            const double *lambda() const { return lam_; }           // same order
            int iterations() const { return iterations_; }
            int facet_hops() const { return hops_; }
            // Certificate (see Solution): active bounds with λ > tol that x̄ violates,
            // and Σ_{i∈V} ζ★ᵢ − 2 in the antipodal frame.
            int violated_active_count(double tol) const;
            double visible_cut_margin() const;

        private:
            bool project(const double *L, const double *x_bar, const Config &config);
            const double *q_inv_row(const double *L, int v);
            void allocate(int n);

            int n_ = 0;

            // All fixed-size arrays live in two arenas (one allocation each), so a
            // freshly constructed solver does not pay for two dozen small mallocs.
            std::vector<double> darena_;
            std::vector<int> iarena_;
            double *lb_ = nullptr, *ub_ = nullptr;       // n each
            // Per position p (row order): variable, sign (+1 upper / −1 lower),
            // right-hand side (u or −ℓ), original row index.
            int *pvar_ = nullptr, *prow_ = nullptr;      // 2n each
            double *psign_ = nullptr, *pb_ = nullptr;    // 2n each
            int *pos_upper_ = nullptr, *pos_lower_ = nullptr; // position of each variable's walls

            bool q_diag_ = false;
            double *q_diagonal_ = nullptr;               // n
            const double *xbar_ = nullptr;

            int *W_ = nullptr;                           // working set, positions, ascending (≤ 2n)
            int W_size_ = 0;
            int *in_W_ = nullptr;                        // 2n flags
            double *x_ = nullptr, *lam_ = nullptr;       // n, 2n
            double *v_hat_ = nullptr, *neg_grad_ = nullptr, *s_frame_ = nullptr, *h_frame_ = nullptr; // n each
            int *violated_frame_ = nullptr;              // ≤ n
            int violated_frame_size_ = 0;
            int *viol_ = nullptr;                        // ≤ 2n
            double *unit_ = nullptr;                     // n
            int *row_slot_ = nullptr;                    // per variable: index into rows_ or −1
            int *active_rows_ = nullptr;                 // ≤ 2n

            // Dense-Q only, sized by the working set as it grows.
            std::vector<double> S_, rhs_;
            std::vector<double> rows_;                   // lazily computed rows of Q⁻¹, n each
            int rows_used_ = 0;

            int iterations_ = 0, hops_ = 0;
            bool solve_failed_ = false; // last project(): the k×k system could not be solved
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
            // Q-weighted distances and the ν-biased BFS
            void set_Q_chol(const std::vector<double> &Q_chol);

            const std::vector<double> &get_Axbar() const { return ws_Axbar_; }

            std::vector<double> find_good_vertex(
                const std::vector<double> &initial_vertex,
                int &iterations,
                int &vertices_explored);

            // Valid when every variable has a ±eⱼ lower and upper bound row.
            const AntipodalFrame &antipodal_frame() const { return frame_; }

        private:
            void compute_axbar();
            void detect_box_constraints();   // structural, from A only
            void refresh_box_bounds();       // values, from b (cheap, O(#box rows))
            void compute_antipodal_frame();  // O(n), needs bounds and x̄

            bool is_good_vertex(
                const std::vector<double> &v,
                const std::vector<int> &active_set) const;

            double compute_q_weighted_distance_squared_ptr(const double *vptr) const;

            std::vector<double> compute_vertex_from_active_set(
                const std::vector<int> &active_indices) const;

            std::vector<double> find_good_vertex_box_fast() const;

            // Compute the separation direction ν ∝ Q(x̄ - x̂) once per
            // find_good_vertex call and store in ws_nu_.  Returns the
            // distance L = ‖x̄ - x̂‖_Q used as the linear-proxy scaling.
            double compute_separation_direction(const double *closest_boundary_pt) const;

            // BFS vertex score 
            inline double bfs_score(const double *vptr, double two_L) const;

            const double *Q_;
            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
            int m_;
            int n_;
            Config config_;

            // Cholesky factor L of Q (lower-triangular), set externally.
            std::vector<double> Q_chol_;
            bool has_Q_chol_ = false;

            bool is_box_constrained_ = false;
            std::vector<double> box_lb_;
            std::vector<double> box_ub_;
            std::vector<bool> has_lb_;
            std::vector<bool> has_ub_;
            std::vector<int> lb_row_;   // row realising box_lb_[j]
            std::vector<int> ub_row_;   // row realising box_ub_[j]
            // Box rows found by detect_box_constraints: (row, var, +1 upper / −1 lower)
            struct BoxRow { int row; int var; int sign; };
            std::vector<BoxRow> box_rows_;
            AntipodalFrame frame_;

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

            // Box-constraint specialization: uses Q_inv for O(1) S-entry access
            bool initialize_box(
                const std::vector<int> &face,
                const std::vector<int> &constraint_var,
                const std::vector<double> &constraint_sign,
                const double *Q_inv,
                int n);

            bool add_constraint_box(
                int constraint_idx,
                const std::vector<int> &constraint_var,
                const std::vector<double> &constraint_sign,
                const double *Q_inv);

            bool solve(const std::vector<double> &rhs, std::vector<double> &lambda) const;

            double get_L_diagonal(int i) const
            {
                if (i >= k_)
                    return 0.0;
                return L_S_[i * k_ + i];
            }

            const std::vector<int> &get_active_constraints() const { return active_constraints_; }
            int size() const { return k_; }

            const double *get_S_matrix()  const { return S_.data(); }
            int           get_S_stride()  const { return s_stride_; }

        private:
            std::vector<double> L_S_;
            // S_: n_ × n_ preallocated Schur complement.  Only the top-left k_ × k_
            // submatrix (with row stride n_) is valid.  
            std::vector<double> S_;
            std::vector<int> active_constraints_;
            int k_;
            const double *A_ptr_ = nullptr;
            const double *L_Q_ptr_ = nullptr;
            int n_ = 0;         // Q_inv row stride (= problem half-dimension for box paths)
            int s_stride_ = 0;  // S_ row stride (= 2*n_ to accommodate up to 2n constraints)
            ConstraintCache *cache_ptr_ = nullptr;
            bool is_box_mode_ = false;
            const double *Q_inv_ptr_ = nullptr;
            const std::vector<int> *constraint_var_ptr_ = nullptr;
            const std::vector<double> *constraint_sign_ptr_ = nullptr;
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

            FaceHopper(
                const double *Q, const double *A, const double *b,
                const std::vector<double> &x_bar,
                int m, int n,
                const Config &config,
                const std::vector<double> &Q_chol);

            FaceHopper(
                const double *Q, const double *A, const double *b,
                const std::vector<double> &x_bar,
                int m, int n,
                const Config &config,
                const std::vector<double> &Q_chol,
                bool precompute_q_inv);

            // New b and x̄ for the same Q and A (hot start).  Recomputes A·x̄, which
            // every Phase II residual uses, and drops the previous frame.
            void update(const double *b, const std::vector<double> &x_bar)
            {
                b_ = b;
                x_bar_ = x_bar;
                ws_Axbar_.resize(m_);
                LinearAlgebra::matvec(A_, x_bar_.data(), ws_Axbar_.data(), m_, n_);
                frame_ = nullptr;
            }

            // Antipodal frame from Phase I (non-owning; must outlive the next
            // find_initial_face call).  When it is valid and v is its v̂, the
            // initial face is {F₀} read off the frame instead of ray-traced.
            void set_antipodal_frame(const AntipodalFrame *frame) { frame_ = frame; }

            std::vector<int> find_initial_face(
                const std::vector<double> &v);

            std::vector<int> find_optimal_face(
                const std::vector<int> &initial_face,
                int &iterations,
                int &facet_hops);

            const std::vector<double> &get_last_x() const { return ws_x_star_; }
            const std::vector<double> &get_last_lambda() const { return ws_lambda_; }

            const std::vector<double> &get_Q_chol() const { return Q_chol_; }
            // The full Q⁻¹ (row-major n×n) when it was precomputed, else empty.
            // In lazy mode only individual rows exist; use q_inv_row().
            const std::vector<double> &get_Q_inv() const { return q_inv_full_ ? Q_inv_ : empty_vec_; }

            // Row v of Q⁻¹ (= column v).  Precomputed mode: a pointer into the full
            // matrix.  Lazy mode: computed by one Cholesky solve on first use and
            // kept; the pointer stays valid for the life of this object.  Returns
            // nullptr when no Cholesky factor is available.
            const double *q_inv_row(int v) const;

            // x★ and λ of the face returned by the last find_optimal_face call
            // (λ in the order of that face).  Valid only if has_last_solution().
            bool has_last_solution() const { return last_valid_; }
            const std::vector<double> &last_x() const { return last_x_; }
            const std::vector<double> &last_lambda() const { return last_lambda_; }
            const std::vector<double> &get_Q_diag() const { return Q_diag_; }
            bool q_is_diagonal() const { return q_is_diagonal_; }
            bool has_diagonal_constraints() const { return has_diagonal_constraints_; }
            const std::vector<int>    &get_constraint_var()  const { return constraint_var_; }
            const std::vector<double> &get_constraint_sign() const { return constraint_sign_; }

            void set_Q_chol(const std::vector<double> &Q_chol) { Q_chol_ = Q_chol; }

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
                int &facet_hops,
                Timer &t) const;

            // Standalone box+dense-Q incremental Schur path..
            std::vector<int> find_optimal_face_box_schur(
                const std::vector<int> &initial_face,
                int &iterations,
                int &facet_hops,
                Timer &t) const;

            int select_single_constraint(
                const std::vector<int> &violated_constraints,
                const std::vector<double> &x_proj,
                const std::vector<int> &initial_face,
                bool force_geom_adj) const;

            // Shared initialisation called by both constructors.
            void init_common(bool precompute_q_inv = true);

            const double *Q_;
            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
            int m_;
            int n_;
            Config config_;

            std::vector<double> Q_chol_;
            // Q⁻¹: either the full matrix (q_inv_full_, hot-start setup where the
            // O(n³) cost is paid once) or rows computed on demand (one-shot solves,
            // where only the rows of constraints that become active are needed).
            mutable std::vector<double> Q_inv_;
            bool q_inv_full_ = false;
            mutable std::vector<std::vector<double>> q_inv_lazy_rows_; // per variable, empty = not yet
            mutable std::vector<double> ws_unit_;
            std::vector<double> empty_vec_;
            // Compact N×N inverse of the shared x/y block (LIPM structure).  In lazy
            // mode its rows are filled on demand by ensure_hs_row().
            mutable std::vector<double> H_s_inv_;
            mutable std::vector<char> hs_have_;
            void ensure_hs_row(int p) const;

            mutable bool last_valid_ = false;
            mutable std::vector<double> last_x_;
            mutable std::vector<double> last_lambda_;
            std::vector<int> constraint_var_half_;
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
            mutable std::vector<double> ws_lambda_y_;  

            bool q_is_diagonal_ = false;
            std::vector<double> Q_diag_;

            std::vector<int>    constraint_var_;
            std::vector<double> constraint_sign_;

            // ── H1 theory-guided hint (Proposition 2) + ray direction (Proposition 3)
            // Populated once in find_initial_face when has_diagonal_constraints_.
            // Fixed for the entire Phase II loop; never changes between hops.
            //
            //   v_hat_             : good vertex v̂ (antipodal, from Phase I)
            //   ws_neg_grad_vhat_  : −∇f(v̂) = Q(x̄ − v̂)  (H1 inward normal)
            //   ws_d_vhat_         : d = x̄ − v̂            (ray direction for Proposition 3 ray-exit)
            //
            // H1 filter for adding box constraint i (var j, sign s):
            //   ws_neg_grad_vhat_[j] * s ≥ 0
            bool has_vhat_hint_ = false;
            const AntipodalFrame *frame_ = nullptr;
            std::vector<double> v_hat_;
            mutable std::vector<double> ws_neg_grad_vhat_;
            mutable std::vector<double> ws_d_vhat_;

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
