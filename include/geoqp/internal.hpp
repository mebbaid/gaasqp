#ifndef GEOQP_INTERNAL_HPP
#define GEOQP_INTERNAL_HPP

#include "geoqp/geoqp.hpp"
#include <vector>
#include <set>
#include <chrono>
#include <iostream>

namespace geoqp
{
    namespace internal
    {
        // Simple timer for profiling
        struct Timer {
            using Clock = std::chrono::high_resolution_clock;
            using TimePoint = Clock::time_point;
            
            TimePoint start;
            
            Timer() : start(Clock::now()) {}
            
            double elapsed_ms() const {
                auto end = Clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                return elapsed.count();
            }
            
            void reset() {
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
            
            // Forward substitution: solve L * x = b where L is lower triangular
            static void forward_sub(
                const double *L, const double *b, double *x,
                int n);
            
            // Backward substitution: solve L^T * x = b where L is lower triangular
            static void backward_sub(
                const double *L, const double *b, double *x,
                int n);
            
            static void solve_cholesky(
                const double *L, const double *b, double *x,
                int n);
            static void solve_pd_system(
                const double *Q, const double *b, double *x,
                int n);
            
            // Rank-1 Cholesky update
            static bool cholesky_rank1_update(double *L, const double *v, int k);
            
            // Cholesky extend
            static bool cholesky_extend(
                const double *L, int k,
                const double *s, double sigma,
                double *L_new);
            
            // Cholesky shrink
            static bool cholesky_shrink(
                const double *L, int k, int idx,
                double *L_new);
        };

        class Polytope
        {
        public:
            Polytope(const double *A, const double *b, int m, int n);

            std::vector<double> find_initial_vertex() const;
            std::vector<std::vector<double>> enumerate_vertices() const;
            bool is_vertex(const std::vector<double> &x, double tol = 1e-8) const;

            std::vector<int> get_active_constraints(
                const std::vector<double> &x,
                double tol = 1e-8) const;

            std::vector<std::vector<double>> get_neighbors(
                const std::vector<double> &vertex) const;

        private:
            const double *A_;
            const double *b_;
            int m_;
            int n_;
        };

        /**
         * @brief Initializer - finds a "good" starting vertex
         * 
         * A good vertex is one where the direction to x_bar is feasible (doesn't
         * immediately violate any constraint). This is Phase I of the algorithm.
         */
        class Initializer
        {
        public:
            Initializer(
                const double *Q, const double *A, const double *b,
                const std::vector<double> &x_bar,
                int m, int n,
                const Config &config);

            std::vector<double> find_good_vertex(
                const std::vector<double> &initial_vertex,
                int &iterations,
                int &vertices_explored);

        private:
            bool is_direction_feasible(
                const std::vector<double> &v,
                const std::vector<double> &d,
                const std::vector<int> &active_set,
                double tol = 1e-8) const;

            int find_most_violated_constraint(
                const std::vector<double> &v,
                const std::vector<double> &d,
                const std::vector<int> &active_set) const;

            std::vector<double> pivot_to_neighbor(
                const std::vector<double> &v,
                int constraint_to_drop);
            
            bool is_good_vertex(
                const std::vector<double> &v,
                const std::vector<int> &active_set) const;
            
            double compute_q_weighted_distance_squared(
                const std::vector<double> &v) const;
            
            double compute_euclidean_distance_squared(
                const std::vector<double> &v) const;
            
            std::vector<double> compute_vertex_from_active_set(
                const std::vector<int> &active_indices) const;
            
            void detect_box_constraints();
            
            std::vector<double> find_good_vertex_box_fast() const;

            const double *Q_;
            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
            int m_;
            int n_;
            Config config_;
            
            bool is_box_constrained_ = false;
            std::vector<double> box_lb_;
            std::vector<double> box_ub_;
        };

        /**
         * @brief FaceHopper - explores faces to find the optimal one
         * 
         * Starting from an initial face (found via ray tracing from a good vertex),
         * hops between neighboring faces to find the one whose projection of x_bar
         * is closest. This is Phase II of the algorithm.
         */
        class FaceHopper
        {
        public:
            FaceHopper(
                const double *Q, const double *A, const double *b,
                const std::vector<double> &x_bar,
                int m, int n,
                const Config &config);

            std::vector<int> find_initial_face(
                const std::vector<double> &v);
            
            std::vector<int> find_optimal_face(
                const std::vector<int> &initial_face,
                int &iterations);
            
            const std::vector<double>& get_Q_chol() const { return Q_chol_; }

        private:
            double compute_intersection_parameter(
                const std::vector<double> &v,
                int constraint_idx) const;
            
            bool compute_projection_and_lambda(
                const std::vector<int> &face,
                std::vector<double> &x_star,
                std::vector<double> &lambda) const;

            const double *Q_;
            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
            int m_;
            int n_;
            Config config_;
            
            std::vector<double> Q_chol_;
            bool has_diagonal_constraints_ = false;
            
            // Timing stats (aggregate)
            struct TimingStats {
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

        /**
         * @brief Projector - projects x_bar onto the optimal face
         * 
         * Given the optimal face (active constraint set), computes the Q-weighted
         * projection of x_bar onto that face. This is Phase III of the algorithm.
         */
        class Projector
        {
        public:
            Projector(
                const double *Q, const double *A, const double *b,
                const std::vector<double> &x_bar,
                const std::vector<double> &Q_chol,
                int m, int n,
                const Config &config);

            std::vector<double> project_onto_face(
                const std::vector<int> &active_set,
                std::vector<double> &lambda_out);

        private:
            const double *Q_;
            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
            std::vector<double> Q_chol_;
            int m_;
            int n_;
            Config config_;
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

    } // namespace internal
} // namespace geoqp

#endif // GEOQP_INTERNAL_HPP
