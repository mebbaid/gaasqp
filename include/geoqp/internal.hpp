#ifndef GEOQP_INTERNAL_HPP
#define GEOQP_INTERNAL_HPP

#include "geoqp/geoqp.hpp"
#include <vector>
#include <set>

namespace geoqp
{
    namespace internal
    {

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
            static void solve_cholesky(
                const double *L, const double *b, double *x,
                int n);
            static void solve_pd_system(
                const double *Q, const double *b, double *x,
                int n);
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
            
            // Check if vertex is "good" (direction to x_bar is feasible)
            bool is_good_vertex(
                const std::vector<double> &v,
                const std::vector<int> &active_set) const;
            
            // Compute Q-weighted squared distance: (v - x_bar)^T Q (v - x_bar)
            double compute_q_weighted_distance_squared(
                const std::vector<double> &v) const;
            
            // Compute Euclidean squared distance to x_bar
            double compute_euclidean_distance_squared(
                const std::vector<double> &v) const;
            
            // Compute vertex from active constraint set
            std::vector<double> compute_vertex_from_active_set(
                const std::vector<int> &active_indices) const;
            
            // Detect if constraints form a box (lb <= x <= ub)
            void detect_box_constraints();
            
            // Fast O(n) algorithm for box-constrained problems
            std::vector<double> find_good_vertex_box_fast() const;

            const double *Q_;
            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
            int m_;
            int n_;
            Config config_;
            
            // Box constraint detection
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

            // Find initial blocking face from good vertex via ray tracing
            std::vector<int> find_initial_face(
                const std::vector<double> &v);
            
            // Hop between faces to find optimal face (closest projection to x_bar)
            std::vector<int> find_optimal_face(
                const std::vector<int> &initial_face,
                int &iterations);
            
            // Get cached Cholesky factorization of Q
            const std::vector<double>& get_Q_chol() const { return Q_chol_; }

        private:
            double compute_intersection_parameter(
                const std::vector<double> &v,
                int constraint_idx) const;
            
            // Compute squared distance from x_bar to its projection on a face
            double compute_projection_distance_sq(
                const std::vector<int> &face) const;
            
            // Compute projection and lambdas for a face
            bool compute_projection_and_lambda(
                const std::vector<int> &face,
                std::vector<double> &x_star,
                std::vector<double> &lambda) const;
            
            // Get neighboring faces by adding/dropping constraints
            std::vector<std::vector<int>> get_neighbor_faces(
                const std::vector<int> &face) const;

            const double *Q_;
            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
            int m_;
            int n_;
            Config config_;
            
            // Cached Cholesky factorization of Q for projections
            std::vector<double> Q_chol_;
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
