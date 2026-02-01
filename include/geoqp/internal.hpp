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

        class Phase1Solver
        {
        public:
            Phase1Solver(
                const double *Q, const double *A, const double *b,
                const std::vector<double> &x_bar,
                int m, int n,
                const Config &config);

            std::vector<double> find_good_vertex(
                const std::vector<double> &initial_vertex,
                int &iterations);

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
            
            // Compute Q-weighted squared distance: (v - x_bar)^T Q (v - x_bar)
            double compute_q_weighted_distance_squared(
                const std::vector<double> &v) const;

            const double *Q_;
            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
            int m_;
            int n_;
            Config config_;
        };

        class Phase2Solver
        {
        public:
            Phase2Solver(
                const double *A, const double *b,
                const std::vector<double> &x_bar,
                int m, int n,
                const Config &config);

            std::vector<int> find_active_face(
                const std::vector<double> &v);

        private:
            double compute_intersection_parameter(
                const std::vector<double> &v,
                int constraint_idx) const;

            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
            int m_;
            int n_;
            Config config_;
        };

        class Phase3Solver
        {
        public:
            Phase3Solver(
                const double *A, const double *b,
                const std::vector<double> &x_bar,
                int m, int n,
                const Config &config);

            std::vector<double> project_onto_face(
                const std::vector<int> &active_set,
                std::vector<double> &lambda_out);

        private:
            const double *A_;
            const double *b_;
            std::vector<double> x_bar_;
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
