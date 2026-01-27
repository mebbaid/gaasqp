#ifndef GEOQP_C_API_H
#define GEOQP_C_API_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include <stdbool.h>

    typedef struct geoqp_solver_t geoqp_solver_t; // Opaque handle to solver

    typedef enum
    {
        GEOQP_SUCCESS = 0,
        GEOQP_UNBOUNDED = 1,
        GEOQP_INFEASIBLE = 2,
        GEOQP_MAX_ITER = 3,
        GEOQP_NUMERICAL_ERROR = 4,
        GEOQP_INVALID_INPUT = 5
    } geoqp_status_t;

    typedef struct
    {
        double tolerance;
        int max_simplex_iterations;
        int max_iterations;
        bool warm_start;
        bool check_optimality;
        int verbosity;
    } geoqp_config_t;

    typedef struct
    {
        int n;
        int m;
        const double *Q;
        const double *g;
        const double *A;
        const double *b;
        const double *lb;
        const double *ub;
    } geoqp_problem_t;

    typedef struct
    {
        double *x;           
        double *lambda;      
        int *active_set;     
        int active_set_size; 
        double objective_value;
        geoqp_status_t status;
        int iterations;
        int phase1_iterations;
        double solve_time;
    } geoqp_solution_t;

    geoqp_solver_t *geoqp_solver_create();
    geoqp_solver_t *geoqp_solver_create_with_config(const geoqp_config_t *config);
    void geoqp_solver_destroy(geoqp_solver_t *solver);

    geoqp_status_t geoqp_solve(
        geoqp_solver_t *solver,
        const geoqp_problem_t *problem,
        geoqp_solution_t *solution);

    geoqp_solution_t *geoqp_solution_create(int n, int m);
    void geoqp_solution_destroy(geoqp_solution_t *solution);

    geoqp_config_t geoqp_get_default_config();

    bool geoqp_validate_problem(const geoqp_problem_t *problem);

    double geoqp_compute_objective(
        const double *x,
        const double *Q,
        const double *g,
        int n);

    bool geoqp_is_feasible(
        const double *x,
        const double *A,
        const double *b,
        int m, int n,
        double tol);

#ifdef __cplusplus
}
#endif

#endif // GEOQP_C_API_H
