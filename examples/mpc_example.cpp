/**
 * @file mpc_example.cpp
 * @brief Model Predictive Control (MPC) example using GeoQP
 * 
 * This demonstrates solving a linear MPC problem for a 2D double integrator
 * (position control with velocity states and acceleration inputs).
 * 
 * Uses the "condensed" MPC formulation where:
 * - Decision variables are only the control inputs u_0, ..., u_{N-1}
 * - States are implicitly computed from dynamics
 * - This results in a smaller, dense QP suitable for GeoQP
 * 
 * The system moves a point mass from (-2, -2) to (2, 2) while respecting
 * acceleration limits of +/- 2 m/s^2.
 */

#include <geoqp/geoqp.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <vector>

// Problem dimensions
constexpr int NX = 4;   // States: [x, y, vx, vy]
constexpr int NU = 2;   // Inputs: [ax, ay]
constexpr int N = 10;   // Horizon

// QP size: only control variables
constexpr int N_VARS = NU * N;  // 20 variables

constexpr double DT = 0.1;

/**
 * @brief Double integrator dynamics
 */
void getDynamics(std::vector<double>& A, std::vector<double>& B)
{
    A.resize(NX * NX, 0.0);
    B.resize(NX * NU, 0.0);
    
    // A matrix
    A[0 * NX + 0] = 1.0;  A[0 * NX + 2] = DT;
    A[1 * NX + 1] = 1.0;  A[1 * NX + 3] = DT;
    A[2 * NX + 2] = 1.0;
    A[3 * NX + 3] = 1.0;
    
    // B matrix  
    B[0 * NU + 0] = 0.5 * DT * DT;
    B[1 * NU + 1] = 0.5 * DT * DT;
    B[2 * NU + 0] = DT;
    B[3 * NU + 1] = DT;
}

/**
 * @brief Matrix multiplication C = A * B
 */
void matmul(const std::vector<double>& A, int rA, int cA,
            const std::vector<double>& B, int rB, int cB,
            std::vector<double>& C)
{
    C.resize(rA * cB, 0.0);
    for (int i = 0; i < rA; ++i) {
        for (int j = 0; j < cB; ++j) {
            for (int k = 0; k < cA; ++k) {
                C[i * cB + j] += A[i * cA + k] * B[k * cB + j];
            }
        }
    }
}

/**
 * @brief Compute A^k
 */
std::vector<double> matpow(const std::vector<double>& A, int n, int k)
{
    std::vector<double> result(n * n, 0.0);
    // Initialize to identity
    for (int i = 0; i < n; ++i) {
        result[i * n + i] = 1.0;
    }
    
    std::vector<double> Ak = A;
    for (int p = 0; p < k; ++p) {
        std::vector<double> temp;
        matmul(result, n, n, A, n, n, temp);
        result = temp;
    }
    return result;
}

/**
 * @brief Build condensed MPC matrices
 * 
 * x_k = A^k x_0 + sum_{j=0}^{k-1} A^{k-1-j} B u_j
 *     = A^k x_0 + [A^{k-1}B, A^{k-2}B, ..., AB, B, 0, ..., 0] * [u_0; u_1; ...; u_{N-1}]
 *
 * Stack for all k: X = Sx * x_0 + Su * U
 * where X = [x_1; x_2; ...; x_N], U = [u_0; u_1; ...; u_{N-1}]
 */
void buildCondensedMatrices(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& Sx,  // NX*N x NX
    std::vector<double>& Su)  // NX*N x NU*N
{
    Sx.resize(NX * N * NX, 0.0);
    Su.resize(NX * N * NU * N, 0.0);
    
    // Precompute A^k for k = 1..N
    std::vector<std::vector<double>> Apowers(N + 1);
    Apowers[0].resize(NX * NX, 0.0);
    for (int i = 0; i < NX; ++i) Apowers[0][i * NX + i] = 1.0;
    for (int k = 1; k <= N; ++k) {
        matmul(Apowers[k-1], NX, NX, A, NX, NX, Apowers[k]);
    }
    
    // Build Sx: row k (0-indexed) contains A^{k+1}
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NX; ++j) {
                Sx[(k * NX + i) * NX + j] = Apowers[k + 1][i * NX + j];
            }
        }
    }
    
    // Build Su: element (k, j) is A^{k-j} B for j <= k, else 0
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j <= k; ++j) {
            // A^{k-j} * B
            std::vector<double> AkjB;
            matmul(Apowers[k - j], NX, NX, B, NX, NU, AkjB);
            
            // Place in Su
            for (int i = 0; i < NX; ++i) {
                for (int l = 0; l < NU; ++l) {
                    Su[(k * NX + i) * (NU * N) + (j * NU + l)] = AkjB[i * NU + l];
                }
            }
        }
    }
}

/**
 * @brief Build QP for condensed MPC
 * 
 * Cost: sum_{k=1}^N (x_k - x_ref)^T Q (x_k - x_ref) + sum_{k=0}^{N-1} u_k^T R u_k
 *     = (Sx*x0 + Su*U - X_ref)^T Qbar (Sx*x0 + Su*U - X_ref) + U^T Rbar U
 * 
 * Let e0 = Sx*x0 - X_ref (constant offset)
 * = (Su*U + e0)^T Qbar (Su*U + e0) + U^T Rbar U
 * = U^T (Su^T Qbar Su + Rbar) U + 2 e0^T Qbar Su U + const
 * 
 * So: H = Su^T Qbar Su + Rbar
 *     g = Su^T Qbar e0
 */
void buildCondensedQP(
    const std::vector<double>& Sx,
    const std::vector<double>& Su,
    const std::vector<double>& Q_diag,  // NX diagonal weights
    const std::vector<double>& R_diag,  // NU diagonal weights
    const std::vector<double>& x0,
    const std::vector<double>& x_ref,
    std::vector<double>& H,  // N_VARS x N_VARS
    std::vector<double>& g)  // N_VARS
{
    // Qbar is block-diagonal with Q repeated N times
    // Rbar is block-diagonal with R repeated N times
    
    // Compute e0 = Sx * x0 - X_ref (stacked)
    std::vector<double> e0(NX * N, 0.0);
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NX; ++i) {
            double sum = 0.0;
            for (int j = 0; j < NX; ++j) {
                sum += Sx[(k * NX + i) * NX + j] * x0[j];
            }
            e0[k * NX + i] = sum - x_ref[i];
        }
    }
    
    // H = Su^T * Qbar * Su + Rbar
    // First compute Qbar * Su (just scale each row of Su by Q_diag)
    std::vector<double> QSu(NX * N * NU * N, 0.0);
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NX; ++i) {
            double q = Q_diag[i];
            for (int j = 0; j < NU * N; ++j) {
                QSu[(k * NX + i) * (NU * N) + j] = q * Su[(k * NX + i) * (NU * N) + j];
            }
        }
    }
    
    // H = Su^T * QSu
    H.resize(N_VARS * N_VARS, 0.0);
    for (int i = 0; i < N_VARS; ++i) {
        for (int j = 0; j < N_VARS; ++j) {
            double sum = 0.0;
            for (int k = 0; k < NX * N; ++k) {
                sum += Su[k * N_VARS + i] * QSu[k * N_VARS + j];
            }
            H[i * N_VARS + j] = sum;
        }
    }
    
    // Add Rbar (diagonal)
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NU; ++i) {
            int idx = k * NU + i;
            H[idx * N_VARS + idx] += R_diag[i];
        }
    }
    
    // Add small regularization for positive definiteness
    for (int i = 0; i < N_VARS; ++i) {
        H[i * N_VARS + i] += 1e-6;
    }
    
    // g = Su^T * Qbar * e0 = Su^T * (Q * e0)
    std::vector<double> Qe0(NX * N);
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NX; ++i) {
            Qe0[k * NX + i] = Q_diag[i] * e0[k * NX + i];
        }
    }
    
    g.resize(N_VARS, 0.0);
    for (int i = 0; i < N_VARS; ++i) {
        double sum = 0.0;
        for (int k = 0; k < NX * N; ++k) {
            sum += Su[k * N_VARS + i] * Qe0[k];
        }
        g[i] = sum;
    }
}

/**
 * @brief Build input bound constraints
 */
void buildInputConstraints(
    const std::vector<double>& u_min,
    const std::vector<double>& u_max,
    std::vector<double>& A_ineq,
    std::vector<double>& b_ineq)
{
    // u_i <= u_max  =>  u_i <= u_max
    // u_i >= u_min  =>  -u_i <= -u_min
    
    int n_con = 2 * N_VARS;
    A_ineq.resize(n_con * N_VARS, 0.0);
    b_ineq.resize(n_con);
    
    int row = 0;
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NU; ++i) {
            int var = k * NU + i;
            
            // u <= u_max
            A_ineq[row * N_VARS + var] = 1.0;
            b_ineq[row] = u_max[i];
            row++;
            
            // -u <= -u_min
            A_ineq[row * N_VARS + var] = -1.0;
            b_ineq[row] = -u_min[i];
            row++;
        }
    }
}

/**
 * @brief Simulate one step
 */
std::vector<double> simulate(
    const std::vector<double>& A,
    const std::vector<double>& B,
    const std::vector<double>& x,
    const std::vector<double>& u)
{
    std::vector<double> x_next(NX, 0.0);
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NX; ++j) {
            x_next[i] += A[i * NX + j] * x[j];
        }
        for (int j = 0; j < NU; ++j) {
            x_next[i] += B[i * NU + j] * u[j];
        }
    }
    return x_next;
}

double computeError(const std::vector<double>& x, const std::vector<double>& x_ref)
{
    double err = 0.0;
    for (int i = 0; i < NX; ++i) {
        err += (x[i] - x_ref[i]) * (x[i] - x_ref[i]);
    }
    return std::sqrt(err);
}

int main()
{
    std::cout << "=== GeoQP Condensed MPC Example ===\n\n";
    std::cout << "System: 2D Double Integrator\n";
    std::cout << "States: " << NX << ", Inputs: " << NU << ", Horizon: " << N << "\n";
    std::cout << "QP variables: " << N_VARS << " (control inputs only)\n\n";
    
    // Dynamics
    std::vector<double> A_dyn, B_dyn;
    getDynamics(A_dyn, B_dyn);
    
    // Build condensed matrices
    std::vector<double> Sx, Su;
    buildCondensedMatrices(A_dyn, B_dyn, Sx, Su);
    
    // Weights
    std::vector<double> Q_diag = {10.0, 10.0, 1.0, 1.0};
    std::vector<double> R_diag = {0.1, 0.1};
    
    // Input bounds
    std::vector<double> u_min = {-2.0, -2.0};
    std::vector<double> u_max = {2.0, 2.0};
    
    // Initial and reference
    std::vector<double> x_current = {-2.0, -2.0, 0.0, 0.0};
    std::vector<double> x_ref = {2.0, 2.0, 0.0, 0.0};
    
    std::cout << "Initial: [" << x_current[0] << ", " << x_current[1] << "]\n";
    std::cout << "Target:  [" << x_ref[0] << ", " << x_ref[1] << "]\n\n";
    
    // Build QP
    geoqp::Problem problem;
    problem.n = N_VARS;
    buildCondensedQP(Sx, Su, Q_diag, R_diag, x_current, x_ref, problem.Q, problem.g);
    buildInputConstraints(u_min, u_max, problem.A, problem.b);
    problem.m = static_cast<int>(problem.b.size());
    
    std::cout << "QP: " << problem.n << " vars, " << problem.m << " constraints\n\n";
    
    // Configure solver
    geoqp::Config config;
    config.verbosity = 0;
    config.tolerance = 1e-8;
    
    geoqp::Solver solver(config);
    
    // Simulation
    const int n_steps = 50;
    double total_time = 0.0;
    
    std::cout << "Step |   x       y      vx      vy   | Error  | Time (ms)\n";
    std::cout << "-----+-------------------------------+--------+----------\n";
    
    for (int step = 0; step < n_steps; ++step) {
        // Rebuild QP with current state
        buildCondensedQP(Sx, Su, Q_diag, R_diag, x_current, x_ref, problem.Q, problem.g);
        
        // Solve
        auto t_start = std::chrono::high_resolution_clock::now();
        geoqp::Solution sol = solver.solve(problem);
        auto t_end = std::chrono::high_resolution_clock::now();
        
        double solve_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        total_time += solve_ms;
        
        if (sol.status != geoqp::Status::SUCCESS) {
            std::cerr << "Solver failed at step " << step << "\n";
            return 1;
        }
        
        // Extract first control input
        std::vector<double> u = {sol.x[0], sol.x[1]};
        
        // Print
        double error = computeError(x_current, x_ref);
        std::cout << std::setw(4) << step << " | "
                  << std::fixed << std::setprecision(3)
                  << std::setw(6) << x_current[0] << " "
                  << std::setw(6) << x_current[1] << " "
                  << std::setw(6) << x_current[2] << " "
                  << std::setw(6) << x_current[3] << " | "
                  << std::setw(6) << error << " | "
                  << std::setw(8) << std::setprecision(3) << solve_ms << "\n";
        
        // Check convergence
        if (error < 0.01) {
            std::cout << "\nConverged at step " << step << "!\n";
            break;
        }
        
        // Simulate
        x_current = simulate(A_dyn, B_dyn, x_current, u);
    }
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Final position: [" << std::fixed << std::setprecision(4) 
              << x_current[0] << ", " << x_current[1] << "]\n";
    std::cout << "Final error: " << computeError(x_current, x_ref) << "\n";
    std::cout << "Total solve time: " << std::setprecision(2) << total_time << " ms\n";
    std::cout << "Average solve time: " << (total_time / n_steps) << " ms\n";
    
    return 0;
}
