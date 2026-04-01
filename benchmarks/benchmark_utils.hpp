#pragma once

#include <geoqp/geoqp.hpp>
#include <Eigen/Dense>

#include <vector>
#include <string>
#include <random>
#include <cmath>

// ─── Shared benchmark types ──────────────────────────────────────────────────

struct QPProblem
{
    int n = 0;
    int m = 0;
    Eigen::MatrixXd H;
    Eigen::VectorXd g;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::VectorXd lbX;
    Eigen::VectorXd ubX;
    std::string name;
};

// ─── Random PD matrix generators ─────────────────────────────────────────────

static Eigen::MatrixXd randomPD(int n, std::mt19937 &rng, double condition = 10.0)
{
    std::normal_distribution<> dist(0.0, 1.0);
    Eigen::MatrixXd R(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            R(i, j) = dist(rng);
    Eigen::MatrixXd H = R.transpose() * R;
    double trace = std::max(H.trace(), 1e-12);
    H += Eigen::MatrixXd::Identity(n, n) * (trace / (n * condition));
    return H;
}

static Eigen::MatrixXd randomPD_conditioned(int n, std::mt19937 &rng, double cond)
{
    std::normal_distribution<> dist(0.0, 1.0);
    Eigen::MatrixXd R(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            R(i, j) = dist(rng);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(R);
    Eigen::MatrixXd Orth = qr.householderQ();
    Eigen::VectorXd eigs(n);
    for (int i = 0; i < n; ++i)
        eigs(i) = std::pow(cond, static_cast<double>(i) / std::max(n - 1, 1));
    return Orth * eigs.asDiagonal() * Orth.transpose();
}

// ─── Eigen conversion helpers ─────────────────────────────────────────────────

static std::vector<double> eigen_to_flat(const Eigen::MatrixXd &M)
{
    int r = M.rows(), c = M.cols();
    std::vector<double> v(r * c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            v[i * c + j] = M(i, j);
    return v;
}

static std::vector<double> eigen_to_vec(const Eigen::VectorXd &v)
{
    return std::vector<double>(v.data(), v.data() + v.size());
}

// Compute x̄ = −Q⁻¹g
static Eigen::VectorXd compute_xbar(const Eigen::MatrixXd &Q, const Eigen::VectorXd &g)
{
    return -Q.ldlt().solve(g);
}

// ─── GeoQP conversion helpers ─────────────────────────────────────────────────

// Native lb/ub API (triggers box fast-path in GeoQP)
static geoqp::Problem to_geoqp(const QPProblem &qp)
{
    geoqp::Problem gp;
    gp.n = qp.n;
    gp.m = qp.m;
    gp.Q = eigen_to_flat(qp.H);
    gp.g = eigen_to_vec(qp.g);
    if (qp.m > 0) { gp.A = eigen_to_flat(qp.A); gp.b = eigen_to_vec(qp.b); }
    bool has_lb = false, has_ub = false;
    for (int i = 0; i < qp.n; ++i)
    {
        if (qp.lbX(i) > -1e19) has_lb = true;
        if (qp.ubX(i) <  1e19) has_ub = true;
    }
    if (has_lb) gp.lb = eigen_to_vec(qp.lbX);
    if (has_ub) gp.ub = eigen_to_vec(qp.ubX);
    return gp;
}

// Explicit A/b form (bounds expanded as rows, consistent constraint indexing)
static geoqp::Problem to_geoqp_explicit(const QPProblem &qp)
{
    geoqp::Problem gp;
    gp.n = qp.n;
    int n_lb = 0, n_ub = 0;
    for (int i = 0; i < qp.n; ++i)
    {
        if (qp.lbX(i) > -1e19) n_lb++;
        if (qp.ubX(i) <  1e19) n_ub++;
    }
    int m_total = qp.m + n_lb + n_ub;
    gp.m = m_total;
    gp.Q = eigen_to_flat(qp.H);
    gp.g = eigen_to_vec(qp.g);
    Eigen::MatrixXd A_exp = Eigen::MatrixXd::Zero(m_total, qp.n);
    Eigen::VectorXd b_exp(m_total);
    int row = 0;
    for (int i = 0; i < qp.m; ++i)
    { A_exp.row(row) = qp.A.row(i); b_exp(row) = qp.b(i); row++; }
    for (int i = 0; i < qp.n; ++i)
        if (qp.ubX(i) < 1e19) { A_exp(row, i) = 1.0; b_exp(row) = qp.ubX(i); row++; }
    for (int i = 0; i < qp.n; ++i)
        if (qp.lbX(i) > -1e19) { A_exp(row, i) = -1.0; b_exp(row) = -qp.lbX(i); row++; }
    gp.A = eigen_to_flat(A_exp);
    gp.b = eigen_to_vec(b_exp);
    return gp;
}
