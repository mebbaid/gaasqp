#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include <cstring>

namespace geoqp
{
    namespace internal
    {

        // Global stats instance (bench-only)
        static SchurStatsGlobal g_schur_stats;

        void schur_stats_reset()
        {
            g_schur_stats = SchurStatsGlobal();
        }

        SchurStatsGlobal &schur_stats_get()
        {
            return g_schur_stats;
        }


        const std::vector<double> &ConstraintCache::get_z(
            int constraint_idx,
            const double *A,
            const double *L_Q)
        {
            if (constraint_idx < 0 || constraint_idx >= m_)
            {
                throw std::out_of_range("Invalid constraint index");
            }

            auto &entry = cache_[constraint_idx];

            if (!entry.computed)
            {
                entry.z.resize(n_);
                const double *a_row = A + constraint_idx * n_;

                LinearAlgebra::forward_sub(L_Q, a_row, entry.z.data(), n_);
                entry.computed = true;
            }

            return entry.z;
        }

        bool SchurComplementManager::initialize(
            const std::vector<int> &face,
            const double *GEOQP_RESTRICT A,
            const double *GEOQP_RESTRICT L_Q,
            int n,
            ConstraintCache &cache)
        {
            k_ = static_cast<int>(face.size());
            ++stats_.initialize_calls;
            active_constraints_ = face;

            // store context for later validation / full recompute
            A_ptr_ = A;
            L_Q_ptr_ = L_Q;
            n_ = n;
            cache_ptr_ = &cache;

            if (k_ == 0)
            {
                L_S_.clear();
                // clear stored context
                A_ptr_ = nullptr; L_Q_ptr_ = nullptr; n_ = 0; cache_ptr_ = nullptr;
                return true;
            }

            std::vector<double> S(k_ * k_);
            double *GEOQP_RESTRICT S_ptr = S.data();

            for (int i = 0; i < k_; ++i)
            {
                const auto &z_i = cache.get_z(face[i], A, L_Q);
                const double *GEOQP_RESTRICT z_i_ptr = z_i.data();

                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                int j = 0;
                for (; j + 3 < n; j += 4)
                {
                    sum0 += z_i_ptr[j] * z_i_ptr[j];
                    sum1 += z_i_ptr[j + 1] * z_i_ptr[j + 1];
                    sum2 += z_i_ptr[j + 2] * z_i_ptr[j + 2];
                    sum3 += z_i_ptr[j + 3] * z_i_ptr[j + 3];
                }
                double diag = (sum0 + sum1) + (sum2 + sum3);
                for (; j < n; ++j)
                {
                    diag += z_i_ptr[j] * z_i_ptr[j];
                }
                S_ptr[i * k_ + i] = diag;

                for (int ji = i + 1; ji < k_; ++ji)
                {
                    const auto &z_j = cache.get_z(face[ji], A, L_Q);
                    const double *GEOQP_RESTRICT z_j_ptr = z_j.data();

                    sum0 = 0.0;
                    sum1 = 0.0;
                    sum2 = 0.0;
                    sum3 = 0.0;
                    int l = 0;
                    for (; l + 3 < n; l += 4)
                    {
                        sum0 += z_i_ptr[l] * z_j_ptr[l];
                        sum1 += z_i_ptr[l + 1] * z_j_ptr[l + 1];
                        sum2 += z_i_ptr[l + 2] * z_j_ptr[l + 2];
                        sum3 += z_i_ptr[l + 3] * z_j_ptr[l + 3];
                    }
                    double val = (sum0 + sum1) + (sum2 + sum3);
                    for (; l < n; ++l)
                    {
                        val += z_i_ptr[l] * z_j_ptr[l];
                    }
                    S_ptr[i * k_ + ji] = val;
                    S_ptr[ji * k_ + i] = val;
                }
            }

            L_S_.resize(k_ * k_);
            
            bool has_issues = false;
            for (int i = 0; i < k_; ++i)
            {
                if (S_ptr[i * k_ + i] <= 1e-14)
                {
                    has_issues = true;
                    break;
                }
            }
            
            if (has_issues)
            {
                return false;
            }
            
            try
            {
                LinearAlgebra::cholesky(S.data(), L_S_.data(), k_);

                // Optional debug print: if environment variable GEOQP_PRINT_SCHUR is set,
                // print the Cholesky factor L_S_ (lower triangular) for inspection.
                if (std::getenv("GEOQP_PRINT_SCHUR") != nullptr)
                {
                    std::cout << "[SCHUR] k=" << k_ << " L_S (lower-triangular):\n";
                    for (int i = 0; i < k_; ++i)
                    {
                        for (int j = 0; j <= i; ++j)
                        {
                            std::cout << " " << L_S_[i * k_ + j];
                        }
                        std::cout << "\n";
                    }
                }

                return true;
            }
            catch (const std::exception& e)
            {
                // Cholesky failed - matrix is not positive definite
                return false;
            }
            catch (...)
            {
                return false;
            }
        }

        bool SchurComplementManager::add_constraint(
            int constraint_idx,
            const double *GEOQP_RESTRICT A,
            const double *GEOQP_RESTRICT L_Q,
            int n,
            ConstraintCache &cache)
        {
            const auto &z_new = cache.get_z(constraint_idx, A, L_Q);
            const double *GEOQP_RESTRICT z_new_ptr = z_new.data();

            std::vector<double> s(k_);
            double *GEOQP_RESTRICT s_ptr = s.data();

            for (int i = 0; i < k_; ++i)
            {
                const auto &z_i = cache.get_z(active_constraints_[i], A, L_Q);
                const double *GEOQP_RESTRICT z_i_ptr = z_i.data();

                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                int j = 0;
                for (; j + 3 < n; j += 4)
                {
                    sum0 += z_i_ptr[j] * z_new_ptr[j];
                    sum1 += z_i_ptr[j + 1] * z_new_ptr[j + 1];
                    sum2 += z_i_ptr[j + 2] * z_new_ptr[j + 2];
                    sum3 += z_i_ptr[j + 3] * z_new_ptr[j + 3];
                }
                s_ptr[i] = (sum0 + sum1) + (sum2 + sum3);
                for (; j < n; ++j)
                {
                    s_ptr[i] += z_i_ptr[j] * z_new_ptr[j];
                }
            }

            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            int j = 0;
            for (; j + 3 < n; j += 4)
            {
                sum0 += z_new_ptr[j] * z_new_ptr[j];
                sum1 += z_new_ptr[j + 1] * z_new_ptr[j + 1];
                sum2 += z_new_ptr[j + 2] * z_new_ptr[j + 2];
                sum3 += z_new_ptr[j + 3] * z_new_ptr[j + 3];
            }
            double sigma = (sum0 + sum1) + (sum2 + sum3);
            for (; j < n; ++j)
            {
                sigma += z_new_ptr[j] * z_new_ptr[j];
            }

            std::vector<double> L_new((k_ + 1) * (k_ + 1));
            bool success = LinearAlgebra::cholesky_extend(
                L_S_.data(), k_,
                s.data(), sigma,
                L_new.data());

            if (success)
            {
                ++stats_.extend_success;
                schur_stats_get().extend_success += 1;
                L_S_ = std::move(L_new);
                active_constraints_.push_back(constraint_idx);
                k_++;

                // Validate incremental update by rebuilding S from cache and
                // comparing with L_S_ * L_S_^T. Use stored cache/context if available.
                ConstraintCache *use_cache = cache_ptr_ ? cache_ptr_ : &cache;
                const double *use_A = A_ptr_ ? A_ptr_ : A;
                const double *use_LQ = L_Q_ptr_ ? L_Q_ptr_ : L_Q;

                if (use_cache)
                {
                    const int kk = k_;
                    std::vector<double> S_cache(kk * kk);
                    for (int i = 0; i < kk; ++i)
                    {
                        const auto &z_i = use_cache->get_z(active_constraints_[i], use_A, use_LQ);
                        const double *GEOQP_RESTRICT z_i_ptr = z_i.data();
                        for (int j = i; j < kk; ++j)
                        {
                            const auto &z_j = use_cache->get_z(active_constraints_[j], use_A, use_LQ);
                            const double *GEOQP_RESTRICT z_j_ptr = z_j.data();
                            double sum = 0.0;
                            int t = 0;
                            for (; t + 3 < n_; t += 4)
                            {
                                sum += z_i_ptr[t] * z_j_ptr[t]
                                     + z_i_ptr[t+1] * z_j_ptr[t+1]
                                     + z_i_ptr[t+2] * z_j_ptr[t+2]
                                     + z_i_ptr[t+3] * z_j_ptr[t+3];
                            }
                            for (; t < n_; ++t) sum += z_i_ptr[t] * z_j_ptr[t];
                            S_cache[i * kk + j] = sum;
                            S_cache[j * kk + i] = sum;
                        }
                    }

                    // reconstruct and compare
                    std::vector<double> S_recon(kk * kk);
                    for (int i = 0; i < kk; ++i)
                        for (int j = 0; j < kk; ++j)
                        {
                            double v = 0.0;
                            for (int t = 0; t < kk; ++t) v += L_S_[i * kk + t] * L_S_[j * kk + t];
                            S_recon[i * kk + j] = v;
                        }

                    double diff_sq = 0.0, ref_sq = 0.0;
                    for (int i = 0; i < kk * kk; ++i) { double d = S_cache[i] - S_recon[i]; diff_sq += d*d; ref_sq += S_cache[i]*S_cache[i]; }
                    double rel_err = std::sqrt(diff_sq) / (std::sqrt(ref_sq) + 1e-20);

                    if (rel_err > 1e-10)
                    {
                        // Fallback: recompute Schur complement from scratch
                        ++stats_.extend_fallback;
                        schur_stats_get().extend_fallback += 1;
                        if (!initialize(active_constraints_, use_A, use_LQ, n_, *use_cache))
                        {
                            return false;
                        }
                    }
                }

                return true;
            }

            ++stats_.extend_fallback;
            schur_stats_get().extend_fallback += 1;
            return false;
        }

        bool SchurComplementManager::remove_constraint(int face_index)
        {
            if (face_index < 0 || face_index >= k_)
            {
                return false;
            }

            if (k_ == 1)
            {
                L_S_.clear();
                active_constraints_.clear();
                k_ = 0;
                return true;
            }

            std::vector<double> L_new((k_ - 1) * (k_ - 1));
            bool success = LinearAlgebra::cholesky_shrink(
                L_S_.data(), k_, face_index,
                L_new.data());

            if (success)
            {
                ++stats_.shrink_success;
                schur_stats_get().shrink_success += 1;
                L_S_ = std::move(L_new);
                active_constraints_.erase(active_constraints_.begin() + face_index);
                k_--;

                // Validate shrink using stored cache/context
                if (cache_ptr_ && A_ptr_ && L_Q_ptr_ && n_ > 0)
                {
                    const int kk = k_;
                    if (kk == 0) return true;

                    std::vector<double> S_cache(kk * kk);
                    for (int i = 0; i < kk; ++i)
                    {
                        const auto &zi = cache_ptr_->get_z(active_constraints_[i], A_ptr_, L_Q_ptr_);
                        for (int j = i; j < kk; ++j)
                        {
                            const auto &zj = cache_ptr_->get_z(active_constraints_[j], A_ptr_, L_Q_ptr_);
                            double s = 0.0;
                            for (int t = 0; t < n_; ++t) s += zi[t] * zj[t];
                            S_cache[i * kk + j] = s; S_cache[j * kk + i] = s;
                        }
                    }

                    std::vector<double> L_full(kk * kk);
                    try { LinearAlgebra::cholesky(S_cache.data(), L_full.data(), kk); }
                    catch (...) { return false; }

                    double diff_sq = 0.0, ref_sq = 0.0;
                    for (int i = 0; i < kk * kk; ++i) { double d = L_S_[i] - L_full[i]; diff_sq += d*d; ref_sq += L_full[i]*L_full[i]; }
                    double rel = std::sqrt(diff_sq) / (std::sqrt(ref_sq) + 1e-20);
                    if (rel > 1e-10)
                    {
                        ++stats_.shrink_fallback;
                        schur_stats_get().shrink_fallback += 1;
                        if (!initialize(active_constraints_, A_ptr_, L_Q_ptr_, n_, *cache_ptr_)) return false;
                    }
                }

                return true;
            }

            ++stats_.shrink_fallback;
            schur_stats_get().shrink_fallback += 1;
            return false;
        }

        bool SchurComplementManager::solve(
            const std::vector<double> &rhs,
            std::vector<double> &lambda) const
        {
            if (k_ == 0)
            {
                lambda.clear();
                return true;
            }

            if (static_cast<int>(rhs.size()) != k_)
            {
                return false;
            }

            // increment solve counter
            const_cast<SchurComplementManager*>(this)->stats_.solve_calls += 1;
            schur_stats_get().solve_calls += 1;

            lambda.resize(k_);

            std::vector<double> y(k_);
            LinearAlgebra::forward_sub(L_S_.data(), rhs.data(), y.data(), k_);
            LinearAlgebra::backward_sub(L_S_.data(), y.data(), lambda.data(), k_);

            return true;
        }

    } // namespace internal
} // namespace geoqp
