// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file MatrixExponential_mdspan.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 */

#include "LinalgUtilities_mdspan.hpp"
#include "shammath/MatrixExponential.hpp"

namespace shammath {

    inline auto sequence_mk() {
        std::array<i64, 9> seq = {0};
        seq[0]                 = 2;
        seq[1]                 = 4;
        seq[2]                 = 6;
        seq[3]                 = 9;
        seq[4]                 = 12;
        seq[5]                 = 16;
        seq[6]                 = 20;
        seq[7]                 = 25;
        seq[8]                 = 30;

        return seq;
    }

    inline auto sequence_qk() {
        std::array<i64, 9> seq = {0};
        seq[0]                 = 1;
        seq[1]                 = 2;
        seq[2]                 = 2;
        seq[3]                 = 3;
        seq[4]                 = 3;
        seq[5]                 = 4;
        seq[6]                 = 4;
        seq[7]                 = 5;
        seq[8]                 = 5;
        return seq;
    }

    inline auto sequence_rk() {
        std::array<i64, 9> seq = {0};
        seq[0]                 = 2;
        seq[1]                 = 2;
        seq[2]                 = 3;
        seq[3]                 = 3;
        seq[4]                 = 4;
        seq[5]                 = 4;
        seq[6]                 = 5;
        seq[7]                 = 5;
        seq[8]                 = 6;
        return seq;
    }

    inline auto sequence_theta_mk() {
        std::array<f64, 9> seq = {0};
        seq[0]                 = 2.5810e-8;
        seq[1]                 = 3.3972e-4;
        seq[2]                 = 9.0657e-3;
        seq[3]                 = 8.9578e-2;
        seq[4]                 = 2.9962e-1;
        seq[5]                 = 7.80e-1;
        seq[6]                 = 1.4383;
        seq[7]                 = 2.4286;
        seq[8]                 = 3.5397;
        return seq;
    }

    inline auto sequence_ntheta_mk() {
        std::array<f64, 9> seq = {0};
        seq[0]                 = 8.7334e-6;
        seq[1]                 = 1.6778e-3;
        seq[2]                 = 1.7720e-3;
        seq[3]                 = 1.1354e-1;
        seq[4]                 = 3.2690e-1;
        seq[5]                 = 7.8738e-1;
        seq[6]                 = 1.4383;
        seq[7]                 = 2.42860;
        seq[8]                 = 3.5397;
        return seq;
    }

    inline auto define_bexp_coef1() {
        std::array<f64, 30> coefs = {0};

        coefs[0]  = 1.0;
        coefs[1]  = -1.0;
        coefs[2]  = 0.5;
        coefs[3]  = -0.16666666666666666;
        coefs[4]  = 0.041666666666666664;
        coefs[5]  = -0.008333333333333333;
        coefs[6]  = 0.001388888888888889;
        coefs[7]  = -0.0001984126984126984;
        coefs[8]  = 2.48015873015873e-05;
        coefs[9]  = -2.7557319223985893e-06;
        coefs[10] = 2.755731922398589e-07;
        coefs[11] = -2.505210838544172e-08;
        coefs[12] = 2.08767569878681e-09;
        coefs[13] = -1.6059043836821613e-10;
        coefs[14] = 1.1470745597729725e-11;
        coefs[15] = -7.647163731819816e-13;
        coefs[16] = 4.779477332387385e-14;
        coefs[17] = -2.8114572543455206e-15;
        coefs[18] = 1.5619206968586225e-16;
        coefs[19] = -8.22063524662433e-18;
        coefs[20] = 4.110317623312165e-19;
        coefs[21] = -1.9572941063391263e-20;
        coefs[22] = 8.896791392450574e-22;
        coefs[23] = -3.868170170630684e-23;
        coefs[24] = 1.6117375710961184e-24;
        coefs[25] = -6.446950284384474e-26;
        coefs[26] = 2.4795962632247976e-27;
        coefs[27] = -9.183689863795546e-29;
        coefs[28] = 3.279889237069838e-30;
        coefs[29] = -1.1309962886447716e-31;

        return coefs;
    }

    inline auto define_bexp_coef2() {
        std::array<f64, 30> coefs = {0};
        coefs[0]                  = 1.0;
        coefs[1]                  = 1.0;
        coefs[2]                  = 0.5;
        coefs[3]                  = 0.16666666666666666;
        coefs[4]                  = 0.041666666666666664;
        coefs[5]                  = 0.008333333333333333;
        coefs[6]                  = 0.001388888888888889;
        coefs[7]                  = 0.0001984126984126984;
        coefs[8]                  = 2.48015873015873e-05;
        coefs[9]                  = 2.7557319223985893e-06;
        coefs[10]                 = 2.755731922398589e-07;
        coefs[11]                 = 2.505210838544172e-08;
        coefs[12]                 = 2.08767569878681e-09;
        coefs[13]                 = 1.6059043836821613e-10;
        coefs[14]                 = 1.1470745597729725e-11;
        coefs[15]                 = 7.647163731819816e-13;
        coefs[16]                 = 4.779477332387385e-14;
        coefs[17]                 = 2.8114572543455206e-15;
        coefs[18]                 = 1.5619206968586225e-16;
        coefs[19]                 = 8.22063524662433e-18;
        coefs[20]                 = 4.110317623312165e-19;
        coefs[21]                 = 1.9572941063391263e-20;
        coefs[22]                 = 8.896791392450574e-22;
        coefs[23]                 = 3.868170170630684e-23;
        coefs[24]                 = 1.6117375710961184e-24;
        coefs[25]                 = 6.446950284384474e-26;
        coefs[26]                 = 2.4795962632247976e-27;
        coefs[27]                 = 9.183689863795546e-29;
        coefs[28]                 = 3.279889237069838e-30;
        coefs[29]                 = 1.1309962886447716e-31;

        return coefs;
    }

    template<typename T>
    inline void mdspan_order_scale(
        const int K,
        std::array<i64, 9> &seq_mk,
        std::array<f64, 9> &seq_theta_mk,
        const mdspan2D<T> &A,
        const size_t size_A,
        i64 &k_star,
        i64 &m_star,
        i64 &s_star) {
        m_star = seq_mk[K - 1];
        k_star = K;
        s_star = 0;

        T norm_A = 0;
        mdspan_L1_norm<T>(A, norm_A);

        i64 s_tilde = static_cast<i64>(sycl::ceil(sycl::max(
            static_cast<f64>(0.0), static_cast<f64>(sycl::log2(norm_A / seq_theta_mk[K - 1])))));
        s_star      = s_tilde;
        i64 k       = 2;
        bool cond   = false;
        for (; k < (K + 1) && !cond; k++) {
            cond   = (norm_A <= seq_theta_mk[k - 1]) && (norm_A <= seq_theta_mk[K - 1]);
            m_star = cond * seq_mk[k - 1] + !cond * m_star;
            k_star = cond * k + !cond * k_star;
        }
        i64 k_choice = sycl::min(K, k); // if we break the preceding loop then use k else K
        auto ld_7    = [&](f64 s_val) {
            k_star = k_choice - 1;
            s_star = fmax(0, s_val);
            m_star = seq_mk[k_star - 1];
        };

        auto ld_8 = [&](f64 s_val) {
            k_star = k_choice - 2;
            s_star = sycl::fmax(1, s_val + 1);
            m_star = seq_mk[k_star - 1];
        };

        auto ld_9 = [&](f64 s_val) {
            k_star = k_choice - 3;
            s_star = sycl::fmax(2, s_val + 2);
            m_star = seq_mk[k_star - 1];
        };

        f64 cmp_1 = norm_A / (1 << s_tilde);

        i64 val_2 = ((k_choice >= 8) && (cmp_1 <= 2 * seq_theta_mk[k_choice - 3])) * 2;
        i64 val_3 = ((k_choice >= 9) && (cmp_1 <= 4 * seq_theta_mk[k_choice - 4])) * 3;
        i64 val_1 = ((k_choice >= 7) && (cmp_1 <= seq_theta_mk[k_choice - 2]));

        i64 val = sycl::max(val_1, sycl::max(val_2, val_3));

        auto process_val = [&](int val) {
            if (val == 1) {
                ld_7(s_tilde);
            } else if (val == 2) {
                ld_8(s_tilde);
            } else if (val == 3) {
                ld_9(s_tilde);
            }
        };

        process_val(val);
    }

    template<typename T>
    inline void mdspan_taylor_eval(
        const i64 r,
        consyt i64 q,
        std::array<f64, 30> &bi_seq,
        const size_t size,
        mdspan2D<T> &A,
        mdspan2D<T> &F,
        mdspan2D<T> B,
        mdspan2D<T> I,
        mdspan2D<T> Id) {

        for (auto k = r - 1; k >= 0; k--) {
            set_nul_to_identity_2d_mdspan<T>(Id);
            set_nul_to_identity_2d_mdspan<T>(I);
            i64 cc = 0;

            for (auto j = 1; j <= q; j++) {
                copy_between_2d_mdspan<T>(I, Id);
                mdspan_MatMatMut<T>(A, I, Id);
                cc = q * k + j;
                mdspan_MatMatAdd<T>(B, Id, 1.0, bi_seq[cc]);
            }

            i64 cond = (k >= 1);
            mdspan_MatMatAdd<T>(F, B);
            mdspan_MatMatAdd<T>(Id, I, cons, (1 - cond));
            mdspan_MatMatMut<T>(F, Id, I);
            copy_between_2d_mdspan<T>(F, I);
        }
        compute_add_id_scal<T>(F);
    }

    template<typename T>
    inline void mat_expo(
        i64 K,
        mdspan2D<T> &A,
        mdspan2D<T> &F,
        mdspan2D<T> &B,
        mdspan2D<T> &I,
        mdspan2D<T> &Id,
        const size_t size_A) {
        auto seq_mk        = sequence_mk();
        auto seq_qk        = sequence_qk();
        auto seq_rk        = sequence_rk();
        auto seq_ntheta_mk = sequence_ntheta_mk();
        auto seq_bi        = define_bexp_coef2();

        i64 k_star{0}, m_star{0}, s_star{0};
        // computation of k*, s*, m*
        order_scale(K, seq_mk, seq_ntheta_mk, A, size_A, k_star, m_star, s_star);
        i64 r = seq_rk[k_star - 1];
        i64 q = seq_qk[k_star - 1];
        // scaling step
        i64 pw           = (1 << s_star);
        f64 scale_factor = 1.0 / pw;
        mdspan_scalMat<T>(A, scale_factor);
        // Taylor polynomial evaluation
        mdspan_taylor_eval<T>(r, q, seq_bi, size_A, A, F, B, I, Id);
        // squaring step
        set_nul_2d_mdspan<T>(Id);
        set_nul_to_identity_2d_mdspan<T>(Id);
        set_nul_2d_mdspan<T>(I);
        set_nul_to_identity_2d_mdspan<T>(I);

        for (auto j = 1; j <= pw; j++) {
            copy_between_2d_mdspan<T>(I, Id);
            mdspan_MatMatMut<T>(F, I, Id);
        }

        copy_between_2d_mdspan<T>(A, Id);
    }
} // namespace shammath
