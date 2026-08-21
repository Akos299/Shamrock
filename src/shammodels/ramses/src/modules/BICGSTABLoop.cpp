// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BICGSTABLoop.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/BICGSTABLoop.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <shambackends/sycl.hpp>
#include <memory>
#include <utility>

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    void NodeBICGSTABLoop<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();
        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_phi.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_phi_res.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_res_bis.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_p.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_Ap.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_s.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_As.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_hadamard_prod.ensure_sizes(edges.sizes.indexes);

        /* compute r0 =  4*\pi*G* \left( \rho - \bar{\rho} \right) - A \phi_{0}
         *          r'0 = 0.5*r0
         *          p0 =  r0
         */
        node_init.evaluate();

        /** compute <b,b> */
        node_ddot_rhs.evaluate();
        // if (shamcomm::world_rank() == 0) {
        //     logger::raw_ln("k= \t ", k, " \t rhs_norm = \t", edges.old_values.value);
        // }

        /** compute <r'_0,r'_0>*/
        node_ddot_rstarj_rstarj.evaluate();

        u32 k = 0;

        if (shamcomm::world_rank() == 0) {
            logger::raw_ln("rhs value = \t", edges.rhs_norm_value.value, "\t\n\n");
            logger::raw_ln(
                "[BICGSTAB] \t k = \t",
                k,
                "\t ||r_k||_2 / ||b_rhs||_2 =  \t ",
                sycl::sqrt(edges.shadow_res_norm.value / edges.rhs_norm_value.value),
                "\t\n\n");
        }

        /* Main loop */
        while ((k < Niter_max)) {

            /** compute compute Hadamard product r_0 x r'_0 */
            node_had_prod_rj_rp0.evaluate();

            /** get the dot product <r_0, r'_0> and assign its value to  edges.old_values.value */
            node_ddot_rj_rp0.evaluate();

            // if (shamcomm::world_rank() == 0) {
            //     logger::raw_ln("k= \t ", k, " \t res = \t", edges.old_values.value);
            // }

            //--------------------------------
            /* comm of p vector*/
            //----------------------------

            if (true) {
                // //exchange p vector
                node_gz_p.evaluate();
                node_exch_gz_p.evaluate();
                node_replace_gz_p.evaluate();
            }

            /** compute Ap_{k} */
            node_Apj.evaluate();

            /** compute Hadamard product Ap_{k} x r'_{0}*/
            node_had_Apj_rp0.evaluate();

            /** compute <Ap_{k}, r'_{0}> and assign its value to edges.e_norm.value*/
            node_ddot_Apj_rp0.evaluate();

            /** compute \alpha_{k} = \frac{ <r_{k},r'_{0}> }{ <r'_{0},Ap_{k}> }*/
            const auto sigma        = edges.e_norm.value;
            const f64 breakdown_tol = 0.0;
            /** BiCGSTAB breakdown check */
            if (sycl::fabs(sigma) <= breakdown_tol) {
                logger::raw_ln("BiCGSTAB breakdown: sigma = ", sigma);
                break;
                // restart mechanism to be implemented
            } else {
                edges.alpha.value = edges.old_values.value / sigma;
            }

            const auto alp_saved = edges.alpha.value;

            /** compute s_{k} = r_{k} - alpha_{k}Ap_{k} */
            edges.e_norm.value = 0;
            edges.beta.value   = 1;
            edges.alpha.value  = -alp_saved;
            node_sj_vec.evaluate();

            /** compute <s_{k}, s_{k}> and set its value to edges.e_norm.value*/
            node_ddot_sj_sj.evaluate();
            // if (shamcomm::world_rank() == 0) {
            //     logger::raw_ln(" [BICGSTAB] \t <s_k,s_k> \t ", edges.e_norm.value, "\n");
            // }

            /** perform cvg test*/
            const auto rel_sj_sqr = edges.e_norm.value / edges.rhs_norm_value.value;
            if (rel_sj_sqr < tol_cvg * tol_cvg) {
                edges.alpha.value = alp_saved;
                node_new_phi_happy_break.evaluate();
                if (shamcomm::world_rank() == 0) {
                    logger::raw_ln(
                        "[BICGSTAB] \t Converge on s-residual: <s_k,s_k> / <b_rhs, b_rhs> = ",
                        rel_sj_sqr,
                        " \t\n");
                }

                break;
            }

            //--------------------------------
            /* comm of s vector*/
            //----------------------------
            if (true) {
                // //exchange s vector
                node_gz_s.evaluate();
                node_exch_gz_s.evaluate();
                node_replace_gz_s.evaluate();
            }

            /** compute  As_{k}*/
            node_Asj.evaluate();
            /** compute As_{k} x s_{k}*/
            node_had_Asj_sj.evaluate();
            /** compute <As_{k},s_{k}> and set its value to edges.e_norm.value*/
            node_ddot_Asj_sj.evaluate();

            // if (shamcomm::world_rank() == 0) {
            //     logger::raw_ln("<As_k,s_k> \t ", edges.e_norm.value, "\n");
            // }

            /** compute <As_{k},As_{k} and set its value to edges.new_values.value*/
            node_ddot_Asj_Asj.evaluate();

            // if (shamcomm::world_rank() == 0) {
            //     logger::raw_ln("<As_k,As_k> \t ", edges.new_values.value, "\n");
            // }

            /** compute w_{k} and set its value to edges.w_stab.value*/

            // breakdown: ||As_k||^2 too small
            if (edges.new_values.value <= breakdown_tol) {
                // handle breakdown/restart
                break;
            } else {
                edges.w_stab.value = (edges.e_norm.value / edges.new_values.value);
            }

            auto w_saved = edges.w_stab.value;

            /** breakdown: omega_k too small */
            if (sycl::fabs(w_saved) <= breakdown_tol) {
                logger::raw_ln("BiCGSTAB breakdown: omega = ", w_saved);
                break;
                // handle breakdown/restart
            }

            /** compute new-phi*/
            edges.alpha.value = alp_saved;
            node_new_phi.evaluate();

            /** compute new-residual*/
            edges.w_stab.value     = -w_saved;
            edges.e_norm.value     = 0;
            edges.new_values.value = 1;
            node_new_res.evaluate();

            /** compute <r_{k+1}, r_{k+1}> and assign its value to edges.e_norm.value*/
            node_ddot_rj_rj.evaluate();

            /** perform cvg test*/
            const auto rel_rj_sqr = edges.e_norm.value / edges.rhs_norm_value.value;
            if (shamcomm::world_rank() == 0) {
                logger::raw_ln(
                    " [BICGSTAB] \t k = \t ",
                    k + 1,
                    "\t ||r_k||_2 / ||b_rhs||_2 = \t ",
                    sycl::sqrt(rel_rj_sqr),
                    "\t\n");
            }
            if (rel_rj_sqr < tol_cvg * tol_cvg) {
                if (shamcomm::world_rank() == 0) {
                    logger::raw_ln("[BICGSTAB] \t Converge on residual \t\n");
                }
                break;
            }
            /** compute r_{k+1} x r'{0} */
            node_had_rjnew_rp0.evaluate();
            /**compute <r_{k+1}, r'{0}> and assign its value to edges.new_values.value */
            node_ddot_rjnew_rp0.evaluate();

            // if (shamcomm::world_rank() == 0) {
            //     logger::raw_ln("<r_{k+1},r'_0> \t ", edges.new_values.value, "\n");
            // }

            const auto rho_new = edges.new_values.value;
            // use the relative error
            if ((rho_new * rho_new) < (tol_happy_bk * tol_happy_bk) * (edges.e_norm.value)
                                          * edges.shadow_res_norm.value) {

                // /** Restart BiCGSTAB when the shadow-residual inner product becomes too small */
                // r'_0 <- r_{k+1}
                // p_{k+1} <- r_{k+1}
                node_overwrite_rp0.evaluate();
                node_overwrite_p.evaluate();

                /** recompute <r'_0,r'_0>*/
                node_ddot_rstarj_rstarj.evaluate();

            } else {

                // /**compute beta_{k} = (alpha_{k} / w_{k}) x (<r_{k+1}, r'{0}> / <r_{k}, r'{0}>)
                // */
                edges.beta.value = (alp_saved / w_saved) * (rho_new / edges.old_values.value);

                /** p_{k+1}
                 *   = r_{k+1}
                 *   + beta_k (p_k - omega_k Ap_k)
                 */
                edges.e_norm.value = 1;
                edges.alpha.value  = -w_saved * edges.beta.value;

                node_new_p_vec.evaluate();
            }

            // increment iteration
            k = k + 1;
        }

        edges.nb_iter.value = k;

        if (true) {
            // update ghost for gravitational forces computation
            node_gz_phi.evaluate();
            node_exch_gz_phi.evaluate();
            node_replace_gz_phi.evaluate();
        }
    }

    template<class Tvec, class TgridVec>
    std::string NodeBICGSTABLoop<Tvec, TgridVec>::_impl_get_tex() const {

        std::string tex = R"tex(
             BICGSTAB Main Loop
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeBICGSTABLoop<f64_3, i64_3>;
