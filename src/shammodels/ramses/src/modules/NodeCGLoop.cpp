// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeCGLoop.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/NodeCGLoop.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <shambackends/sycl.hpp>
#include <memory>

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    void NodeCGLoop<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();
        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_phi.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_phi_res.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_p.ensure_sizes(edges.sizes.indexes);

        gz.merge_phi_ghost();

        // logger::raw_ln("CG-MAIN-LOOP [INIT] ");
        node0.evaluate();
        // logger::raw_ln("CG-MAIN-LOOP-[OLD-VAL] ");
        node1.evaluate();

        u32 k = 0;
        logger::raw_ln(" k = ", k);
        logger::raw_ln(" RES = ", edges.old_values.value);
        while ((k < Niter_max)) {
            // increment iteration
            k = k + 1;
            //     logger::raw_ln("CG-MAIN-LOOP-[for loop over k] ", k);

            //     logger::raw_ln("CG-MAIN-LOOP-[AP] ");
            node2.evaluate();

            //     logger::raw_ln("CG-MAIN-LOOP-[P x AP] ");
            node3.evaluate();

            //     logger::raw_ln("CG-MAIN-LOOP-[A-norm av] ", edges.e_norm.value);
            node4.evaluate();
            //     logger::raw_ln("CG-MAIN-LOOP-[A-norm af] ", edges.e_norm.value);

            //     logger::raw_ln("CG-MAIN-LOOP-[alpha av] ", edges.alpha.value);
            edges.alpha.value = edges.old_values.value / edges.e_norm.value;
            //     logger::raw_ln("CG-MAIN-LOOP-[alpha af] ", edges.alpha.value);

            //     logger::raw_ln("CG-MAIN-LOOP-[New-phi] ");
            node5.evaluate();

            //     logger::raw_ln("CG-MAIN-LOOP-[New-res] ");
            node6.evaluate();

            //     logger::raw_ln("CG-MAIN-LOOP-[NEW-VAL] ", edges.new_values.value);
            node7.evaluate();
            //     logger::raw_ln("CG-MAIN-LOOP-[NEW-VAL] ", edges.new_values.value);

            //     logger::raw_ln("CG-MAIN-LOOP-[beta av] ", edges.beta.value);
            edges.beta.value = edges.new_values.value / edges.old_values.value;
            //     logger::raw_ln("CG-MAIN-LOOP-[beta af] ", edges.beta.value);

            //     logger::raw_ln("CG-MAIN-LOOP-[update-old] ");
            edges.old_values.value = edges.new_values.value;

            logger::raw_ln(" k = ", k);
            logger::raw_ln(" RES = ", edges.old_values.value);

            //     logger::raw_ln("CG-MAIN-LOOP-[New-rep] ");
            node8.evaluate();

            if (sycl::sqrt(edges.old_values.value) < tol)
                break;
        }
    }

    template<class Tvec, class TgridVec>
    std::string NodeCGLoop<Tvec, TgridVec>::_impl_get_tex() {

        std::string tex = R"tex(
             CG Main Loop
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeCGLoop<f64_3, i64_3>;
