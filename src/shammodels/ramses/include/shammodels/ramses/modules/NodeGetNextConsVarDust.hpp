// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file NodeGetNextConsVarDust.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/solvegraph/NeighGraphLinkFieldEdge.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <memory>
#include <string>

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    class NodeGetNextConsVarDust : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;
        Tscal dt_over2;
        u32 ndust;

        public:
        NodeGetNextConsVarDust(u32 block_size, u32 ndust) : block_size(block_size), ndust(ndust) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_dt_rho_old_d;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_rho_next_d;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_rhov_old_d;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_dt_rhov_old_d;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_phi_g_old;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_phi_g_next;
            const shamrock::solvergraph::ScalarEdge<Tscal> &dt_over2;
            shamrock::solvergraph::IFieldSpan<Tvec> &spans_rhov_next_d;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_dt_rho_old_d,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rho_next_d,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_rhov_old_d,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_dt_rhov_old_d,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_phi_g_old,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_phi_g_next,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> dt_over2,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_rhov_next_d) {

            __internal_set_ro_edges(
                {sizes,
                 spans_dt_rho_old_d,
                 spans_rho_next_d,
                 spans_rhov_old_d,
                 spans_dt_rhov_old_d,
                 spans_phi_g_old,
                 spans_phi_g_next,
                 dt_over2});

            __internal_set_rw_edges({spans_rhov_next_d});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(3),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(4),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(5),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(6),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(7),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "NodeGetNextConsVarDust"; };

        virtual std::string _impl_get_tex() const { return "TODO"; }
    };

} // namespace shammodels::basegodunov::modules
