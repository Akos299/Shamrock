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
 * @file NodePrecondResidual.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

// #define NODE_PRECOND_RES_EDGES(X_RO, X_RW) \
//     /* inputs */ \
//     X_RO(shamrock::solvergraph::Indexes<u32>, sizes) \
//     X_RO(solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>, cell_neigh_graph) \
//     X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, block_cell_sizes) \
//     X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_res) \
//     /* outputs*/ \ X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_z)

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class NodePrecondRes : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;

        public:
        NodePrecondRes(u32 block_size) : block_size(block_size) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_phi_res;
            shamrock::solvergraph::IFieldSpan<Tscal> &spans_phi_z;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_phi_res,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_phi_z) {
            __internal_set_ro_edges(
                {sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi_res});
            __internal_set_rw_edges({spans_phi_z});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(3),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "NodePrecondRes"; };

        virtual std::string _impl_get_tex() const { return "TODO"; };
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_PRECOND_RES_EDGES
