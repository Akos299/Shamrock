// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodePrecondResidual.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */
#include "shammodels/ramses/modules/NodePrecondResidual.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/CGLaplacianStencil.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamsys/NodeInstance.hpp"

using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;

namespace {
    using Direction = shammodels::basegodunov::modules::Direction;

    template<class Tvec, class TgridVec>
    struct KernelPrecondRes {

        using Tscal            = shambase::VecComponent<Tvec>;
        using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;
        using AMRGraph         = shammodels::basegodunov::modules::AMRGraph;
        using Edges =
            typename shammodels::basegodunov::modules::NodePrecondRes<Tvec, TgridVec>::Edges;

        inline static void kernel(Edges &edges, u32 block_size) {

            edges.cell_neigh_graph.graph.for_each(
                [&](u64 id, const OrientedAMRGraph &oriented_cell_graph) {
                    auto &cell_sizes_span = edges.spans_block_cell_sizes.get_spans().get(id);
                    auto &phi_res_span    = edges.spans_phi_res.get_spans().get(id);
                    auto &phi_z_span      = edges.spans_phi_z.get_spans().get(id);

                    AMRGraph &graph_neigh_xp
                        = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::xp]);
                    AMRGraph &graph_neigh_xm
                        = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::xm]);
                    AMRGraph &graph_neigh_yp
                        = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::yp]);
                    AMRGraph &graph_neigh_ym
                        = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::ym]);
                    AMRGraph &graph_neigh_zp
                        = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::zp]);
                    AMRGraph &graph_neigh_zm
                        = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::zm]);

                    sham::EventList depends_list;

                    auto cell_sizes = cell_sizes_span.get_read_access(depends_list);
                    auto phi_res    = phi_res_span.get_read_access(depends_list);
                    auto phi_z      = phi_z_span.get_write_access(depends_list);

                    auto graph_iter_xp = graph_neigh_xp.get_read_access(depends_list);
                    auto graph_iter_xm = graph_neigh_xm.get_read_access(depends_list);
                    auto graph_iter_yp = graph_neigh_yp.get_read_access(depends_list);
                    auto graph_iter_ym = graph_neigh_ym.get_read_access(depends_list);
                    auto graph_iter_zp = graph_neigh_zp.get_read_access(depends_list);
                    auto graph_iter_zm = graph_neigh_zm.get_read_access(depends_list);

                    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
                    auto e               = q.submit(depends_list, [&](sycl::handler &cgh) {
                        u32 cell_count = (edges.sizes.indexes.get(id)) * block_size;

                        shambase::parallel_for(cgh, cell_count, "z = M^{-1} r", [=](u64 gid) {
                            const u32 cell_global_id = (u32) gid;
                            const u32 block_id       = cell_global_id / block_size;
                            const u32 cell_loc_id    = cell_global_id % block_size;

                            Tscal delta_cell = cell_sizes[block_id];

                            auto jac_weight = shammodels::basegodunov::Jacobi_weight<Tscal, Tvec>(
                                cell_sizes,
                                block_size,
                                cell_global_id,
                                graph_iter_xp,
                                graph_iter_xm,
                                graph_iter_yp,
                                graph_iter_ym,
                                graph_iter_zp,
                                graph_iter_zm);

                            phi_z[cell_global_id] = phi_res[cell_global_id] / jac_weight;

                            // if (jac_weight != 6.0 * delta_cell)
                            // logger::raw_ln("\n computed: \t ",jac_weight, "\t ", "expected:
                            // \t", 6.0 * delta_cell);
                        });
                    });

                    cell_sizes_span.complete_event_state(e);
                    phi_res_span.complete_event_state(e);
                    phi_z_span.complete_event_state(e);

                    graph_neigh_xp.complete_event_state(e);
                    graph_neigh_xm.complete_event_state(e);
                    graph_neigh_yp.complete_event_state(e);
                    graph_neigh_ym.complete_event_state(e);
                    graph_neigh_zp.complete_event_state(e);
                    graph_neigh_zm.complete_event_state(e);
                });
        }
    };
} // namespace

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::NodePrecondRes<Tvec, TgridVec>::_impl_evaluate_internal() {
    StackEntry stack_loc{};
    auto edges = get_edges();

    edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
    edges.spans_phi_res.check_sizes(edges.sizes.indexes);
    edges.spans_phi_z.ensure_sizes(edges.sizes.indexes);

    KernelPrecondRes<Tvec, TgridVec>::kernel(edges, block_size);
}

template class shammodels::basegodunov::modules::NodePrecondRes<f64, i64_3>;
template class shammodels::basegodunov::modules::NodePrecondRes<f64_3, i64_3>;
