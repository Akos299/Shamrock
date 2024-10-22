// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputePseudoGradient.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/amr/basegodunov/modules/ComputePseudoGradient.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraphLinkiterator;

template<class T, class Tvec, class ACCField>
inline T get_pseudo_grad(
    const u32 cell_global_id,
    const shambase::VecComponent<Tvec> delta_cell,
    const AMRGraphLinkiterator &graph_iter_xp,
    const AMRGraphLinkiterator &graph_iter_xm,
    const AMRGraphLinkiterator &graph_iter_yp,
    const AMRGraphLinkiterator &graph_iter_ym,
    const AMRGraphLinkiterator &graph_iter_zp,
    const AMRGraphLinkiterator &graph_iter_zm,
    ACCField &&field_access)

{

    using namespace sham;
    using namespace sham::details;

    auto grad_scalar = [&](T u_curr, T u_neigh) {
        T max = g_sycl_max(g_sycl_abs(u_curr), g_sycl_abs(u_neigh));
        max   = g_sycl_abs(u_curr - u_neigh) / max;
        return g_sycl_max(g_sycl_min(max, 1.0), 0.0);
    };

    auto get_amr_grad = [&](auto &graph_links) -> T {
        T acc   = shambase::VectorProperties<T>::get_zero();
        u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
            T u_cur_acc   = field_access(cell_global_id);
            T u_neigh_acc = field_access(id_b);
            acc           = g_sycl_max(acc, grad_scalar(u_cur_acc, u_neigh_acc));
        });

        return (cnt > 0) ? acc : shambase::VectorProperties<T>::get_zero();
    };

    T u_xp_dir = get_amr_grad(graph_iter_xp);
    T u_xm_dir = get_amr_grad(graph_iter_xm);
    T u_yp_dir = get_amr_grad(graph_iter_yp);
    T u_ym_dir = get_amr_grad(graph_iter_ym);
    T u_zp_dir = get_amr_grad(graph_iter_zp);
    T u_zm_dir = get_amr_grad(graph_iter_zm);

    T res = max_8points(
        shambase::VectorProperties<T>::get_zero(),
        shambase::VectorProperties<T>::get_zero(),
        u_xp_dir,
        u_xm_dir,
        u_yp_dir,
        u_ym_dir,
        u_zp_dir,
        u_zm_dir);
    return res;
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputePseudoGradient<Tvec, TgridVec>::
    compute_pseudo_gradient() {
    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tscal> result
        = utility.make_compute_field<Tscal>("pseudo grad", AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });
    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_ghost                                 = ghost_layout.get_field_idx<Tscal>("rho");

    storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

        sycl::queue &q = shamsys::instance::get_compute_queue();

        sycl::buffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_ghost);

        AMRGraph &graph_neigh_xp
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]);
        AMRGraph &graph_neigh_xm
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]);
        AMRGraph &graph_neigh_yp
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]);
        AMRGraph &graph_neigh_ym
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]);
        AMRGraph &graph_neigh_zp
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]);
        AMRGraph &graph_neigh_zm
            = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]);

        sycl::buffer<Tscal> &block_cell_sizes
            = storage.cell_infos.get().block_cell_sizes.get_buf_check(id);
        sycl::buffer<Tvec> &cell0block_aabb_lower
            = storage.cell_infos.get().cell0block_aabb_lower.get_buf_check(id);

        q.submit([&](sycl::handler &cgh) {
            AMRGraphLinkiterator graph_iter_xp{graph_neigh_xp, cgh};
            AMRGraphLinkiterator graph_iter_xm{graph_neigh_xm, cgh};
            AMRGraphLinkiterator graph_iter_yp{graph_neigh_yp, cgh};
            AMRGraphLinkiterator graph_iter_ym{graph_neigh_ym, cgh};
            AMRGraphLinkiterator graph_iter_zp{graph_neigh_zp, cgh};
            AMRGraphLinkiterator graph_iter_zm{graph_neigh_zm, cgh};

            sycl::accessor acc_aabb_block_lower{cell0block_aabb_lower, cgh, sycl::read_only};
            sycl::accessor acc_aabb_cell_size{block_cell_sizes, cgh, sycl::read_only};

            sycl::accessor rho{buf_rho, cgh, sycl::read_only};
            sycl::accessor pseudo_grad_rho{
                shambase::get_check_ref(result.get_buf(id)), cgh, sycl::write_only, sycl::no_init};

            u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

            shambase::parralel_for(cgh, cell_count, "compute_pseudo_grad_rho", [=](u64 gid) {
                const u32 cell_global_id = (u32) gid;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                Tscal delta_cell = acc_aabb_cell_size[block_id];

                auto result = get_pseudo_grad<Tscal, Tvec>(
                    cell_global_id,
                    delta_cell,
                    graph_iter_xp,
                    graph_iter_xm,
                    graph_iter_yp,
                    graph_iter_ym,
                    graph_iter_zp,
                    graph_iter_zm,
                    [=](u32 id) {
                        return rho[id];
                    });

                pseudo_grad_rho[cell_global_id] = result;
            });
        });
    });

    storage.pseudo_gradient_rho.set(std::move(result));
}

template class shammodels::basegodunov::modules::ComputePseudoGradient<f64_3, i64_3>;
