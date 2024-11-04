// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ModifiedSecondDerivative.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shammodels/amr/basegodunov/modules/ModifiedSecondDerivative.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraphLinkiterator;

template<class T, class Tvec, class ACCField>
inline T modif_second_derivative(
    const u32 cell_global_id,
    const shambase::VecComponent<Tvec> delta_cell,
    const AMRGraphLinkiterator &graph_iter_xp,
    const AMRGraphLinkiterator &graph_iter_xm,
    const AMRGraphLinkiterator &graph_iter_yp,
    const AMRGraphLinkiterator &graph_iter_ym,
    const AMRGraphLinkiterator &graph_iter_zp,
    const AMRGraphLinkiterator &graph_iter_zm,
    ACCField &&field_access) {
    using namespace sham;
    using namespace sham::details;

    auto get_avg_neigh = [&](auto &graph_links) -> T {
        T acc   = shambase::VectorProperties<T>::get_zero();
        u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
            acc += field_access(id_b);
        });
        return (cnt > 0) ? acc / cnt : shambase::VectorProperties<T>::get_zero();
    };

    auto eps_ref = 0.01;
    auto epsilon = shambase::get_epsilon<T>();
    T u_cur      = field_access(cell_global_id);
    T u_xp       = get_avg_neigh(graph_iter_xp);
    T u_xm       = get_avg_neigh(graph_iter_xm);
    T u_yp       = get_avg_neigh(graph_iter_yp);
    T u_ym       = get_avg_neigh(graph_iter_ym);
    T u_zp       = get_avg_neigh(graph_iter_zp);
    T u_zm       = get_avg_neigh(graph_iter_zm);

    T delta_u_xp = u_xp - u_cur;
    T delta_u_xm = u_xm - u_cur;
    T delta_u_yp = u_yp - u_cur;
    T delta_u_ym = u_ym - u_cur;
    T delta_u_zp = u_zp - u_cur;
    T delta_u_zm = u_zm - u_cur;

    T scalar_x = g_sycl_abs(u_xp) + g_sycl_abs(u_xm) + 2 * g_sycl_abs(u_cur);
    T scalar_y = g_sycl_abs(u_yp) + g_sycl_abs(u_ym) + 2 * g_sycl_abs(u_cur);
    T scalar_z = g_sycl_abs(u_zp) + g_sycl_abs(u_zm) + 2 * g_sycl_abs(u_cur);

    T res_x = g_sycl_abs(delta_u_xm + delta_u_xp)
              / (g_sycl_abs(delta_u_xm) + g_sycl_abs(delta_u_xp) + eps_ref * scalar_x + epsilon);
    T res_y = g_sycl_abs(delta_u_ym + delta_u_yp)
              / (g_sycl_abs(delta_u_ym) + g_sycl_abs(delta_u_yp) + eps_ref * scalar_y + epsilon);
    T res_z = g_sycl_abs(delta_u_zm + delta_u_zp)
              / (g_sycl_abs(delta_u_zm) + g_sycl_abs(delta_u_zp) + eps_ref * scalar_z + epsilon);

    // return g_sycl_max(res_x, g_sycl_max(res_y, res_z)) * delta_cell;
    // return g_sycl_max(res_x, g_sycl_max(res_y, res_z));
    return (res_x + res_y + res_z);
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ModifiedSecondDerivative<Tvec, TgridVec>::
    compute_modified_second_derivative() {
    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tscal> result_rho = utility.make_compute_field<Tscal>(
        "second derivative rho", AMRBlock::block_size, [&](u64 id) {
            return storage.merged_patchdata_ghost.get().get(id).total_elements;
        });

    shamrock::ComputeField<Tscal> result_press = utility.make_compute_field<Tscal>(
        "second derivative press", AMRBlock::block_size, [&](u64 id) {
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
        // sycl::buffer<Tvec> &buf_vel = shambase::get_check_ref(storage.vel.get().get_buf(id));
        sycl::buffer<Tscal> &buf_press = shambase::get_check_ref(storage.press.get().get_buf(id));

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
            sycl::accessor press{buf_press, cgh, sycl::read_only};
            sycl::accessor sec_der_rho{
                shambase::get_check_ref(result_rho.get_buf(id)),
                cgh,
                sycl::write_only,
                sycl::no_init};
            sycl::accessor sec_der_press{
                shambase::get_check_ref(result_press.get_buf(id)),
                cgh,
                sycl::write_only,
                sycl::no_init};
            u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

            shambase::parralel_for(cgh, cell_count, "second derivative", [=](u64 gid) {
                const u32 cell_global_id = (u32) gid;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                Tscal delta_cell = acc_aabb_cell_size[block_id];

                // TODO : will be optimize later. Think of how to combine those 2 in on so we
                // can iterate though cell just on times and do all computation
                auto result_r = modif_second_derivative<Tscal, Tvec>(
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

                auto result_p = modif_second_derivative<Tscal, Tvec>(
                    cell_global_id,
                    delta_cell,
                    graph_iter_xp,
                    graph_iter_xm,
                    graph_iter_yp,
                    graph_iter_ym,
                    graph_iter_zp,
                    graph_iter_zm,
                    [=](u32 id) {
                        return press[id];
                    });
                sec_der_rho[cell_global_id]   = result_r;
                sec_der_press[cell_global_id] = result_p;
            });
        });
    });
    storage.sec_der_rho.set(std::move(result_rho));
    storage.sec_der_press.set(std::move(result_press));
}

template class shammodels::basegodunov::modules::ModifiedSecondDerivative<f64_3, i64_3>;
