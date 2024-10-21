// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AMRGridRefinementHandler.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/amr/basegodunov/modules/AMRGridRefinementHandler.hpp"
#include "shammodels/amr/basegodunov/modules/AMRSortBlocks.hpp"

template<class T, class Tvec, class ACCField>
inline T get_pseudo_grad(
    const cell_global_id,
    const shambase::VecComponent<Tvec> delta_cell,
    const AMRGraphLinkiterator &graph_iter_xp,
    const AMRGraphLinkiterator &graph_iter_xm,
    const AMRGraphLinkiterator &graph_iter_yp,
    const AMRGraphLinkiterator &graph_iter_ym,
    const AMRGraphLinkiterator &graph_iter_zp,
    const AMRGraphLinkiterator &graph_iter_zm,
    ACCField &&field_access) {

    auto grad_scalar = [&](Tscal u_curr, Tscal u_neigh) {
        Tscal max = g_sycl_max(g_sycl_abs(u_curr), g_sycl_abs(u_neigh));
        max       = g_sycl_abs(u_curr - u_neigh) / max;
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
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
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

                get_pseudo_grad[cell_global_id] = result;
            });
        });
    });

    storage.pseudo_gradient_rho.set(std::move(result));
}

template<class Tvec, class TgridVec>
template<class UserAcc, class... T>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    gen_refine_block_changes(
        shambase::DistributedData<OptIndexList> &refine_list,
        shambase::DistributedData<OptIndexList> &derefine_list,
        T &&...args) {

    using namespace shamrock::patch;

    u64 tot_refine   = 0;
    u64 tot_derefine = 0;

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        sycl::queue &q = shamsys::instance::get_compute_queue();

        u64 id_patch = cur_p.id_patch;

        // create the refine and derefine flags buffers
        u32 obj_cnt = pdat.get_obj_cnt();

        sycl::buffer<u32> refine_flags(obj_cnt);
        sycl::buffer<u32> derefine_flags(obj_cnt);

        // fill in the flags
        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor refine_acc{refine_flags, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor derefine_acc{derefine_flags, cgh, sycl::write_only, sycl::no_init};

            UserAcc uacc(cgh, id_patch, cur_p, pdat, args...);

            cgh.parallel_for(sycl::range<1>(obj_cnt), [=](sycl::item<1> gid) {
                bool flag_refine   = false;
                bool flag_derefine = false;
                uacc.refine_criterion(gid.get_linear_id(), uacc, flag_refine, flag_derefine);

                // This is just a safe guard to avoid this nonsensicall case
                if (flag_refine && flag_derefine) {
                    flag_derefine = false;
                }

                refine_acc[gid]   = (flag_refine) ? 1 : 0;
                derefine_acc[gid] = (flag_derefine) ? 1 : 0;
            });
        });

        // keep only derefine flags on only if the eight cells want to merge and if they can
        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor acc_min{*pdat.get_field<TgridVec>(0).get_buf(), cgh, sycl::read_only};
            sycl::accessor acc_max{*pdat.get_field<TgridVec>(1).get_buf(), cgh, sycl::read_only};

            sycl::accessor acc_merge_flag{derefine_flags, cgh, sycl::read_write};

            cgh.parallel_for(sycl::range<1>(obj_cnt), [=](sycl::item<1> gid) {
                u32 id = gid.get_linear_id();

                std::array<BlockCoord, split_count> blocks;

                bool all_want_to_merge = true;
                for (u32 lid = 0; lid < split_count; lid++) {
                    blocks[lid]       = BlockCoord{acc_min[gid + lid], acc_max[gid + lid]};
                    all_want_to_merge = all_want_to_merge && acc_merge_flag[gid + lid];
                }

                acc_merge_flag[gid] = all_want_to_merge && BlockCoord::are_mergeable(blocks);
            });
        });

        ////////////////////////////////////////////////////////////////////////////////
        // refinement
        ////////////////////////////////////////////////////////////////////////////////

        // perform stream compactions on the refinement flags
        auto [buf_refine, len_refine] = shamalgs::numeric::stream_compact(q, refine_flags, obj_cnt);

        logger::debug_ln("AMRGrid", "patch ", id_patch, "refine block count = ", len_refine);

        tot_refine += len_refine;

        // add the results to the map
        refine_list.add_obj(id_patch, OptIndexList{std::move(buf_refine), len_refine});

        ////////////////////////////////////////////////////////////////////////////////
        // derefinement
        ////////////////////////////////////////////////////////////////////////////////

        // perform stream compactions on the derefinement flags
        auto [buf_derefine, len_derefine]
            = shamalgs::numeric::stream_compact(q, derefine_flags, obj_cnt);

        logger::debug_ln("AMRGrid", "patch ", id_patch, "merge block count = ", len_derefine);

        tot_derefine += len_derefine;

        // add the results to the map
        derefine_list.add_obj(id_patch, OptIndexList{std::move(buf_derefine), len_derefine});
    });

    logger::info_ln("AMRGrid", "on this process", tot_refine, "blocks were refined");
    logger::info_ln(
        "AMRGrid", "on this process", tot_derefine * split_count, "blocks were derefined");
}
template<class Tvec, class TgridVec>
template<class UserAcc>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    internal_refine_grid(shambase::DistributedData<OptIndexList> &&refine_list) {

    using namespace shamrock::patch;

    u64 sum_block_count = 0;

    scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
        sycl::queue &q = shamsys::instance::get_compute_queue();

        u32 old_obj_cnt = pdat.get_obj_cnt();

        OptIndexList &refine_flags = refine_list.get(id_patch);

        if (refine_flags.count > 0) {

            // alloc memory for the new blocks to be created
            pdat.expand(refine_flags.count * (split_count - 1));

            // Refine the block (set the positions) and fill the corresponding fields
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor index_to_ref{*refine_flags.idx, cgh, sycl::read_only};

                sycl::accessor block_bound_low{
                    *pdat.get_field<TgridVec>(0).get_buf(), cgh, sycl::read_write};
                sycl::accessor block_bound_high{
                    *pdat.get_field<TgridVec>(1).get_buf(), cgh, sycl::read_write};

                u32 start_index_push = old_obj_cnt;

                constexpr u32 new_splits = split_count - 1;

                UserAcc uacc(cgh, pdat);

                cgh.parallel_for(sycl::range<1>(refine_flags.count), [=](sycl::item<1> gid) {
                    u32 tid = gid.get_linear_id();

                    u32 idx_to_refine = index_to_ref[gid];

                    // gen splits coordinates
                    BlockCoord cur_block{
                        block_bound_low[idx_to_refine], block_bound_high[idx_to_refine]};

                    std::array<BlockCoord, split_count> block_coords
                        = BlockCoord::get_split(cur_block.bmin, cur_block.bmax);

                    // generate index for the refined blocks
                    std::array<u32, split_count> blocks_ids;
                    blocks_ids[0] = idx_to_refine;

                    // generate index for the new blocks (the current index is reused for the first
                    // new block, the others are pushed at the end of the patchdata)
#pragma unroll
                    for (u32 pid = 0; pid < new_splits; pid++) {
                        blocks_ids[pid + 1] = start_index_push + tid * new_splits + pid;
                    }

                    // write coordinates

#pragma unroll
                    for (u32 pid = 0; pid < split_count; pid++) {
                        block_bound_low[blocks_ids[pid]]  = block_coords[pid].bmin;
                        block_bound_high[blocks_ids[pid]] = block_coords[pid].bmax;
                    }

                    // user lambda to fill the fields
                    uacc.apply_refine(idx_to_refine, cur_block, blocks_ids, block_coords, uacc);
                });
            });
        }

        sum_block_count += pdat.get_obj_cnt();
    });

    logger::info_ln("AMRGrid", "process block count =", sum_block_count);
}

template<class Tvec, class TgridVec>
template<class UserAcc>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    internal_derefine_grid(shambase::DistributedData<OptIndexList> &&derefine_list) {

    using namespace shamrock::patch;

    scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
        sycl::queue &q = shamsys::instance::get_compute_queue();

        u32 old_obj_cnt = pdat.get_obj_cnt();

        OptIndexList &derefine_flags = derefine_list.get(id_patch);

        if (derefine_flags.count > 0) {

            // init flag table
            sycl::buffer<u32> keep_block_flag
                = shamalgs::algorithm::gen_buffer_device(q, old_obj_cnt, [](u32 i) -> u32 {
                      return 1;
                  });

            // edit block content + make flag of blocks to keep
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor index_to_deref{*derefine_flags.idx, cgh, sycl::read_only};

                sycl::accessor block_bound_low{
                    *pdat.get_field<TgridVec>(0).get_buf(), cgh, sycl::read_write};
                sycl::accessor block_bound_high{
                    *pdat.get_field<TgridVec>(1).get_buf(), cgh, sycl::read_write};

                sycl::accessor flag_keep{keep_block_flag, cgh, sycl::read_write};

                UserAcc uacc(cgh, pdat);

                cgh.parallel_for(sycl::range<1>(derefine_flags.count), [=](sycl::item<1> gid) {
                    u32 tid = gid.get_linear_id();

                    u32 idx_to_derefine = index_to_deref[gid];

                    // compute old block indexes
                    std::array<u32, split_count> old_indexes;
#pragma unroll
                    for (u32 pid = 0; pid < split_count; pid++) {
                        old_indexes[pid] = idx_to_derefine + pid;
                    }

                    // load block coords
                    std::array<BlockCoord, split_count> block_coords;
#pragma unroll
                    for (u32 pid = 0; pid < split_count; pid++) {
                        block_coords[pid] = BlockCoord{
                            block_bound_low[old_indexes[pid]], block_bound_high[old_indexes[pid]]};
                    }

                    // make new block coord
                    BlockCoord merged_block_coord = BlockCoord::get_merge(block_coords);

                    // write new coord
                    block_bound_low[idx_to_derefine]  = merged_block_coord.bmin;
                    block_bound_high[idx_to_derefine] = merged_block_coord.bmax;

// flag the old blocks for removal
#pragma unroll
                    for (u32 pid = 1; pid < split_count; pid++) {
                        flag_keep[idx_to_derefine + pid] = 0;
                    }

                    // user lambda to fill the fields
                    uacc.apply_derefine(
                        old_indexes, block_coords, idx_to_derefine, merged_block_coord, uacc);
                });
            });

            // stream compact the flags
            auto [opt_buf, len] = shamalgs::numeric::stream_compact(
                shamsys::instance::get_compute_queue(), keep_block_flag, old_obj_cnt);

            logger::debug_ln(
                "AMR Grid", "patch", id_patch, "derefine block count ", old_obj_cnt, "->", len);

            if (!opt_buf) {
                throw std::runtime_error("opt buf must contain something at this point");
            }

            // remap pdat according to stream compact
            pdat.index_remap_resize(*opt_buf, len);
        }
    });
}

template<class Tvec, class TgridVec>
template<class UserAccCrit, class UserAccSplit, class UserAccMerge>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    internal_update_refinement() {

    // Ensure that the blocks are sorted before refinement
    AMRSortBlocks block_sorter(context, solver_config, storage);
    block_sorter.reorder_amr_blocks();

    // get refine and derefine list
    shambase::DistributedData<OptIndexList> refine_list;
    shambase::DistributedData<OptIndexList> derefine_list;

    gen_refine_block_changes<UserAccCrit>(refine_list, derefine_list);

    //////// apply refine ////////
    // Note that this only add new blocks at the end of the patchdata
    internal_refine_grid<UserAccSplit>(std::move(refine_list));

    //////// apply derefine ////////
    // Note that this will perform the merge then remove the old blocks
    // This is ok to call straight after the refine without edditing the index list in derefine_list
    // since no permutations were applied in internal_refine_grid and no cells can be both refined
    // and derefined in the same pass
    internal_derefine_grid<UserAccMerge>(std::move(derefine_list));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    update_refinement() {

    class RefineCritBlock {
        public:
        sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device> block_low_bound;
        sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device>
            block_high_bound;
        sycl::accessor<Tscal, 1, sycl::access::mode::read, sycl::target::device>
            block_density_field;

        Tscal one_over_Nside = 1. / AMRBlock::Nside;

        Tscal dxfact;
        Tscal wanted_mass;

        RefineCritBlock(
            sycl::handler &cgh,
            u64 id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchData &pdat,
            Tscal dxfact,
            Tscal wanted_mass)
            : block_low_bound{*pdat.get_field<TgridVec>(0).get_buf(), cgh, sycl::read_only},
              block_high_bound{*pdat.get_field<TgridVec>(1).get_buf(), cgh, sycl::read_only},
              block_density_field{
                  *pdat.get_field<Tscal>(pdat.pdl.get_field_idx<Tscal>("rho")).get_buf(),
                  cgh,
                  sycl::read_only},
              dxfact(dxfact), wanted_mass(wanted_mass) {}

        void refine_criterion(
            u32 block_id, RefineCritBlock acc, bool &should_refine, bool &should_derefine) const {

            TgridVec low_bound  = acc.block_low_bound[block_id];
            TgridVec high_bound = acc.block_high_bound[block_id];

            Tvec lower_flt = low_bound.template convert<Tscal>() * dxfact;
            Tvec upper_flt = high_bound.template convert<Tscal>() * dxfact;

            Tvec block_cell_size = (upper_flt - lower_flt) * one_over_Nside;

            Tscal sum_mass = 0;
            for (u32 i = 0; i < AMRBlock::block_size; i++) {
                sum_mass += acc.block_density_field[i + block_id * AMRBlock::block_size];
            }
            sum_mass *= block_cell_size.x() * block_cell_size.y() * block_cell_size.z();

            if (sum_mass > wanted_mass * 8) {
                should_refine   = true;
                should_derefine = false;
            } else if (sum_mass < wanted_mass) {
                should_refine   = false;
                should_derefine = true;
            } else {
                should_refine   = false;
                should_derefine = false;
            }

            should_refine = should_refine && (high_bound.x() - low_bound.x() > AMRBlock::Nside);
            should_refine = should_refine && (high_bound.y() - low_bound.y() > AMRBlock::Nside);
            should_refine = should_refine && (high_bound.z() - low_bound.z() > AMRBlock::Nside);
        }
    };

    class RefinePseudoGradientBlock {
        public:
        sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device> block_low_bound;
        sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device>
            block_high_bound;
        sycl::accessor<Tscal, 1, sycl::access::mode::read, sycl::target::device>
            block_field_gradient; // For now we use dust density as field

        Tscal one_over_Nside = 1. / AMRBlock::Nside;
        Tscal dxfact;
        Tscal error_min, error_max;

        RefinePseudoGradientBlock(
            sycl::handler &cgh,
            u64 id_patch,
            shamrock::patch::Patch p,
            Tscal dxfact,
            Tscal error_min,
            Tscal error_max)
            : block_low_bound{*pdat.get_field<TgridVec>(0).get_buf(), cgh, sycl::read_only},
              block_high_bound{*pdat.get_field<TgridVec>(1).get_buf(), cgh, sycl::read_only},
              block_field_gradient{
                  storage.pseudo_gradient_rho.get().get_buf_check(id_patch), cgh, sycl::read_only},
              dxfact(dxfact), error_min(error_min), error_max(error_max) {}

        void refine_criterion(
            u32 block_id, RefineCritBlock acc, bool &should_refine, bool &should_derefine) const {

            TgridVec low_bound  = acc.block_low_bound[block_id];
            TgridVec high_bound = acc.block_high_bound[block_id];

            Tvec lower_flt = low_bound.template convert<Tscal>() * dxfact;
            Tvec upper_flt = high_bound.template convert<Tscal>() * dxfact;

            Tvec block_cell_size = (upper_flt - lower_flt) * one_over_Nside;

            Tscal diff_factor = 0;
            for (u32 i = 0; i < AMRBlock::block_size; i++) {
                // diff_factor += acc.block_field_gradient[i + block_id * AMRBlock::block_size];
                diff_factor = g_sycl_max(
                    acc.block_field_gradient[i + block_id * AMRBlock::block_size],
                    diff_factor); // pensez aux max notamment sur le
                                  // block_base
            }

            if (diff_factor > error_max) {
                should_refine   = true;
                should_derefine = false;
            } else if (diff_factor <= error_min) {
                should_refine   = false;
                should_derefine = true;
            } else {
                should_refine   = false;
                should_derefine = false;
            }

            should_refine = should_refine && (high_bound.x() - low_bound.x() > AMRBlock::Nside);
            should_refine = should_refine && (high_bound.y() - low_bound.y() > AMRBlock::Nside);
            should_refine = should_refine && (high_bound.z() - low_bound.z() > AMRBlock::Nside);
        }
    };

    class RefineCellAccessor {
        public:
        sycl::accessor<f64, 1, sycl::access::mode::read_write, sycl::target::device> rho;
        sycl::accessor<f64_3, 1, sycl::access::mode::read_write, sycl::target::device> rho_vel;
        sycl::accessor<f64, 1, sycl::access::mode::read_write, sycl::target::device> rhoE;

        RefineCellAccessor(sycl::handler &cgh, shamrock::patch::PatchData &pdat)
            : rho{*pdat.get_field<f64>(2).get_buf(), cgh, sycl::read_write},
              rho_vel{*pdat.get_field<f64_3>(3).get_buf(), cgh, sycl::read_write},
              rhoE{*pdat.get_field<f64>(4).get_buf(), cgh, sycl::read_write} {}

        void apply_refine(
            u32 cur_idx,
            BlockCoord cur_coords,
            std::array<u32, 8> new_blocks,
            std::array<BlockCoord, 8> new_block_coords,
            RefineCellAccessor acc) const {

            auto get_coord_ref = [](u32 i) -> std::array<u32, dim> {
                constexpr u32 NsideBlockPow = 1;
                constexpr u32 Nside         = 1U << NsideBlockPow;

                if constexpr (dim == 3) {
                    const u32 tmp = i >> NsideBlockPow;
                    return {i % Nside, (tmp) % Nside, (tmp) >> NsideBlockPow};
                }
            };

            auto get_index_block = [](std::array<u32, dim> coord) -> u32 {
                constexpr u32 NsideBlockPow = 1;
                constexpr u32 Nside         = 1U << NsideBlockPow;

                if constexpr (dim == 3) {
                    return coord[0] + Nside * coord[1] + Nside * Nside * coord[2];
                }
            };

            auto get_gid_write = [&](std::array<u32, dim> &glid) -> u32 {
                std::array<u32, dim> bid
                    = {glid[0] >> AMRBlock::NsideBlockPow,
                       glid[1] >> AMRBlock::NsideBlockPow,
                       glid[2] >> AMRBlock::NsideBlockPow};

                // logger::raw_ln(glid,bid);
                return new_blocks[get_index_block(bid)] * AMRBlock::block_size
                       + AMRBlock::get_index(
                           {glid[0] % AMRBlock::Nside,
                            glid[1] % AMRBlock::Nside,
                            glid[2] % AMRBlock::Nside});
            };

            std::array<f64, AMRBlock::block_size> old_rho_block;
            std::array<f64_3, AMRBlock::block_size> old_rho_vel_block;
            std::array<f64, AMRBlock::block_size> old_rhoE_block;

            // save old block
            for (u32 loc_id = 0; loc_id < AMRBlock::block_size; loc_id++) {

                auto [lx, ly, lz]         = get_coord_ref(loc_id);
                u32 old_cell_idx          = cur_idx * AMRBlock::block_size + loc_id;
                old_rho_block[loc_id]     = acc.rho[old_cell_idx];
                old_rho_vel_block[loc_id] = acc.rho_vel[old_cell_idx];
                old_rhoE_block[loc_id]    = acc.rhoE[old_cell_idx];
            }

            for (u32 loc_id = 0; loc_id < AMRBlock::block_size; loc_id++) {

                auto [lx, ly, lz] = get_coord_ref(loc_id);
                u32 old_cell_idx  = cur_idx * AMRBlock::block_size + loc_id;

                Tscal rho_block    = old_rho_block[loc_id];
                Tvec rho_vel_block = old_rho_vel_block[loc_id];
                Tscal rhoE_block   = old_rhoE_block[loc_id];
                for (u32 subdiv_lid = 0; subdiv_lid < 8; subdiv_lid++) {

                    auto [sx, sy, sz] = get_coord_ref(subdiv_lid);

                    std::array<u32, 3> glid = {lx * 2 + sx, ly * 2 + sy, lz * 2 + sz};

                    u32 new_cell_idx = get_gid_write(glid);
                    /*
                                        if (1627 == cur_idx) {
                                            logger::raw_ln(
                                                cur_idx,
                                                "set cell ",
                                                new_cell_idx,
                                                " from cell",
                                                old_cell_idx,
                                                "old",
                                                rho_block,
                                                rho_vel_block,
                                                rhoE_block);
                                        }
                                        */
                    acc.rho[new_cell_idx]     = rho_block;
                    acc.rho_vel[new_cell_idx] = rho_vel_block;
                    acc.rhoE[new_cell_idx]    = rhoE_block;
                }
            }
        }

        void apply_derefine(
            std::array<u32, 8> old_blocks,
            std::array<BlockCoord, 8> old_coords,
            u32 new_cell,
            BlockCoord new_coord,

            RefineCellAccessor acc) const {

            std::array<f64, AMRBlock::block_size> rho_block;
            std::array<f64_3, AMRBlock::block_size> rho_vel_block;
            std::array<f64, AMRBlock::block_size> rhoE_block;

            for (u32 cell_id = 0; cell_id < AMRBlock::block_size; cell_id++) {
                rho_block[cell_id]     = {};
                rho_vel_block[cell_id] = {};
                rhoE_block[cell_id]    = {};
            }

            for (u32 pid = 0; pid < 8; pid++) {
                for (u32 cell_id = 0; cell_id < AMRBlock::block_size; cell_id++) {
                    rho_block[cell_id] += acc.rho[old_blocks[pid] * AMRBlock::block_size + cell_id];
                    rho_vel_block[cell_id]
                        += acc.rho_vel[old_blocks[pid] * AMRBlock::block_size + cell_id];
                    rhoE_block[cell_id]
                        += acc.rhoE[old_blocks[pid] * AMRBlock::block_size + cell_id];
                }
            }

            for (u32 cell_id = 0; cell_id < AMRBlock::block_size; cell_id++) {
                rho_block[cell_id] /= 8;
                rho_vel_block[cell_id] /= 8;
                rhoE_block[cell_id] /= 8;
            }

            for (u32 cell_id = 0; cell_id < AMRBlock::block_size; cell_id++) {
                u32 newcell_idx          = new_cell * AMRBlock::block_size + cell_id;
                acc.rho[newcell_idx]     = rho_block[cell_id];
                acc.rho_vel[newcell_idx] = rho_vel_block[cell_id];
                acc.rhoE[newcell_idx]    = rhoE_block[cell_id];
            }
        }
    };

    // Ensure that the blocks are sorted before refinement
    AMRSortBlocks block_sorter(context, solver_config, storage);
    block_sorter.reorder_amr_blocks();

    using AMRmode_None           = typename AMRMode<Tvec, TgridVec>::None;
    using AMRmode_DensityBased   = typename AMRMode<Tvec, TgridVec>::DensityBased;
    using AMRmode_PseudoGradient = typename AMRMode<Tvec, TgridVec>::PseudoGradient;

    if (AMRmode_None *cfg = std::get_if<AMRmode_None>(&solver_config.amr_mode.config)) {
        // no refinment here turn around there is nothing to see
    } else if (
        AMRmode_DensityBased *cfg
        = std::get_if<AMRmode_DensityBased>(&solver_config.amr_mode.config)) {
        Tscal dxfact(solver_config.grid_coord_to_pos_fact);

        // get refine and derefine list
        shambase::DistributedData<OptIndexList> refine_list;
        shambase::DistributedData<OptIndexList> derefine_list;

        gen_refine_block_changes<RefineCritBlock>(
            refine_list, derefine_list, dxfact, cfg->crit_mass);

        //////// apply refine ////////
        // Note that this only add new blocks at the end of the patchdata
        internal_refine_grid<RefineCellAccessor>(std::move(refine_list));

        //////// apply derefine ////////
        // Note that this will perform the merge then remove the old blocks
        // This is ok to call straight after the refine without edditing the index list in
        // derefine_list since no permutations were applied in internal_refine_grid and no cells can
        // be both refined and derefined in the same pass
        internal_derefine_grid<RefineCellAccessor>(std::move(derefine_list));
    }

    else if (
        AMRmode_PseudoGradient *cfg
        = std::get_if<AMRmode_PseudoGradient>(&solver_config.amr_mode.config)) {
        Tscal dxfact(solver_config.grid_coord_to_pos_fact);

        // get refine and derefine list
        shambase::DistributedData<OptIndexList> refine_list;
        shambase::DistributedData<OptIndexList> derefine_list;

        gen_refine_block_changes<RefinePseudoGradientBlock>(
            refine_list, derefine_list, dxfact, cfg->error_min, cfg->error_max);

        internal_refine_grid<RefineCellAccessor>(std::move(refine_list));

        internal_derefine_grid<RefineCellAccessor>(std::move(derefine_list));
    }

    else {
        shambase::throw_unimplented();
    }
}

template class shammodels::basegodunov::modules::AMRGridRefinementHandler<f64_3, i64_3>;
