// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BasicGhosts.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamalgs/numeric.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammath/AABB.hpp"
#include "shammath/CoordRange.hpp"
#include "shammodels/ramses/BasicGhosts.hpp"
#include "shammodels/ramses/modules/BasicGhosts.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, class TgridVec, shammodels::basegodunov::modules::GhostDir dir>

auto shammodels::basegodunov::modules::BasicGhostHandler<Tvec, TgridVec>::find_interfaces_dir<
    GhostDir dir>(PatchScheduler &sched, SerialPatchTree<TgridVec> &sptree) {
    StackEntry stack_loc{};
    using namespace shamrock::patch;
    using namespace shammath;
    using CfgClass     = basedgodunov::BasicGhostHandlerConfig<Tvec>;
    using BCConfig     = typename CfgClass::Variant;
    using BCPeriodic   = typename CfgClass::Periodic;
    using BCReflective = typename CfgClass::Reflective;
    using BCAbsorbing  = typename CfgClass::Absorbing;

    i32 repetition_x = 1;
    i32 repetition_y = 1;
    i32 repetition_z = 1;

    GeneratorMap results;

    shamrock::patch::SimulationBox &sim_box = sched.get_sim_box();

    PatchCoordTransform<TgridVec> patch_coord_transd = sim_box.get_patch_transform<TgridVec>();
    TgridVec bsize                                   = sim_box.get_bounding_box_size<TgridVec>();

    if (BCPeriodic *cfg = std::get_if<BCPeriodic>(&ghost_config)) {

        if constexpr (dir == Direction::xp) {
            i32 xoff = repetition_x;
            for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
                for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {

                    TgridVec _offset
                        = TgridVec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                    /** May be get a big buffer for all translate patches for all direction and
                     * boundary condition and just perform the search on it */

                    sched.for_each_local_patch([&](const Patch psender) {
                        CoordRange<TgridVec> sender_bsize
                            = patch_coord_transf.to_obj_coord(psender);
                        CoordRange<TgridVec> sender_bsize_off = sender_bsize.add_offset(_offset);

                        shammath::AABB<TgridVec> sender_bsize_off_aabb{
                            sender_bsize_off.lower, sender_bsize_off.upper};

                        using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

                        sptree.host_for_each_leafs(
                            [&](u64 tree_id, PtNode n) {
                                shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};
                                bool result
                                    = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();
                                return result;
                            },

                            [&](u64 id_found, PtNode n) {
                                if ((id_found == psender.id_patch) && (xoff == 0) && (yoff = 0)
                                    && (zoff = 0)) {
                                    return;
                                }

                                InterfaceBuildInfos ret{
                                    _offset,
                                    {xoff, yoff, zoff},
                                    shammath::AABB<TgridVec>{
                                        n.box_min - _offset, n.box_max - _offset}};

                                results.add_obj(psender.id_patch, id_found, std::move(ret));
                            });
                    });
                }
            }
        }

        if constexpr (dir == Direction::xm) {
            i32 xoff = -repetition_x;
            for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
                for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {

                    TgridVec _offset
                        = TgridVec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                    /** May be get a big buffer for all translate patches for all direction and
                     * boundary condition and just perform the search on it */

                    sched.for_each_local_patch([&](const Patch psender) {
                        CoordRange<TgridVec> sender_bsize
                            = patch_coord_transf.to_obj_coord(psender);
                        CoordRange<TgridVec> sender_bsize_off = sender_bsize.add_offset(_offset);

                        shammath::AABB<TgridVec> sender_bsize_off_aabb{
                            sender_bsize_off.lower, sender_bsize_off.upper};

                        using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

                        sptree.host_for_each_leafs(
                            [&](u64 tree_id, PtNode n) {
                                shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};
                                bool result
                                    = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();
                                return result;
                            },

                            [&](u64 id_found, PtNode n) {
                                if ((id_found == psender.id_patch) && (xoff == 0) && (yoff = 0)
                                    && (zoff = 0)) {
                                    return;
                                }

                                InterfaceBuildInfos ret{
                                    _offset,
                                    {xoff, yoff, zoff},
                                    shammath::AABB<TgridVec>{
                                        n.box_min - _offset, n.box_max - _offset}};

                                results.add_obj(psender.id_patch, id_found, std::move(ret));
                            });
                    });
                }
            }
        }

        if constexpr (dir == Direction::yp) {
            i32 yoff = repetition_y;
            for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
                for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {

                    TgridVec _offset
                        = TgridVec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                    /** May be get a big buffer for all translate patches for all direction and
                     * boundary condition and just perform the search on it */

                    sched.for_each_local_patch([&](const Patch psender) {
                        CoordRange<TgridVec> sender_bsize
                            = patch_coord_transf.to_obj_coord(psender);
                        CoordRange<TgridVec> sender_bsize_off = sender_bsize.add_offset(_offset);

                        shammath::AABB<TgridVec> sender_bsize_off_aabb{
                            sender_bsize_off.lower, sender_bsize_off.upper};

                        using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

                        sptree.host_for_each_leafs(
                            [&](u64 tree_id, PtNode n) {
                                shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};
                                bool result
                                    = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();
                                return result;
                            },

                            [&](u64 id_found, PtNode n) {
                                if ((id_found == psender.id_patch) && (xoff == 0) && (yoff = 0)
                                    && (zoff = 0)) {
                                    return;
                                }

                                InterfaceBuildInfos ret{
                                    _offset,
                                    {xoff, yoff, zoff},
                                    shammath::AABB<TgridVec>{
                                        n.box_min - _offset, n.box_max - _offset}};

                                results.add_obj(psender.id_patch, id_found, std::move(ret));
                            });
                    });
                }
            }
        }

        if constexpr (dir == Direction::ym) {
            i32 yoff = -repetition_y;
            for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
                for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {

                    TgridVec _offset
                        = TgridVec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                    /** May be get a big buffer for all translate patches for all direction and
                     * boundary condition and just perform the search on it */

                    sched.for_each_local_patch([&](const Patch psender) {
                        CoordRange<TgridVec> sender_bsize
                            = patch_coord_transf.to_obj_coord(psender);
                        CoordRange<TgridVec> sender_bsize_off = sender_bsize.add_offset(_offset);

                        shammath::AABB<TgridVec> sender_bsize_off_aabb{
                            sender_bsize_off.lower, sender_bsize_off.upper};

                        using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

                        sptree.host_for_each_leafs(
                            [&](u64 tree_id, PtNode n) {
                                shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};
                                bool result
                                    = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();
                                return result;
                            },

                            [&](u64 id_found, PtNode n) {
                                if ((id_found == psender.id_patch) && (xoff == 0) && (yoff = 0)
                                    && (zoff = 0)) {
                                    return;
                                }

                                InterfaceBuildInfos ret{
                                    _offset,
                                    {xoff, yoff, zoff},
                                    shammath::AABB<TgridVec>{
                                        n.box_min - _offset, n.box_max - _offset}};

                                results.add_obj(psender.id_patch, id_found, std::move(ret));
                            });
                    });
                }
            }
        }

        if constexpr (dir == Direction::zp) {
            i32 zoff = repetition_z;
            for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
                for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
                    TgridVec _offset
                        = TgridVec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                    /** May be get a big buffer for all translate patches for all direction and
                     * boundary condition and just perform the search on it */

                    sched.for_each_local_patch([&](const Patch psender) {
                        CoordRange<TgridVec> sender_bsize
                            = patch_coord_transf.to_obj_coord(psender);
                        CoordRange<TgridVec> sender_bsize_off = sender_bsize.add_offset(_offset);

                        shammath::AABB<TgridVec> sender_bsize_off_aabb{
                            sender_bsize_off.lower, sender_bsize_off.upper};

                        using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

                        sptree.host_for_each_leafs(
                            [&](u64 tree_id, PtNode n) {
                                shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};
                                bool result
                                    = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();
                                return result;
                            },

                            [&](u64 id_found, PtNode n) {
                                if ((id_found == psender.id_patch) && (xoff == 0) && (yoff = 0)
                                    && (zoff = 0)) {
                                    return;
                                }

                                InterfaceBuildInfos ret{
                                    _offset,
                                    {xoff, yoff, zoff},
                                    shammath::AABB<TgridVec>{
                                        n.box_min - _offset, n.box_max - _offset}};

                                results.add_obj(psender.id_patch, id_found, std::move(ret));
                            });
                    });
                }
            }
        }

        if constexpr (dir == Direction::zm) {
            i32 zoff = -repetition_z;
            for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
                for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
                    TgridVec _offset
                        = TgridVec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                    /** May be get a big buffer for all translate patches for all direction and
                     * boundary condition and just perform the search on it */

                    sched.for_each_local_patch([&](const Patch psender) {
                        CoordRange<TgridVec> sender_bsize
                            = patch_coord_transf.to_obj_coord(psender);
                        CoordRange<TgridVec> sender_bsize_off = sender_bsize.add_offset(_offset);

                        shammath::AABB<TgridVec> sender_bsize_off_aabb{
                            sender_bsize_off.lower, sender_bsize_off.upper};

                        using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

                        sptree.host_for_each_leafs(
                            [&](u64 tree_id, PtNode n) {
                                shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};
                                bool result
                                    = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();
                                return result;
                            },

                            [&](u64 id_found, PtNode n) {
                                if ((id_found == psender.id_patch) && (xoff == 0) && (yoff = 0)
                                    && (zoff = 0)) {
                                    return;
                                }

                                InterfaceBuildInfos ret{
                                    _offset,
                                    {xoff, yoff, zoff},
                                    shammath::AABB<TgridVec>{
                                        n.box_min - _offset, n.box_max - _offset}};

                                results.add_obj(psender.id_patch, id_found, std::move(ret));
                            });
                    });
                }
            }
        }

    }

    else if (
        BCReflective *cfg = std::get_if<BCReflective>(&ghost_config) || BCAbsorbing *cfg
        = std::get_if<BCAbsorbing>(&ghost_config)) {
        if constexpr (dir == Direction::xp) {
            sched.for_each_local_patch([&](const Patch psender) {
                CoordRange<TgridVec> sender_bsize = patch_coord_transf.to_obj_coord(psender);

                Tvec _lower  = sender_bsize.lower;
                Tvec _upper  = sender_bsize.upper;
                Tvec _offset = {_upper[0], 0, 0};

                CoordRange<TgridVec> sender_bsize_off = sender_bsize.add_offset(_offset);
            });

            shammath::AABB<TgridVec> sender_bsize_off_aabb{
                sender_bsize_off.lower, sender_bsize_off.upper};

            using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

            sptree.host_for_each_leafs(
                [&](u64 tree_id, PtNode n) {
                    shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};
                    bool result = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();
                    return result;
                },

                [&](u64 id_found, PtNode n) {
                    if ((id_found == psender.id_patch)) {
                        return;
                    }

                    InterfaceBuildInfos ret{
                        _offset,
                        {1, 0, 0},
                        shammath::AABB<TgridVec>{n.box_min - _offset, n.box_max - _offset}};

                    results.add_obj(psender.id_patch, id_found, std::move(ret));
                });
        }

        if constexpr (dir == Direction::xm) {
            sched.for_each_local_patch([&](const Patch psender) {
                CoordRange<TgridVec> sender_bsize = patch_coord_transf.to_obj_coord(psender);

                Tvec _lower  = sender_bsize.lower;
                Tvec _upper  = sender_bsize.upper;
                Tvec _offset = {-1 * _upper[0], 0, 0};

                CoordRange<TgridVec> sender_bsize_off = sender_bsize.add_offset(_offset);
            });

            shammath::AABB<TgridVec> sender_bsize_off_aabb{
                sender_bsize_off.lower, sender_bsize_off.upper};

            using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

            sptree.host_for_each_leafs(
                [&](u64 tree_id, PtNode n) {
                    shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};
                    bool result = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();
                    return result;
                },

                [&](u64 id_found, PtNode n) {
                    if ((id_found == psender.id_patch)) {
                        return;
                    }

                    InterfaceBuildInfos ret{
                        _offset,
                        {-1, 0, 0},
                        shammath::AABB<TgridVec>{n.box_min - _offset, n.box_max - _offset}};

                    results.add_obj(psender.id_patch, id_found, std::move(ret));
                });
        }

        if constexpr (dir == Direction::yp) {
            sched.for_each_local_patch([&](const Patch psender) {
                CoordRange<TgridVec> sender_bsize = patch_coord_transf.to_obj_coord(psender);

                Tvec _lower  = sender_bsize.lower;
                Tvec _upper  = sender_bsize.upper;
                Tvec _offset = {0, _upper[1], 0};

                CoordRange<TgridVec> sender_bsize_off = sender_bsize.add_offset(_offset);
            });

            shammath::AABB<TgridVec> sender_bsize_off_aabb{
                sender_bsize_off.lower, sender_bsize_off.upper};

            using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

            sptree.host_for_each_leafs(
                [&](u64 tree_id, PtNode n) {
                    shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};
                    bool result = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();
                    return result;
                },

                [&](u64 id_found, PtNode n) {
                    if ((id_found == psender.id_patch)) {
                        return;
                    }

                    InterfaceBuildInfos ret{
                        _offset,
                        {0, 1, 0},
                        shammath::AABB<TgridVec>{n.box_min - _offset, n.box_max - _offset}};

                    results.add_obj(psender.id_patch, id_found, std::move(ret));
                });
        }

        if constexpr (dir == Direction::ym) {
            sched.for_each_local_patch([&](const Patch psender) {
                CoordRange<TgridVec> sender_bsize = patch_coord_transf.to_obj_coord(psender);

                Tvec _lower  = sender_bsize.lower;
                Tvec _upper  = sender_bsize.upper;
                Tvec _offset = {0, -1 * _upper[1], 0};

                CoordRange<TgridVec> sender_bsize_off = sender_bsize.add_offset(_offset);
            });

            shammath::AABB<TgridVec> sender_bsize_off_aabb{
                sender_bsize_off.lower, sender_bsize_off.upper};

            using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

            sptree.host_for_each_leafs(
                [&](u64 tree_id, PtNode n) {
                    shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};
                    bool result = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();
                    return result;
                },

                [&](u64 id_found, PtNode n) {
                    if ((id_found == psender.id_patch)) {
                        return;
                    }

                    InterfaceBuildInfos ret{
                        _offset,
                        {0, -1, 0},
                        shammath::AABB<TgridVec>{n.box_min - _offset, n.box_max - _offset}};

                    results.add_obj(psender.id_patch, id_found, std::move(ret));
                });
        }

        if constexpr (dir == Direction::zp) {
            sched.for_each_local_patch([&](const Patch psender) {
                CoordRange<TgridVec> sender_bsize = patch_coord_transf.to_obj_coord(psender);

                Tvec _lower  = sender_bsize.lower;
                Tvec _upper  = sender_bsize.upper;
                Tvec _offset = {0, 0, _upper[2]};

                CoordRange<TgridVec> sender_bsize_off = sender_bsize.add_offset(_offset);
            });

            shammath::AABB<TgridVec> sender_bsize_off_aabb{
                sender_bsize_off.lower, sender_bsize_off.upper};

            using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

            sptree.host_for_each_leafs(
                [&](u64 tree_id, PtNode n) {
                    shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};
                    bool result = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();
                    return result;
                },

                [&](u64 id_found, PtNode n) {
                    if ((id_found == psender.id_patch)) {
                        return;
                    }

                    InterfaceBuildInfos ret{
                        _offset,
                        {0, 0, 1},
                        shammath::AABB<TgridVec>{n.box_min - _offset, n.box_max - _offset}};

                    results.add_obj(psender.id_patch, id_found, std::move(ret));
                });
        }

        if constexpr (dir == Direction::zm) {
            sched.for_each_local_patch([&](const Patch psender) {
                CoordRange<TgridVec> sender_bsize = patch_coord_transf.to_obj_coord(psender);

                Tvec _lower  = sender_bsize.lower;
                Tvec _upper  = sender_bsize.upper;
                Tvec _offset = {0, 0, -1 * _upper[2]};

                CoordRange<TgridVec> sender_bsize_off = sender_bsize.add_offset(_offset);
            });

            shammath::AABB<TgridVec> sender_bsize_off_aabb{
                sender_bsize_off.lower, sender_bsize_off.upper};

            using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

            sptree.host_for_each_leafs(
                [&](u64 tree_id, PtNode n) {
                    shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};
                    bool result = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();
                    return result;
                },

                [&](u64 id_found, PtNode n) {
                    if ((id_found == psender.id_patch)) {
                        return;
                    }

                    InterfaceBuildInfos ret{
                        _offset,
                        {0, 0, -1},
                        shammath::AABB<TgridVec>{n.box_min - _offset, n.box_max - _offset}};

                    results.add_obj(psender.id_patch, id_found, std::move(ret));
                });
        }

    }

    else {
        shambase::throw_unimplemented();
        return;
    }

    return results;
}
