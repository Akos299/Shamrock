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
 * @file BasicGhosts.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "variant"

namespace shammodels::basegodunov {

    template<class Tvec>
    struct BasicGhostHandlerConfig {
        struct Periodic {};
        struct Reflective {};
        struct Absorbing {};
        struct UserDefine {};

        using Variant = std::variant<Periodic, Reflective, Absorbing, UserDefine>;
    };

    template<class Tvec, class TgridVec>
    class BasicGhostHandler {
        using CfgClass           = BasicGhostHandlerConfig<Tvec>;
        using BCConfig           = typename CfgClass::Variant;
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Config             = SolverConfig<Tvec, TgridVec>;
        using Storage            = SolverStorage<Tvec, TgridVec, u64>;

        PatchScheduler &sched;
        BCConfig ghost_config;

        public:
        struct InterfaceBuildInfos {
            TgridVec offset;
            sycl::vec<i32, dim> periodicity_index; // rename it to translation index
            shammath::AABB<TgridVec> volume_target;
        };

        struct InterfaceIdTable {
            InterfaceBuildInfos build_infos;
            std::unique_ptr<sycl::buffer<u32>> ids_interf;
            f64 cell_count_ratio;
        };

        using GeneratorMap = shambase::DistributedDataShared<InterfaceBuildInfos>;
        BasicGhostHandler(PatchScheduler &sched, BCConfig ghost_config)
            : sched(sched), ghost_config(ghost_config) {}

        /**
         * @brief Find interfaces
         *
         * @param sptree the serial patch tree
         *
         *
         */
        auto find_interfaces(PatchScheduler &sched, SerialPatchTree<TgridVec> &sptree);

        /**
         * @brief
         */

        void build_ghost_cache();

        /**
         * @brief
         */
        shambase::DistributedDataShared<shamrock::patch::PatchData> communicate_pdat(
            shamrock::patch::PatchDataLayout &pdl,
            shambase::DistributedShared<shamrock::patch::PatchData> &&interf)

            /**
             * @brief
             */

            template<class T>
            communicate_pdat_field(shambase::DistributedDataShared<PatchDataField<T>> &&interf);

        /**
         * @brief
         */

        template<class T>
        shamrock::ComputeField<T> exchange_compute_field(shamrock::ComputeField<T> &in);

        template<class T>
        shambase::DistributedDataShared<T> build_interface_native(
            std::function<T(u64, u64, InterfaceBuildInfos, sycl::buffer<u32> &, u32)> fct);
    };

} // namespace shammodels::basegodunov
