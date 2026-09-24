// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeGetNextConsVarDust.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/NodeGetNextConsVarDust.hpp"
#include <type_traits>

namespace {
    using Direction = shammodels::basegodunov::modules::Direction;

    template<class Tvec>
    struct KernelNextConsVarDust {

        using Tscal = sham::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<u32> &sizes,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_dt_rho_old_d,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_rho_next_d,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_rhov_old_d,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_dt_rhov_old_d,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_phi_g_old,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_phi_g_next,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_rhov_next_d,
            const f64 dt_over_2,
            u32 block_size,
            u32 ndust)

        {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size * ndust;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{
                    spans_dt_rho_old_d,
                    spans_rho_next_d,
                    spans_rhov_old_d,
                    spans_dt_rhov_old_d,
                    spans_phi_g_old,
                    spans_phi_g_next,
                },
                sham::DDMultiRef{spans_rhov_next_d},
                cell_counts,
                [dt_over_2, ndust](
                    u32 i,
                    const Tscal *__restrict dt_rho_old_d,
                    const Tscal *__restrict rho_next_d,
                    const Tvec *__restrict rhov_old_d,
                    const Tvec *__restrict dt_rhov_old_d,
                    const Tvec *__restrict phi_g_old,
                    const Tvec *__restrict phi_g_next,
                    Tvec *__restrict rhov_new_d) {
                    // for(u32 idust = 0; idust < ndust ; idust++){
                    //     auto rho_old_d = rho_next_d[i * ndust + idust] - (2. * dt_over_2) *
                    //     dt_rho_old_d[i * ndust + idust]; rhov_new_d[i * ndust + idust] =
                    //     rhov_old_d[i * ndust + idust] + (2. * dt_over_2) * dt_rhov_old_d[i *
                    //     ndust + idust]
                    //       + dt_over_2 * (rho_old_d * phi_g_old[i] + rho_next_d[i * ndust + idust]
                    //       * phi_g_next[i]);
                    // }

                    auto rho_old_d = rho_next_d[i] - (2. * dt_over_2) * dt_rho_old_d[i];
                    u32 cell_id    = i / ndust;
                    rhov_new_d[i]  = rhov_old_d[i] + (2. * dt_over_2) * dt_rhov_old_d[i]
                                     + dt_over_2
                                           * (rho_old_d * phi_g_old[cell_id]
                                              + rho_next_d[i] * phi_g_next[cell_id]);
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {
    template<class Tvec>
    void NodeGetNextConsVarDust<Tvec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        {
            edges.spans_dt_rho_old_d.check_sizes(edges.sizes.indexes);
            edges.spans_rho_next_d.check_sizes(edges.sizes.indexes);
            edges.spans_rhov_old_d.check_sizes(edges.sizes.indexes);
            edges.spans_dt_rhov_old_d.check_sizes(edges.sizes.indexes);

            edges.spans_phi_g_old.check_sizes(edges.sizes.indexes);
            edges.spans_phi_g_next.check_sizes(edges.sizes.indexes);

            edges.spans_rhov_next_d.ensure_sizes(edges.sizes.indexes);

            KernelNextConsVarDust<Tvec>::kernel(
                edges.sizes.indexes,
                edges.spans_dt_rho_old_d.get_spans(),
                edges.spans_rho_next_d.get_spans(),
                edges.spans_rhov_old_d.get_spans(),
                edges.spans_dt_rhov_old_d.get_spans(),
                edges.spans_phi_g_old.get_spans(),
                edges.spans_phi_g_next.get_spans(),
                edges.spans_rhov_next_d.get_spans(),
                edges.dt_over2.value,
                block_size,
                ndust);
        }
    }
} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeGetNextConsVarDust<f64_3>;
