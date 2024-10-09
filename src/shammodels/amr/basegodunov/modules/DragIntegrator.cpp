// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DragIntegrator.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */
#include "shammodels/amr/basegodunov/modules/DragIntegrator.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::involve_with_no_src(Tscal dt){

   StackEntry stack_lock{};

   using namespace shamrock::patch;
   using namespace shamrock;
   using namespace shammath;
   using MergedPDat = shamrock::MergedPatchData;
   const u32 ndust  = solver_config.dust_config.ndust;

   shamrock::SchedulerUtility utility(scheduler);
   shamrock::ComputeField<Tscal> cfield_rho_next_bf_drag   = utility.make_compute_field<Tscal>("rho_next_bf_drag", AMRBlock::block_size);
   shamrock::ComputeField<Tvec>  cfield_rhov_next_bf_drag  = utility.make_compute_field<Tvec>("rhov_next_bf_drag", AMRBlock::block_size);
   shamrock::ComputeField<Tscal> cfield_rhoe_next_bf_drag  = utility.make_compute_field<Tscal>("rhoe_next_bf_drag" AMRBlock::block_size);
   shamrock::ComputeField<Tscal> cfield_rho_d_next_bf_drag   = utility.make_compute_field<Tscal>("rho_d_next_bf_drag", ndust * AMRBlock::block_size);
   shamrock::ComputeField<Tvec>  cfield_rhov_d_next_bf_drag  = utility.make_compute_field<Tvec>("rhov_d_next_bf_drag", ndust * AMRBlock::block_size);


   shamrock::ComputeField<Tscal> &cfield_dtrho   = storage.dtrho.get();
   shamrock::ComputeField<Tvec>  &cfield_dtrhov  = storage.dtrhov.get();
   shamrock::ComputeField<Tscal> &cfield_dtrhoe  = storage.dtrhoe.get();
   shamrock::ComputeField<Tscal> &cfield_dtrho_d = storage.dtrho_dust.get();
   shamrock::ComputeField<Tvec> &cfield_dtrhov_d = storage.dtrhov_dust.get();



//    // load layout info
//    PatchDataLayout &pdl  = scheduler().pdl;

//    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
//    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
//    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
//    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
//    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");
//    const u32 irho_d    

    scheduler().for_each_patchdata_nonempty([&,dt] (const shamrock::patch::Patch p, shamrock::patch::PatchData &pdat) {
        logger::debug_ln("[AMR evolve time step before drag ]", "evolve field with no drag patch", p.id_patch);

        sycl::queue &q          = shamsys::instance::get_compute_queue();
        u32 id                  = p.id_patch;

        sycl::buffer<Tscal> &dt_rho_patch    = cfield_dtrho.get_buf_check(id);
        sycl::buffer<Tvec>  &dt_rhov_patch   = cfield_dtrhov.get_buf_check(id);
        sycl::buffer<Tscal> &dt_rhoe_patch   = cfield_dtrhoe.get_buf_check(id);
        sycl::buffer<Tscal> &dt_rho_d_patch  = cfield_dtrho_d_.get_buf_check(id);
        sycl::buffer<Tvec>  &dt_rhov_d_patch = cfield_dtrhov_d_.get_buf_check(id);

        sycl::buffer<Tscal> &rho_patch    = cfield_rho_next_bf_drag.get_buf_check(id);
        sycl::buffer<Tvec>  &rhov_patch   = cfield_rhov_next_bf_drag.get_buf_check(id);
        sycl::buffer<Tscal> &rhoe_patch   = cfield_rhoe_next_bf_drag.get_buf_check(id);
        sycl::buffer<Tscal> &rho_d_patch  = cfield_rho_d_next_bf_drag.get_buf_check(id);
        sycl::buffer<Tvec>  &rhov_d_patch = cfield_rhov_d_next_bf_drag.get_buf_check(id);

        u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

        q.submit([&, dt](sycl::handler &cgh){
            sycl::accessor acc_dt_rho_patch {dt_rho_patch, cgh, sycl::read_only};
            sycl::accessor acc_dt_rhov_patch {dt_rhov_patch, cgh, sycl::read_only};
            sycl::accessor acc_dt_rhoe_patch {dt_rhoe_patch, cgh, sycl::read_only};

            sycl::accessor acc_rho  {rho_patch, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_rhov {rhov_patch, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_rhoe {rhoe_patch, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, cell_count, "evolve field no drag", [=](u32 id_a) {
                acc_rho[id_a] +=  dt * acc_dt_rho_patch[id_a];
                acc_rhov[id_a] += dt * acc_dt_rhov_patch[id_a];
                acc_rhoe[id_a] += dt * acc_dt_rhoe_patch[id_a];

            });
        });

        q.submit([&, dt, ndust](sycl::handler &cgh){
            sycl::accessor acc_dt_rho_d_patch {dt_rho_d_patch, cgh, sycl::read_only};
            sycl::accessor acc_dt_rhov_d_patch {dt_rhov_d_patch, cgh, sycl::read_only};

            sycl::accessor acc_rho_d  {rho_d_patch, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_rhov_d {rhov_d_patch, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, ndust*cell_count, "dust  evolve field no drag", [=](u32 id_a){
                acc_rho_d[id_a] = dt * acc_dt_rho_d_patch[id_a];
                acc_rhov_d[id_a] = dt * acc_dt_rhov_d_patch[id_a];
            });
        });
    });

    storage.rho_next_no_drag.set(std::move(cfield_rho_next_bf_drag));
    storage.rhov_next_no_drag.set(std::move(cfield_rhov_next_bf_drag));
    storage.rhoe_next_no_drag.set(std::move(cfield_rhoe_next_bf_drag));
    storage.rho_d_next_no_drag.set(std::move(cfield_rho_d_next_bf_drag));
    storage.rhov_d_next_no_drag.set(std::move(cfield_rhov_d_next_bf_drag));
}