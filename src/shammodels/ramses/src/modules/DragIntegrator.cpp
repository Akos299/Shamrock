// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DragIntegrator.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */
#include "shammodels/ramses/modules/DragIntegrator.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/EventList.hpp"
#include "shammath/MatrixExponential.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/NodeInstance.hpp"
#include <sys/types.h>
#include <array>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::involve_with_no_src(
    Tscal dt) {

    StackEntry stack_lock{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    const u32 ndust = solver_config.dust_config.ndust;

    SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tscal> cfield_rho_next_bf_drag
        = utility.make_compute_field<Tscal>("rho_next_bf_drag", AMRBlock::block_size);
    shamrock::ComputeField<Tvec> cfield_rhov_next_bf_drag
        = utility.make_compute_field<Tvec>("rhov_next_bf_drag", AMRBlock::block_size);
    shamrock::ComputeField<Tscal> cfield_rhoe_next_bf_drag
        = utility.make_compute_field<Tscal>("rhoe_next_bf_drag", AMRBlock::block_size);
    shamrock::ComputeField<Tscal> cfield_rho_d_next_bf_drag
        = utility.make_compute_field<Tscal>("rho_d_next_bf_drag", ndust * AMRBlock::block_size);
    shamrock::ComputeField<Tvec> cfield_rhov_d_next_bf_drag
        = utility.make_compute_field<Tvec>("rhov_d_next_bf_drag", ndust * AMRBlock::block_size);

    shamrock::ComputeField<Tscal> &cfield_dtrho   = storage.dtrho.get();
    shamrock::ComputeField<Tvec> &cfield_dtrhov   = storage.dtrhov.get();
    shamrock::ComputeField<Tscal> &cfield_dtrhoe  = storage.dtrhoe.get();
    shamrock::ComputeField<Tscal> &cfield_dtrho_d = storage.dtrho_dust.get();
    shamrock::ComputeField<Tvec> &cfield_dtrhov_d = storage.dtrhov_dust.get();

    // load layout info
    PatchDataLayout &pdl = scheduler().pdl;

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");
    const u32 irho_d    = pdl.get_field_idx<Tscal>("rho_dust");
    const u32 irhovel_d = pdl.get_field_idx<Tvec>("rhovel_dust");

    scheduler().for_each_patchdata_nonempty([&, dt, ndust](
                                                const shamrock::patch::Patch p,
                                                shamrock::patch::PatchData &pdat) {
        logger::debug_ln(
            "[AMR evolve time step before drag ]", "evolve field with no drag patch", p.id_patch);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        u32 id               = p.id_patch;

        sham::DeviceBuffer<Tscal> &dt_rho_patch   = cfield_dtrho.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &dt_rhov_patch   = cfield_dtrhov.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &dt_rhoe_patch  = cfield_dtrhoe.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &dt_rho_d_patch = cfield_dtrho_d.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &dt_rhov_d_patch = cfield_dtrhov_d.get_buf_check(id);

        sham::DeviceBuffer<Tscal> &buf_rho   = pdat.get_field_buf_ref<Tscal>(irho);
        sham::DeviceBuffer<Tvec> &buf_rhov   = pdat.get_field_buf_ref<Tvec>(irhovel);
        sham::DeviceBuffer<Tscal> &buf_rhoe  = pdat.get_field_buf_ref<Tscal>(irhoetot);
        sham::DeviceBuffer<Tscal> &buf_rho_d = pdat.get_field_buf_ref<Tscal>(irho_d);
        sham::DeviceBuffer<Tvec> &buf_rhov_d = pdat.get_field_buf_ref<Tvec>(irhovel_d);

        sham::DeviceBuffer<Tscal> &rho_patch   = cfield_rho_next_bf_drag.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &rhov_patch   = cfield_rhov_next_bf_drag.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &rhoe_patch  = cfield_rhoe_next_bf_drag.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &rho_d_patch = cfield_rho_d_next_bf_drag.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &rhov_d_patch = cfield_rhov_d_next_bf_drag.get_buf_check(id);

        u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

        sham::EventList depend_list;
        auto acc_dt_rho_patch  = dt_rho_patch.get_read_access(depend_list);
        auto acc_dt_rhov_patch = dt_rhov_patch.get_read_access(depend_list);
        auto acc_dt_rhoe_patch = dt_rhoe_patch.get_read_access(depend_list);

        auto rho  = buf_rho.get_read_access(depend_list);
        auto rhov = buf_rhov.get_read_access(depend_list);
        auto rhoe = buf_rhoe.get_read_access(depend_list);

        auto acc_rho  = rho_patch.get_write_access(depend_list);
        auto acc_rhov = rhov_patch.get_write_access(depend_list);
        auto acc_rhoe = rhoe_patch.get_write_access(depend_list);

        auto e1 = q.submit(depend_list, [&, dt](sycl::handler &cgh) {
            shambase::parralel_for(cgh, cell_count, "evolve field with no drag", [=](u32 id_a) {
                acc_rho[id_a]  = rho[id_a] + dt * acc_dt_rho_patch[id_a];
                acc_rhov[id_a] = rhov[id_a] + dt * acc_dt_rhov_patch[id_a];
                acc_rhoe[id_a] = rhoe[id_a] + dt * acc_dt_rhoe_patch[id_a];
            });
        });

        dt_rho_patch.complete_event_state(e1);
        dt_rhov_patch.complete_event_state(e1);
        dt_rhoe_patch.complete_event_state(e1);

        buf_rho.complete_event_state(e1);
        buf_rhov.complete_event_state(e1);
        buf_rhoe.complete_event_state(e1);

        rho_patch.complete_event_state(e1);
        rhov_patch.complete_event_state(e1);
        rhoe_patch.complete_event_state(e1);

        sham::EventList depend_list1;
        auto acc_dt_rho_d_patch  = dt_rho_d_patch.get_read_access(depend_list1);
        auto acc_dt_rhov_d_patch = dt_rhov_d_patch.get_read_access(depend_list1);

        auto rho_d  = buf_rho_d.get_read_access(depend_list1);
        auto rhov_d = buf_rhov_d.get_read_access(depend_list1);

        auto acc_rho_d  = rho_d_patch.get_write_access(depend_list1);
        auto acc_rhov_d = rhov_d_patch.get_write_access(depend_list1);

        auto e2 = q.submit(depend_list1, [&, dt, ndust](sycl::handler &cgh) {
            shambase::parralel_for(
                cgh, ndust * cell_count, "dust  evolve field no drag", [=](u32 id_a) {
                    acc_rho_d[id_a]  = rho_d[id_a] + dt * acc_dt_rho_d_patch[id_a];
                    acc_rhov_d[id_a] = rhov_d[id_a] + dt * acc_dt_rhov_d_patch[id_a];
                });
        });

        dt_rho_d_patch.complete_event_state(e2);
        dt_rhov_d_patch.complete_event_state(e2);

        buf_rho_d.complete_event_state(e2);
        buf_rhov_d.complete_event_state(e2);

        rho_d_patch.complete_event_state(e2);
        rhov_d_patch.complete_event_state(e2);
    });

    storage.rho_next_no_drag.set(std::move(cfield_rho_next_bf_drag));
    storage.rhov_next_no_drag.set(std::move(cfield_rhov_next_bf_drag));
    storage.rhoe_next_no_drag.set(std::move(cfield_rhoe_next_bf_drag));
    storage.rho_d_next_no_drag.set(std::move(cfield_rho_d_next_bf_drag));
    storage.rhov_d_next_no_drag.set(std::move(cfield_rhov_d_next_bf_drag));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::enable_irk1_drag_integrator(
    Tscal dt) {
    StackEntry stack_lock{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    shamrock::ComputeField<Tscal> &cfield_rho_new   = storage.rho_next_no_drag.get();
    shamrock::ComputeField<Tvec> &cfield_rhov_new   = storage.rhov_next_no_drag.get();
    shamrock::ComputeField<Tscal> &cfield_rhoe_new  = storage.rhoe_next_no_drag.get();
    shamrock::ComputeField<Tscal> &cfield_rho_d_new = storage.rho_d_next_no_drag.get();
    shamrock::ComputeField<Tvec> &cfield_rhov_d_new = storage.rhov_d_next_no_drag.get();

    // load layout info
    PatchDataLayout &pdl = scheduler().pdl;

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");
    const u32 irho_d    = pdl.get_field_idx<Tscal>("rho_dust");
    const u32 irhovel_d = pdl.get_field_idx<Tvec>("rhovel_dust");

    const u32 ndust = solver_config.dust_config.ndust;
    // alphas are dust collision rates
    auto alphas_vector = solver_config.drag_config.alphas;
    std::vector<f32> inv_dt_alphas(ndust);
    bool enable_frictional_heating = solver_config.drag_config.enable_frictional_heating;
    u32 friction_control           = (enable_frictional_heating == false) ? 1 : 0;

    scheduler().for_each_patchdata_nonempty([&, dt, ndust, friction_control](
                                                const shamrock::patch::Patch p,
                                                shamrock::patch::PatchData &pdat) {
        logger::debug_ln("[AMR enable drag ]", "irk1 drag patch", p.id_patch);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        u32 id               = p.id_patch;
        u32 cell_count       = pdat.get_obj_cnt() * AMRBlock::block_size;

        sham::DeviceBuffer<Tscal> &rho_new_patch   = cfield_rho_new.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &rhov_new_patch   = cfield_rhov_new.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &rhoe_new_patch  = cfield_rhoe_new.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &rho_d_new_patch = cfield_rho_d_new.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &rhov_d_new_patch = cfield_rhov_d_new.get_buf_check(id);

        sham::DeviceBuffer<Tscal> &rho_old   = pdat.get_field_buf_ref<Tscal>(irho);
        sham::DeviceBuffer<Tvec> &rhov_old   = pdat.get_field_buf_ref<Tvec>(irhovel);
        sham::DeviceBuffer<Tscal> &rhoe_old  = pdat.get_field_buf_ref<Tscal>(irhoetot);
        sham::DeviceBuffer<Tscal> &rho_d_old = pdat.get_field_buf_ref<Tscal>(irho_d);
        sham::DeviceBuffer<Tvec> &rhov_d_old = pdat.get_field_buf_ref<Tvec>(irhovel_d);

        sham::DeviceBuffer<f32> alphas_buf(ndust, shamsys::instance::get_compute_scheduler_ptr());

        alphas_buf.copy_from_stdvec(alphas_vector);

        sham::EventList depend_list;
        auto acc_rho_new_patch    = rho_new_patch.get_read_access(depend_list);
        auto acc_rhov_new_patch   = rhov_new_patch.get_read_access(depend_list);
        auto acc_rhoe_new_patch   = rhoe_new_patch.get_read_access(depend_list);
        auto acc_rho_d_new_patch  = rho_d_new_patch.get_read_access(depend_list);
        auto acc_rhov_d_new_patch = rhov_d_new_patch.get_read_access(depend_list);

        auto acc_rho_old    = rho_old.get_write_access(depend_list);
        auto acc_rhov_old   = rhov_old.get_write_access(depend_list);
        auto acc_rhoe_old   = rhoe_old.get_write_access(depend_list);
        auto acc_rho_d_old  = rho_d_old.get_write_access(depend_list);
        auto acc_rhov_d_old = rhov_d_old.get_write_access(depend_list);

        auto acc_alphas = alphas_buf.get_read_access(depend_list);

        auto e = q.submit(depend_list, [&, dt, ndust, friction_control](sycl::handler &cgh) {
            shambase::parralel_for(cgh, cell_count, "add_drag [irk1]", [=](u32 id_a) {
                f64_3 tmp_mom_1 = acc_rhov_new_patch[id_a];
                f64 tmp_rho     = acc_rho_old[id_a];

                for (u32 i = 0; i < ndust; i++) {
                    const f32 inv_dt_alphas = 1.0 / (1.0 + acc_alphas[i] * dt);
                    const f32 dt_alphas     = dt * acc_alphas[i];

                    tmp_mom_1
                        = tmp_mom_1
                          + dt_alphas * inv_dt_alphas * acc_rhov_d_new_patch[id_a * ndust + i];
                    tmp_rho = tmp_rho + dt_alphas * inv_dt_alphas * acc_rho_d_old[id_a * ndust + i];
                }

                f64 tmp_inv_rho = 1.0 / tmp_rho;
                f64_3 tmp_vel   = tmp_inv_rho * tmp_mom_1;
                f64 Eg          = 0.0;

                f64 inv_rho_g = 1.0 / acc_rho_new_patch[id_a];
                f64_3 vg_bf   = inv_rho_g * acc_rhov_new_patch[id_a];
                f64_3 vg_af   = inv_rho_g * acc_rho_old[id_a] * tmp_vel;
                ;
                f64 work_drag = 0.5
                                * ((acc_rho_old[id_a] * tmp_vel[0] - acc_rhov_new_patch[id_a][0])
                                       * (vg_bf[0] + vg_af[0])
                                   + (acc_rho_old[id_a] * tmp_vel[1] - acc_rhov_new_patch[id_a][1])
                                         * (vg_bf[1] + vg_af[1])
                                   + (acc_rho_old[id_a] * tmp_vel[2] - acc_rhov_new_patch[id_a][2])
                                         * (vg_bf[2] + vg_af[2]));
                f64 dissipation = 0.0;
                for (u32 i = 0; i < ndust; i++) {
                    const f32 inv_dt_alphas = 1.0 / (1.0 + acc_alphas[i] * dt);
                    const f32 dt_alphas     = dt * acc_alphas[i];
                    f64 inv_rho_d           = 1.0 / acc_rho_d_new_patch[id_a * ndust + i];
                    f64_3 vd_bf             = inv_rho_d * acc_rhov_d_new_patch[id_a * ndust + i];
                    f64_3 vd_af             = inv_rho_d * inv_dt_alphas
                                  * (acc_rhov_d_new_patch[id_a * ndust + i]
                                     + dt_alphas * acc_rho_d_old[id_a * ndust + i] * tmp_vel);
                    dissipation += 0.5 * dt_alphas * inv_dt_alphas
                                   * ((acc_rho_d_old[id_a * ndust + i] * tmp_vel[0]
                                       - acc_rhov_d_new_patch[id_a * ndust + i][0])
                                          * (vd_af[0] + vd_bf[0])
                                      + (acc_rho_d_old[id_a * ndust + i] * tmp_vel[1]
                                         - acc_rhov_d_new_patch[id_a * ndust + i][1])
                                            * (vd_af[1] + vd_bf[1])
                                      + (acc_rho_d_old[id_a * ndust + i] * tmp_vel[2]
                                         - acc_rhov_d_new_patch[id_a * ndust + i][2])
                                            * (vd_af[2] + vd_bf[2]));
                }

                Eg += acc_rhoe_new_patch[id_a] + (1 - friction_control) * work_drag
                      - friction_control * dissipation;
                acc_rhov_old[id_a] = tmp_vel * acc_rho_old[id_a];
                acc_rhoe_old[id_a] = Eg;
                acc_rho_old[id_a]  = acc_rho_new_patch[id_a];
                for (u32 i = 0; i < ndust; i++) {
                    const f32 inv_dt_alphas = 1.0 / (1.0 + acc_alphas[i] * dt);
                    const f32 dt_alphas     = dt * acc_alphas[i];
                    acc_rhov_d_old[id_a * ndust + i]
                        = inv_dt_alphas
                          * (acc_rhov_d_new_patch[id_a * ndust + i]
                             + dt_alphas * acc_rho_d_old[id_a * ndust + i] * tmp_vel);
                    acc_rho_d_old[id_a * ndust + i] = acc_rho_d_new_patch[id_a * ndust + i];
                }
            });
        });

        rho_new_patch.complete_event_state(e);
        rhov_new_patch.complete_event_state(e);
        rhoe_new_patch.complete_event_state(e);
        rho_d_new_patch.complete_event_state(e);
        rhov_d_new_patch.complete_event_state(e);

        rho_old.complete_event_state(e);
        rhov_old.complete_event_state(e);
        rhoe_old.complete_event_state(e);
        rho_d_old.complete_event_state(e);
        rhov_d_old.complete_event_state(e);

        alphas_buf.complete_event_state(e);
    });
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::enable_expo_drag_integrator(
    Tscal dt) {
    StackEntry stack_lock{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    shamrock::ComputeField<Tscal> &cfield_rho_new   = storage.rho_next_no_drag.get();
    shamrock::ComputeField<Tvec> &cfield_rhov_new   = storage.rhov_next_no_drag.get();
    shamrock::ComputeField<Tscal> &cfield_rhoe_new  = storage.rhoe_next_no_drag.get();
    shamrock::ComputeField<Tscal> &cfield_rho_d_new = storage.rho_d_next_no_drag.get();
    shamrock::ComputeField<Tvec> &cfield_rhov_d_new = storage.rhov_d_next_no_drag.get();

    // load layout info
    PatchDataLayout &pdl = scheduler().pdl;

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");
    const u32 irho_d    = pdl.get_field_idx<Tscal>("rho_dust");
    const u32 irhovel_d = pdl.get_field_idx<Tvec>("rhovel_dust");

    // const u32 ndust = solver_config.dust_config.ndust;

    const u32 ndust = 1;
    // alphas are dust collision rates
    auto alphas_vector = solver_config.drag_config.alphas;
    std::vector<f32> inv_dt_alphas(ndust);
    bool enable_frictional_heating = solver_config.drag_config.enable_frictional_heating;
    u32 friction_control           = (enable_frictional_heating == false) ? 1 : 0;

    scheduler().for_each_patchdata_nonempty([&, dt, ndust, friction_control](
                                                const shamrock::patch::Patch p,
                                                shamrock::patch::PatchData &pdat) {
        logger::debug_ln("[AMR enable drag ]", "irk1 drag patch", p.id_patch);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        u32 id               = p.id_patch;
        u32 cell_count       = pdat.get_obj_cnt() * AMRBlock::block_size;

        sham::DeviceBuffer<Tscal> &rho_new_patch   = cfield_rho_new.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &rhov_new_patch   = cfield_rhov_new.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &rhoe_new_patch  = cfield_rhoe_new.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &rho_d_new_patch = cfield_rho_d_new.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &rhov_d_new_patch = cfield_rhov_d_new.get_buf_check(id);

        sham::DeviceBuffer<Tscal> &rho_old   = pdat.get_field_buf_ref<Tscal>(irho);
        sham::DeviceBuffer<Tvec> &rhov_old   = pdat.get_field_buf_ref<Tvec>(irhovel);
        sham::DeviceBuffer<Tscal> &rhoe_old  = pdat.get_field_buf_ref<Tscal>(irhoetot);
        sham::DeviceBuffer<Tscal> &rho_d_old = pdat.get_field_buf_ref<Tscal>(irho_d);
        sham::DeviceBuffer<Tvec> &rhov_d_old = pdat.get_field_buf_ref<Tvec>(irhovel_d);

        sham::DeviceBuffer<f32> alphas_buf(ndust, shamsys::instance::get_compute_scheduler_ptr());

        alphas_buf.copy_from_stdvec(alphas_vector);

        sham::EventList depend_list;
        auto acc_rho_new_patch    = rho_new_patch.get_read_access(depend_list);
        auto acc_rhov_new_patch   = rhov_new_patch.get_read_access(depend_list);
        auto acc_rhoe_new_patch   = rhoe_new_patch.get_read_access(depend_list);
        auto acc_rho_d_new_patch  = rho_d_new_patch.get_read_access(depend_list);
        auto acc_rhov_d_new_patch = rhov_d_new_patch.get_read_access(depend_list);

        auto acc_rho_old    = rho_old.get_write_access(depend_list);
        auto acc_rhov_old   = rhov_old.get_write_access(depend_list);
        auto acc_rhoe_old   = rhoe_old.get_write_access(depend_list);
        auto acc_rho_d_old  = rho_d_old.get_write_access(depend_list);
        auto acc_rhov_d_old = rhov_d_old.get_write_access(depend_list);

        auto acc_alphas = alphas_buf.get_read_access(depend_list);

        auto e = q.submit(depend_list, [&, dt, ndust, friction_control](sycl::handler &cgh) {
            // sparse jacobian matrix
            auto get_jacobian = [=](u32 id,
                                    std::array<std::array<f64, ndust + 1>, ndust + 1> &jacobian) {
                // fill first row
                for (auto j = 1; j < ndust + 1; j++)
                    jacobian[0][j] = acc_alphas[j - 1];
                // fil first column
                for (auto i = 1; i < ndust + 1; i++) {
                    jacobian[i][0]
                        = acc_alphas[i - 1]
                          * (acc_rho_new_patch[id] / acc_rho_d_new_patch[id * ndust + (i - 1)]);
                    jacobian[0][0] -= jacobian[i][0];
                }
                // fill diagonal from (i,j)=(1,1)
                for (auto i = 1; i < ndust + 1; i++)
                    jacobian[i][i] = -acc_alphas[i - 1];
                // the rest of the buffer is set to zero
            };

            shambase::parralel_for(cgh, cell_count, "add_drag [expo]", [=](u32 id_a) {
                f64 mu = 0;
                for (auto i = 0; i < ndust; i++) {
                    mu += (1 + (acc_rho_d_new_patch[id_a * ndust + i] / acc_rho_new_patch[id_a]))
                          * acc_alphas[i];
                }
                mu *= (-dt / (ndust + 1));

                // construct jacobian matrix for current cell
                std::array<std::array<f64, ndust + 1>, ndust + 1> jacob = {0};
                get_jacobian(id_a, jacob);

                //
                shammath::compute_scalMat(jacob, dt, ndust + 1);
                shammath::compute_add_id_scal(jacob, 1.0, -mu);

                const i32 K_exp = 9;
                shammath::mat_expo(K_exp, jacob, ndust + 1);

                shammath::compute_scalMat(jacob, sycl::exp(mu), ndust + 1);
                f64_3 r = {0., 0., 0.}, dd = {0., 0., 0.};
                r += jacob[0][0] * acc_rhov_new_patch[id_a];

                for (auto j = 1; j < ndust + 1; j++) {
                    r[0] += jacob[0][j] * acc_rhov_d_new_patch[id_a * ndust + (j - 1)][0];
                    r[1] += jacob[0][j] * acc_rhov_d_new_patch[id_a * ndust + (j - 1)][1];
                    r[2] += jacob[0][j] * acc_rhov_d_new_patch[id_a * ndust + (j - 1)][2];
                }

                dd[0] = r[0] - acc_rhov_new_patch[id_a][0];
                dd[1] = r[1] - acc_rhov_new_patch[id_a][1];
                dd[2] = r[2] - acc_rhov_new_patch[id_a][2];

                // f64 Eg = acc_rhoe_new_patch [id_a];
                f64 dissipation = 0, drag_work = 0;
                f64 inv_rho = 1.0 / (acc_rho_new_patch[id_a]);

                f64_3 v_bf = inv_rho * acc_rhov_new_patch[id_a];
                f64_3 v_af = inv_rho * r;

                drag_work = 0.5
                            * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
                               + dd[2] * (v_bf[2] + v_af[2]));

                // save gas momentum back
                acc_rhov_old[id_a] = r;

                //
                for (auto d_id = 1; d_id <= ndust; d_id++) {
                    r *= 0;
                    r += jacob[d_id][0] * acc_rhov_new_patch[id_a];

                    for (auto j = 1; j <= ndust; j++) {

                        r += jacob[d_id][j] * acc_rhov_d_new_patch[id_a * ndust + (j - 1)];
                    }

                    dd = r - acc_rhov_d_new_patch[id_a * ndust + (d_id - 1)];

                    inv_rho = 1.0 / (acc_rho_d_new_patch[id_a * ndust + (d_id - 1)]);

                    v_bf = inv_rho * acc_rhov_d_new_patch[id_a * ndust + (d_id - 1)];

                    v_af = inv_rho * r;

                    dissipation += 0.5
                                   * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
                                      + dd[2] * (v_bf[2] + v_af[2]));

                    // save dust momentum back
                    acc_rhov_d_old[id_a * ndust + (d_id - 1)] = r;
                }

                acc_rhoe_old[id_a]
                    += (1 - friction_control) * drag_work - friction_control * dissipation;
            });
        });

        rho_new_patch.complete_event_state(e);
        rhov_new_patch.complete_event_state(e);
        rhoe_new_patch.complete_event_state(e);
        rho_d_new_patch.complete_event_state(e);
        rhov_d_new_patch.complete_event_state(e);

        rho_old.complete_event_state(e);
        rhov_old.complete_event_state(e);
        rhoe_old.complete_event_state(e);
        rho_d_old.complete_event_state(e);
        rhov_d_old.complete_event_state(e);

        alphas_buf.complete_event_state(e);
    });
}
template class shammodels::basegodunov::modules::DragIntegrator<f64_3, i64_3>;
