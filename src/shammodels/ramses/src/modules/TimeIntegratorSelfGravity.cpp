// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TimeIntegratorSelfGravity.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shambase/memory.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shammodels/ramses/modules/TimeIntegratorSelfGravity.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::TimeIntegratorSelfGravity<Tvec, TgridVec>::forward_euler(
    Tscal dt) {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    // load layout info
    PatchDataLayerLayout &pdl = scheduler().pdl_old();

    const u32 irho     = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel  = pdl.get_field_idx<Tvec>("rhovel");
    const u32 iphi     = pdl.get_field_idx<Tscal>("phi_old");
    const u32 iphi_new = pdl.get_field_idx<Tscal>("phi");
    {

        // auto &rho_next  = shambase::get_check_ref(storage.refs_rho_next);

        auto &rho_next     = shambase::get_check_ref(storage.refs_rho);
        auto &rhov_next    = shambase::get_check_ref(storage.refs_rhov_next);
        auto &rhoe_next    = shambase::get_check_ref(storage.refs_rhoe_next);
        auto &phi_next     = shambase::get_check_ref(storage.refs_phi);
        auto &phi_new_next = shambase::get_check_ref(storage.refs_phi_new);

        scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch p,
                                                    shamrock::patch::PatchDataLayer &pdat) {
            shamlog_debug_ln(
                "[AMR Flux]", "forward euler integration-self-gravity patch", p.id_patch);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::DeviceBuffer<Tscal> &rho_next_patch     = rho_next.get(p.id_patch).get_buf();
            sham::DeviceBuffer<Tvec> &rhov_next_patch     = rhov_next.get_buf(p.id_patch);
            sham::DeviceBuffer<Tscal> &rhoe_next_patch    = rhoe_next.get_buf(p.id_patch);
            sham::DeviceBuffer<Tscal> &phi_old_next_patch = phi_next.get(p.id_patch).get_buf();
            sham::DeviceBuffer<Tscal> &phi_new_next_patch = phi_new_next.get(p.id_patch).get_buf();

            u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

            sham::DeviceBuffer<Tscal> &buf_rho  = pdat.get_field_buf_ref<Tscal>(irho);
            sham::DeviceBuffer<Tvec> &buf_rhov  = pdat.get_field_buf_ref<Tvec>(irhovel);
            sham::DeviceBuffer<Tscal> &buf_rhoe = pdat.get_field_buf_ref<Tscal>(irhoetot);
            sham::DeviceBuffer<Tscal> &phi_old  = pdat.get_field_buf_ref<Tscal>(iphi);
            sham::DeviceBuffer<Tscal> &phi_new  = pdat.get_field_buf_ref<Tscal>(iphi_new);

            sham::EventList depends_list;
            auto acc_rho_next_patch  = rho_next_patch.get_read_access(depends_list);
            auto acc_rhov_next_patch = rhov_next_patch.get_read_access(depends_list);
            auto acc_rhoe_next_patch = rhoe_next_patch.get_write_access(depends_list);
            auto rho_old             = buf_rho.get_write_access(depends_list);
            auto rhov_old            = buf_rhov.get_write_access(depends_list);
            auto rhoe_old            = buf_rhoe.get_write_access(depends_list);

            auto acc_phi_new      = phi_old_next_patch.get_read_access(depends_list);
            auto acc_phi_old      = phi_old.get_write_access(depends_list);
            auto acc_phi_next_new = phi_new_next_patch.get_read_access(depends_list);
            auto acc_phi_next_old = phi_new.get_write_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                shambase::parallel_for(cgh, cell_count, "saveback", [=](u32 id_a) {
                    auto vel  = acc_rhov_next_patch[id_a] / acc_rho_next_patch[id_a];
                    auto Ekin = 0.5 * acc_rho_next_patch[id_a]
                                * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);

                    shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
                    auto m_H   = ctes.proton_mass(); // [kg]
                    auto kb    = ctes.kb();          // []
                    auto mu    = 2.3;                // molecular gas
                    auto gamma = 5. / 3.;            //

                    // auto m_H = 1.67262192e-27;  //[kg]
                    // auto kb = 1.380649e-23;
                    // auto T = 10.;
                    auto T = 10.747;

                    auto cs0_sqr  = (kb * T) / (mu * m_H);
                    auto rho_crit = 3.7e-13 * 1e3; //[kg*m^-3]
                    auto P = acc_rho_next_patch[id_a] * cs0_sqr
                             * (1. + sycl::pow(acc_rho_next_patch[id_a] / rho_crit, 2. / 3.));

                    auto Eint     = P / (gamma - 1.);
                    rho_old[id_a] = acc_rho_next_patch[id_a];

                    rhov_old[id_a] = acc_rhov_next_patch[id_a];
                    // rhoe_old[id_a] = acc_rhoe_next_patch[id_a];

                    rhoe_old[id_a]            = Ekin + Eint;
                    acc_rhoe_next_patch[id_a] = Ekin + Eint;
                    acc_phi_old[id_a]         = acc_phi_new[id_a];
                    acc_phi_next_old[id_a]    = acc_phi_next_new[id_a];
                });
            });

            rho_next_patch.complete_event_state(e);
            rhov_next_patch.complete_event_state(e);
            rhoe_next_patch.complete_event_state(e);
            buf_rho.complete_event_state(e);
            buf_rhov.complete_event_state(e);
            buf_rhoe.complete_event_state(e);

            phi_old_next_patch.complete_event_state(e);
            phi_old.complete_event_state(e);
            phi_new_next_patch.complete_event_state(e);
            phi_new.complete_event_state(e);
        });
    }

    if (solver_config.is_dust_on()) {
        const u32 irho_d    = pdl.get_field_idx<Tscal>("rho_dust");
        const u32 irhovel_d = pdl.get_field_idx<Tvec>("rhovel_dust");
        auto &rho_next_d    = shambase::get_check_ref(storage.refs_rho_dust);
        auto &rhov_next_d   = shambase::get_check_ref(storage.refs_rhov_next_d);
        scheduler().for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                shamlog_debug_ln(
                    "[AMR Flux]", "forward euler integration-self-gravity patch dust", p.id_patch);

                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
                sham::DeviceBuffer<Tscal> &rho_next_patch_d = rho_next_d.get(p.id_patch).get_buf();
                sham::DeviceBuffer<Tvec> &rhov_next_patch_d = rhov_next_d.get_buf(p.id_patch);

                u32 cell_count                       = pdat.get_obj_cnt() * AMRBlock::block_size;
                u32 ndust                            = solver_config.dust_config.ndust;
                sham::DeviceBuffer<Tscal> &buf_rho_d = pdat.get_field_buf_ref<Tscal>(irho_d);
                sham::DeviceBuffer<Tvec> &buf_rhov_d = pdat.get_field_buf_ref<Tvec>(irhovel_d);

                sham::EventList depends_list;
                auto acc_rho_next_patch_d  = rho_next_patch_d.get_read_access(depends_list);
                auto acc_rhov_next_patch_d = rhov_next_patch_d.get_read_access(depends_list);
                auto rho_old_d             = buf_rho_d.get_write_access(depends_list);
                auto rhov_old_d            = buf_rhov_d.get_write_access(depends_list);
                auto e                     = q.submit(depends_list, [&](sycl::handler &cgh) {
                    shambase::parallel_for(cgh, ndust * cell_count, "saveback", [=](u32 id_a) {
                        rho_old_d[id_a]  = acc_rho_next_patch_d[id_a];
                        rhov_old_d[id_a] = acc_rhov_next_patch_d[id_a];
                    });
                });
                buf_rho_d.complete_event_state(e);
                buf_rhov_d.complete_event_state(e);
                rho_next_patch_d.complete_event_state(e);
                rhov_next_patch_d.complete_event_state(e);
            });
    }
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::TimeIntegratorSelfGravity<Tvec, TgridVec>::
    enable_irk1_drag_integrator(Tscal dt) {
    StackEntry stack_lock{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    // shamrock::ComputeField<Tscal> &cfield_rho_new   = storage.rho_next_no_drag.get();
    // shamrock::ComputeField<Tvec> &cfield_rhov_new   = storage.rhov_next_no_drag.get();
    // shamrock::ComputeField<Tscal> &cfield_rhoe_new  = storage.rhoe_next_no_drag.get();
    // shamrock::ComputeField<Tscal> &cfield_rho_d_new = storage.rho_d_next_no_drag.get();
    // shamrock::ComputeField<Tvec> &cfield_rhov_d_new = storage.rhov_d_next_no_drag.get();

    // load layout info
    PatchDataLayerLayout &pdl = scheduler().pdl_old();

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");
    const u32 irho_d    = pdl.get_field_idx<Tscal>("rho_dust");
    const u32 irhovel_d = pdl.get_field_idx<Tvec>("rhovel_dust");

    const u32 ndust = solver_config.dust_config.ndust;
    // // alphas are dust collision rates
    // auto alphas_vector = solver_config.drag_config.alphas;
    auto internal_rho = solver_config.drag_config.intrinsic_density;
    auto grains_size  = solver_config.drag_config.grains_sizes;
    std::vector<Tscal> inv_dt_alphas(ndust);
    bool enable_frictional_heating    = solver_config.drag_config.enable_frictional_heating;
    bool compute_epstein_stoping_time = solver_config.drag_config.compute_epstein_stoping_time;
    u32 friction_control              = (enable_frictional_heating == false) ? 1 : 0;
    auto gamma                        = solver_config.eos_gamma;

    auto &rho_next_star    = shambase::get_check_ref(storage.refs_rho);
    auto &rhov_next_star   = shambase::get_check_ref(storage.refs_rhov_next);
    auto &rhoe_next        = shambase::get_check_ref(storage.refs_rhoe_next);
    auto &rho_next_star_d  = shambase::get_check_ref(storage.refs_rho_dust);
    auto &rhov_next_star_d = shambase::get_check_ref(storage.refs_rhov_next_d);

    scheduler().for_each_patchdata_nonempty([&, dt, ndust, friction_control](
                                                const shamrock::patch::Patch p,
                                                shamrock::patch::PatchDataLayer &pdat) {
        shamlog_debug_ln("[AMR enable drag with SG ]", "irk1 drag patch", p.id_patch);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        u32 id               = p.id_patch;
        u32 cell_count       = pdat.get_obj_cnt() * AMRBlock::block_size;

        sham::DeviceBuffer<Tscal> &rho_new_patch   = rho_next_star.get(p.id_patch).get_buf();
        sham::DeviceBuffer<Tvec> &rhov_new_patch   = rhov_next_star.get(p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &rhoe_new_patch  = rhoe_next.get(p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &rho_d_new_patch = rho_next_star_d.get(p.id_patch).get_buf();
        sham::DeviceBuffer<Tvec> &rhov_d_new_patch = rhov_next_star_d.get(p.id_patch).get_buf();

        sham::DeviceBuffer<Tscal> &rho_old   = pdat.get_field_buf_ref<Tscal>(irho);
        sham::DeviceBuffer<Tvec> &rhov_old   = pdat.get_field_buf_ref<Tvec>(irhovel);
        sham::DeviceBuffer<Tscal> &rhoe_old  = pdat.get_field_buf_ref<Tscal>(irhoetot);
        sham::DeviceBuffer<Tscal> &rho_d_old = pdat.get_field_buf_ref<Tscal>(irho_d);
        sham::DeviceBuffer<Tvec> &rhov_d_old = pdat.get_field_buf_ref<Tvec>(irhovel_d);

        // sham::DeviceBuffer<Tscal> alphas_buf(ndust,
        // shamsys::instance::get_compute_scheduler_ptr());
        sham::DeviceBuffer<Tscal> internal_rho_buf(
            ndust, shamsys::instance::get_compute_scheduler_ptr());
        sham::DeviceBuffer<Tscal> grains_size_buf(
            ndust, shamsys::instance::get_compute_scheduler_ptr());

        // alphas_buf.copy_from_stdvec(alphas_vector);
        internal_rho_buf.copy_from_stdvec(internal_rho);
        grains_size_buf.copy_from_stdvec(grains_size);

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

        // auto acc_alphas       = alphas_buf.get_read_access(depend_list);
        auto acc_internal_rho = internal_rho_buf.get_read_access(depend_list);
        auto acc_grains_size  = grains_size_buf.get_read_access(depend_list);

        auto e = q.submit(depend_list, [&, dt, ndust, friction_control](sycl::handler &cgh) {
            shambase::parallel_for(cgh, cell_count, "add_drag [irk1] SG", [=](u32 id_a) {
                Tvec tmp_mom_1 = acc_rhov_new_patch[id_a];
                Tscal tmp_rho  = acc_rho_old[id_a];

                auto conststate = shammath::ConsState<Tvec>{
                    acc_rho_new_patch[id_a], acc_rhoe_new_patch[id_a], acc_rhov_new_patch[id_a]};
                auto primstate = shammath::cons_to_prim(conststate, gamma);
                auto cs        = sound_speed(primstate, gamma);

                for (u32 i = 0; i < ndust; i++) {
                    // logger::raw_ln("_alppha \t", )
                    // const Tscal inv_dt_alphas = 1.0 / (1.0 + acc_alphas[i] * dt);
                    // const Tscal dt_alphas     = dt * acc_alphas[i];
                    // logger::raw_ln("sg_i\t", acc_grains_size[i], "\t rg_i ", acc_internal_rho[i]
                    // ,"\n\n");
                    const Tscal ts_i
                        = sycl::sqrt((shamunits::pi<Tscal> * solver_config.eos_gamma) / 8.0)
                          * (acc_internal_rho[i] * acc_grains_size[i])
                          / (acc_rho_new_patch[id_a] * cs);
                    const Tscal _alpha        = 1. / ts_i;
                    const Tscal inv_dt_alphas = 1.0 / (1.0 + _alpha * dt);
                    const Tscal dt_alphas     = dt * _alpha;

                    tmp_mom_1
                        = tmp_mom_1
                          + dt_alphas * inv_dt_alphas * acc_rhov_d_new_patch[id_a * ndust + i];
                    tmp_rho = tmp_rho + dt_alphas * inv_dt_alphas * acc_rho_d_old[id_a * ndust + i];
                }

                Tscal tmp_inv_rho = 1.0 / tmp_rho;
                Tvec tmp_vel      = tmp_inv_rho * tmp_mom_1;
                Tscal Eg          = 0.0;

                Tscal inv_rho_g = 1.0 / acc_rho_new_patch[id_a];
                Tvec vg_bf      = inv_rho_g * acc_rhov_new_patch[id_a];
                Tvec vg_af      = inv_rho_g * acc_rho_old[id_a] * tmp_vel;
                ;
                Tscal work_drag
                    = 0.5
                      * ((acc_rho_old[id_a] * tmp_vel[0] - acc_rhov_new_patch[id_a][0])
                             * (vg_bf[0] + vg_af[0])
                         + (acc_rho_old[id_a] * tmp_vel[1] - acc_rhov_new_patch[id_a][1])
                               * (vg_bf[1] + vg_af[1])
                         + (acc_rho_old[id_a] * tmp_vel[2] - acc_rhov_new_patch[id_a][2])
                               * (vg_bf[2] + vg_af[2]));
                Tscal dissipation = 0.0;
                for (u32 i = 0; i < ndust; i++) {
                    // const Tscal inv_dt_alphas = 1.0 / (1.0 + acc_alphas[i] * dt);
                    // const Tscal dt_alphas     = dt * acc_alphas[i];
                    const Tscal ts_i
                        = sycl::sqrt((shamunits::pi<Tscal> * solver_config.eos_gamma) / 8.0)
                          * (acc_internal_rho[i] * acc_grains_size[i])
                          / (acc_rho_new_patch[id_a] * cs);
                    const Tscal _alpha        = 1. / ts_i;
                    const Tscal inv_dt_alphas = 1.0 / (1.0 + _alpha * dt);
                    const Tscal dt_alphas     = dt * _alpha;

                    Tscal inv_rho_d = 1.0 / acc_rho_d_new_patch[id_a * ndust + i];
                    Tvec vd_bf      = inv_rho_d * acc_rhov_d_new_patch[id_a * ndust + i];
                    Tvec vd_af      = inv_rho_d * inv_dt_alphas
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
                    // const Tscal inv_dt_alphas = 1.0 / (1.0 + acc_alphas[i] * dt);
                    // const Tscal dt_alphas     = dt * acc_alphas[i];

                    const Tscal ts_i
                        = sycl::sqrt((shamunits::pi<Tscal> * solver_config.eos_gamma) / 8.0)
                          * (acc_internal_rho[i] * acc_grains_size[i])
                          / (acc_rho_new_patch[id_a] * cs);
                    const Tscal _alpha        = 1. / ts_i;
                    const Tscal inv_dt_alphas = 1.0 / (1.0 + _alpha * dt);
                    const Tscal dt_alphas     = dt * _alpha;
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

        // alphas_buf.complete_event_state(e);
        internal_rho_buf.complete_event_state(e);
        grains_size_buf.complete_event_state(e);
    });
}

// template<class Tvec, class TgridVec>
// void shammodels::basegodunov::modules::TimeIntegratorSelfGravity<Tvec,
// TgridVec>::enable_expo_drag_integrator(
//     Tscal dt) {
//     StackEntry stack_lock{};

//     using namespace shamrock::patch;
//     using namespace shamrock;
//     using namespace shammath;

//     shamrock::ComputeField<Tscal> &cfield_rho_new   = storage.rho_next_no_drag.get();
//     shamrock::ComputeField<Tvec> &cfield_rhov_new   = storage.rhov_next_no_drag.get();
//     shamrock::ComputeField<Tscal> &cfield_rhoe_new  = storage.rhoe_next_no_drag.get();
//     shamrock::ComputeField<Tscal> &cfield_rho_d_new = storage.rho_d_next_no_drag.get();
//     shamrock::ComputeField<Tvec> &cfield_rhov_d_new = storage.rhov_d_next_no_drag.get();

//     // load layout info
//     PatchDataLayerLayout &pdl = scheduler().pdl_old();

//     const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
//     const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
//     const u32 irho      = pdl.get_field_idx<Tscal>("rho");
//     const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
//     const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");
//     const u32 irho_d    = pdl.get_field_idx<Tscal>("rho_dust");
//     const u32 irhovel_d = pdl.get_field_idx<Tvec>("rhovel_dust");

//     const u32 ndust = solver_config.dust_config.ndust;

//     // alphas are dust collision rates
//     auto alphas_vector = solver_config.drag_config.alphas;
//     std::vector<Tscal> inv_dt_alphas(ndust);
//     bool enable_frictional_heating = solver_config.drag_config.enable_frictional_heating;
//     u32 friction_control           = (enable_frictional_heating == false) ? 1 : 0;

//     scheduler().for_each_patchdata_nonempty([&, dt, ndust, friction_control](
//                                                 const shamrock::patch::Patch p,
//                                                 shamrock::patch::PatchDataLayer &pdat) {
//         shamlog_debug_ln("[Ramses]", "expo drag on patch", p.id_patch);

//         sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
//         u32 id               = p.id_patch;
//         u32 cell_count       = pdat.get_obj_cnt() * AMRBlock::block_size;

//         sham::DeviceBuffer<Tscal> &rho_new_patch   = cfield_rho_new.get_buf_check(id);
//         sham::DeviceBuffer<Tvec> &rhov_new_patch   = cfield_rhov_new.get_buf_check(id);
//         sham::DeviceBuffer<Tscal> &rhoe_new_patch  = cfield_rhoe_new.get_buf_check(id);
//         sham::DeviceBuffer<Tscal> &rho_d_new_patch = cfield_rho_d_new.get_buf_check(id);
//         sham::DeviceBuffer<Tvec> &rhov_d_new_patch = cfield_rhov_d_new.get_buf_check(id);

//         sham::DeviceBuffer<Tscal> &rho_old   = pdat.get_field_buf_ref<Tscal>(irho);
//         sham::DeviceBuffer<Tvec> &rhov_old   = pdat.get_field_buf_ref<Tvec>(irhovel);
//         sham::DeviceBuffer<Tscal> &rhoe_old  = pdat.get_field_buf_ref<Tscal>(irhoetot);
//         sham::DeviceBuffer<Tscal> &rho_d_old = pdat.get_field_buf_ref<Tscal>(irho_d);
//         sham::DeviceBuffer<Tvec> &rhov_d_old = pdat.get_field_buf_ref<Tvec>(irhovel_d);

//         sham::DeviceBuffer<Tscal> alphas_buf(ndust,
//         shamsys::instance::get_compute_scheduler_ptr());

//         alphas_buf.copy_from_stdvec(alphas_vector);

//         sham::EventList depend_list;
//         auto acc_rho_new_patch    = rho_new_patch.get_read_access(depend_list);
//         auto acc_rhov_new_patch   = rhov_new_patch.get_read_access(depend_list);
//         auto acc_rhoe_new_patch   = rhoe_new_patch.get_read_access(depend_list);
//         auto acc_rho_d_new_patch  = rho_d_new_patch.get_read_access(depend_list);
//         auto acc_rhov_d_new_patch = rhov_d_new_patch.get_read_access(depend_list);

//         auto acc_rho_old    = rho_old.get_write_access(depend_list);
//         auto acc_rhov_old   = rhov_old.get_write_access(depend_list);
//         auto acc_rhoe_old   = rhoe_old.get_write_access(depend_list);
//         auto acc_rho_d_old  = rho_d_old.get_write_access(depend_list);
//         auto acc_rhov_d_old = rhov_d_old.get_write_access(depend_list);

//         auto acc_alphas = alphas_buf.get_read_access(depend_list);

//         size_t mat_size         = ndust + 1;
//         size_t mat_size_squared = mat_size * mat_size;
//         size_t group_size
//             = (q.get_device_prop().local_mem_size) / (5 * mat_size_squared * sizeof(Tscal));
//         size_t loc_acc_size = mat_size_squared * group_size;

//         size_t loc_mem_size = 5 * sizeof(Tscal) * loc_acc_size;

//         if (group_size < 8) {
//             sham::DeviceBuffer<Tscal> scratch_expo(
//                 5 * mat_size_squared * cell_count,
//                 shamsys::instance::get_compute_scheduler_ptr());
//             Tscal *exp_scratch_ptr_base = scratch_expo.get_write_access(depend_list);
//             auto e = q.submit(depend_list, [&, dt, ndust, friction_control](sycl::handler &cgh) {
//                 shambase::parallel_for(
//                     cgh, cell_count, "add_drag [expo-global-mem]", [=](u32 id_a) {
//                         // sparse jacobian matrix
//                         auto get_jacobian =
//                             [=](u32 id,
//                                 std::mdspan<
//                                     Tscal,
//                                     std::extents<size_t, std::dynamic_extent,
//                                     std::dynamic_extent>> &jacobian) {
//                                 mat_set_nul<Tscal>(jacobian);
//                                 // fill first row
//                                 for (auto j = 1; j < jacobian.extent(1); j++)
//                                     jacobian(0, j) = acc_alphas[j - 1];
//                                 // fil first column
//                                 for (auto i = 1; i < jacobian.extent(0); i++) {
//                                     jacobian(i, 0) = acc_alphas[i - 1]
//                                                      * (acc_rho_d_new_patch[id * ndust + (i - 1)]
//                                                         / acc_rho_new_patch[id]);
//                                     jacobian(0, 0) -= jacobian(i, 0);
//                                 }
//                                 // fill diagonal from (i,j)=(1,1)
//                                 for (auto i = 1; i < jacobian.extent(0); i++)
//                                     jacobian(i, i) = -acc_alphas[i - 1];
//                                 // the rest of the buffer is set to zero
//                             };
//                         Tscal mu = 0;
//                         for (auto i = 0; i < ndust; i++) {
//                             mu += (1
//                                    + (acc_rho_d_new_patch[id_a * ndust + i]
//                                       / acc_rho_new_patch[id_a]))
//                                   * acc_alphas[i];
//                         }
//                         mu *= (-dt / (ndust + 1));

//                         // get ptr to datas
//                         Tscal *ptr_A  = exp_scratch_ptr_base + (id_a * 5 * mat_size_squared);
//                         Tscal *ptr_B  = exp_scratch_ptr_base + (id_a * 5 * mat_size_squared)
//                                         + mat_size_squared;
//                         Tscal *ptr_F  = exp_scratch_ptr_base + (id_a * 5 * mat_size_squared)
//                                         + 2 * mat_size_squared;
//                         Tscal *ptr_I  = exp_scratch_ptr_base + (id_a * 5 * mat_size_squared)
//                                         + 3 * mat_size_squared;
//                         Tscal *ptr_Id = exp_scratch_ptr_base + (id_a * 5 * mat_size_squared)
//                                         + 4 * mat_size_squared;

//                         // create mdspan(s)
//                         std::mdspan<
//                             Tscal,
//                             std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
//                             mdspan_A(ptr_A, mat_size, mat_size);
//                         std::mdspan<
//                             Tscal,
//                             std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
//                             mdspan_B(ptr_B, mat_size, mat_size);
//                         std::mdspan<
//                             Tscal,
//                             std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
//                             mdspan_F(ptr_F, mat_size, mat_size);
//                         std::mdspan<
//                             Tscal,
//                             std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
//                             mdspan_I(ptr_I, mat_size, mat_size);
//                         std::mdspan<
//                             Tscal,
//                             std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
//                             mdspan_Id(ptr_Id, mat_size, mat_size);

//                         get_jacobian(id_a, mdspan_A);

//                         // pre-processing step
//                         shammath::mat_set_identity<Tscal>(mdspan_Id);
//                         shammath::mat_axpy_beta<Tscal, Tscal>(-mu, mdspan_Id, dt, mdspan_A);

//                         // compute matrix exponential
//                         const i32 K_exp = 9;
//                         shammath::mat_exp<Tscal, Tscal>(
//                             K_exp, mdspan_A, mdspan_F, mdspan_B, mdspan_I, mdspan_Id, ndust + 1);

//                         // post-processing step
//                         shammath::mat_mul_scalar<Tscal>(mdspan_A, sycl::exp(mu));

//                         // use the matrix exponential to for to updates momemtum
//                         Tvec r = {0., 0., 0.}, dd = {0., 0., 0.};
//                         r += mdspan_A(0, 0) * acc_rhov_new_patch[id_a];

//                         for (auto j = 1; j < ndust + 1; j++) {
//                             r += mdspan_A(0, j) * acc_rhov_d_new_patch[id_a * ndust + (j - 1)];
//                         }

//                         dd = r - acc_rhov_new_patch[id_a];

//                         Tscal dissipation = 0, drag_work = 0;

//                         // compute work of drag terms
//                         Tscal inv_rho = 1.0 / (acc_rho_new_patch[id_a]);

//                         Tvec v_bf = inv_rho * acc_rhov_new_patch[id_a];
//                         Tvec v_af = inv_rho * r;

//                         drag_work = 0.5
//                                     * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
//                                        + dd[2] * (v_bf[2] + v_af[2]));

//                         // save gas momentum back
//                         acc_rhov_old[id_a] = r;
//                         acc_rho_old[id_a]  = acc_rho_new_patch[id_a];

//                         for (auto d_id = 1; d_id <= ndust; d_id++) {
//                             r *= 0;
//                             r += mdspan_A(d_id, 0) * acc_rhov_new_patch[id_a];

//                             for (auto j = 1; j <= ndust; j++) {

//                                 r += mdspan_A(d_id, j)
//                                      * acc_rhov_d_new_patch[id_a * ndust + (j - 1)];
//                             }

//                             dd = r - acc_rhov_d_new_patch[id_a * ndust + (d_id - 1)];

//                             inv_rho = 1.0 / (acc_rho_d_new_patch[id_a * ndust + (d_id - 1)]);

//                             v_bf = inv_rho * acc_rhov_d_new_patch[id_a * ndust + (d_id - 1)];

//                             v_af = inv_rho * r;

//                             // compute dissipaation by id-th dust
//                             dissipation
//                                 += 0.5
//                                    * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
//                                       + dd[2] * (v_bf[2] + v_af[2]));

//                             // save dust momentum back
//                             acc_rhov_d_old[id_a * ndust + (d_id - 1)] = r;
//                             acc_rho_d_old[id_a * ndust + (d_id - 1)]
//                                 = acc_rho_d_new_patch[id_a * ndust + (d_id - 1)];
//                         }

//                         // updates energy
//                         acc_rhoe_old[id_a] = acc_rhoe_new_patch[id_a]
//                                              + (1 - friction_control) * drag_work
//                                              - friction_control * dissipation;
//                     });
//             });

//             rho_new_patch.complete_event_state(e);
//             rhov_new_patch.complete_event_state(e);
//             rhoe_new_patch.complete_event_state(e);
//             rho_d_new_patch.complete_event_state(e);
//             rhov_d_new_patch.complete_event_state(e);

//             rho_old.complete_event_state(e);
//             rhov_old.complete_event_state(e);
//             rhoe_old.complete_event_state(e);
//             rho_d_old.complete_event_state(e);
//             rhov_d_old.complete_event_state(e);

//             alphas_buf.complete_event_state(e);
//             scratch_expo.complete_event_state(e);

//         } else {

//             if (loc_mem_size > q.get_device_prop().local_mem_size) {
//                 shambase::throw_with_loc<std::runtime_error>(shambase::format(
//                     "not enough local memory for expo drag integrator:\n"
//                     "loc_mem_size: {} > max_local_mem: {}\n"
//                     "loc_acc_size: {}\n"
//                     "group_size: {}\n"
//                     "ndust: {}\n",
//                     loc_mem_size,
//                     q.get_device_prop().local_mem_size,
//                     loc_acc_size,
//                     group_size,
//                     ndust));
//             }

//             auto e = q.submit(depend_list, [&, dt, ndust, friction_control](sycl::handler &cgh) {
//                 // local/shared memory alloc for each work-item
//                 sycl::local_accessor<Tscal> local_A(loc_acc_size, cgh);
//                 sycl::local_accessor<Tscal> local_B(loc_acc_size, cgh);
//                 sycl::local_accessor<Tscal> local_F(loc_acc_size, cgh);
//                 sycl::local_accessor<Tscal> local_I(loc_acc_size, cgh);
//                 sycl::local_accessor<Tscal> local_Id(loc_acc_size, cgh);

//                 logger::debug_sycl_ln(
//                     "SYCL", shambase::format("parallel_for add_drag [expo-shared-mem]"));
//                 cgh.parallel_for(
//                     shambase::make_range(cell_count, group_size), [=](sycl::nd_item<1> id) {
//                         u32 loc_id = id.get_local_id();
//                         u32 id_a   = id.get_global_id();
//                         if (id_a >= cell_count)
//                             return;

//                         // sparse jacobian matrix
//                         auto get_jacobian =
//                             [=](u32 id,
//                                 std::mdspan<
//                                     Tscal,
//                                     std::extents<size_t, std::dynamic_extent,
//                                     std::dynamic_extent>> &jacobian) {
//                                 mat_set_nul<Tscal>(jacobian);
//                                 // fill first row
//                                 for (auto j = 1; j < jacobian.extent(1); j++)
//                                     jacobian(0, j) = acc_alphas[j - 1];
//                                 // fil first column
//                                 for (auto i = 1; i < jacobian.extent(0); i++) {
//                                     jacobian(i, 0) = acc_alphas[i - 1]
//                                                      * (acc_rho_d_new_patch[id * ndust + (i - 1)]
//                                                         / acc_rho_new_patch[id]);
//                                     jacobian(0, 0) -= jacobian(i, 0);
//                                 }
//                                 // fill diagonal from (i,j)=(1,1)
//                                 for (auto i = 1; i < jacobian.extent(0); i++)
//                                     jacobian(i, i) = -acc_alphas[i - 1];
//                                 // the rest of the buffer is set to zero
//                             };

//                         Tscal mu = 0;
//                         for (auto i = 0; i < ndust; i++) {
//                             mu += (1
//                                    + (acc_rho_d_new_patch[id_a * ndust + i]
//                                       / acc_rho_new_patch[id_a]))
//                                   * acc_alphas[i];
//                         }
//                         mu *= (-dt / (ndust + 1));

//                         // get ptr to datas
//                         Tscal *ptr_loc_A  = &(local_A[0]) + mat_size_squared * loc_id;
//                         Tscal *ptr_loc_B  = &(local_B[0]) + mat_size_squared * loc_id;
//                         Tscal *ptr_loc_F  = &(local_F[0]) + mat_size_squared * loc_id;
//                         Tscal *ptr_loc_I  = &(local_I[0]) + mat_size_squared * loc_id;
//                         Tscal *ptr_loc_Id = &(local_Id[0]) + mat_size_squared * loc_id;

//                         // create mdspan(s)
//                         std::mdspan<
//                             Tscal,
//                             std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
//                             mdspan_A(ptr_loc_A, mat_size, mat_size);
//                         std::mdspan<
//                             Tscal,
//                             std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
//                             mdspan_B(ptr_loc_B, mat_size, mat_size);
//                         std::mdspan<
//                             Tscal,
//                             std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
//                             mdspan_F(ptr_loc_F, mat_size, mat_size);
//                         std::mdspan<
//                             Tscal,
//                             std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
//                             mdspan_I(ptr_loc_I, mat_size, mat_size);
//                         std::mdspan<
//                             Tscal,
//                             std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
//                             mdspan_Id(ptr_loc_Id, mat_size, mat_size);

//                         // get local Jacobian matrix

//                         get_jacobian(id_a, mdspan_A);

//                         // pre-processing step
//                         shammath::mat_set_identity<Tscal>(mdspan_Id);
//                         shammath::mat_axpy_beta<Tscal, Tscal>(-mu, mdspan_Id, dt, mdspan_A);

//                         // compute matrix exponential
//                         const i32 K_exp = 9;
//                         shammath::mat_exp<Tscal, Tscal>(
//                             K_exp, mdspan_A, mdspan_F, mdspan_B, mdspan_I, mdspan_Id, ndust + 1);

//                         // post-processing step
//                         shammath::mat_mul_scalar<Tscal>(mdspan_A, sycl::exp(mu));

//                         // use the matrix exponential to for to updates momemtum
//                         Tvec r = {0., 0., 0.}, dd = {0., 0., 0.};
//                         r += mdspan_A(0, 0) * acc_rhov_new_patch[id_a];

//                         for (auto j = 1; j < ndust + 1; j++) {
//                             r += mdspan_A(0, j) * acc_rhov_d_new_patch[id_a * ndust + (j - 1)];
//                         }

//                         dd = r - acc_rhov_new_patch[id_a];

//                         Tscal dissipation = 0, drag_work = 0;

//                         // compute work of drag terms
//                         Tscal inv_rho = 1.0 / (acc_rho_new_patch[id_a]);

//                         Tvec v_bf = inv_rho * acc_rhov_new_patch[id_a];
//                         Tvec v_af = inv_rho * r;

//                         drag_work = 0.5
//                                     * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
//                                        + dd[2] * (v_bf[2] + v_af[2]));

//                         // save gas momentum back
//                         acc_rhov_old[id_a] = r;
//                         acc_rho_old[id_a]  = acc_rho_new_patch[id_a];

//                         for (auto d_id = 1; d_id <= ndust; d_id++) {
//                             r *= 0;
//                             r += mdspan_A(d_id, 0) * acc_rhov_new_patch[id_a];

//                             for (auto j = 1; j <= ndust; j++) {

//                                 r += mdspan_A(d_id, j)
//                                      * acc_rhov_d_new_patch[id_a * ndust + (j - 1)];
//                             }

//                             dd = r - acc_rhov_d_new_patch[id_a * ndust + (d_id - 1)];

//                             inv_rho = 1.0 / (acc_rho_d_new_patch[id_a * ndust + (d_id - 1)]);

//                             v_bf = inv_rho * acc_rhov_d_new_patch[id_a * ndust + (d_id - 1)];

//                             v_af = inv_rho * r;

//                             // compute dissipaation by id-th dust
//                             dissipation
//                                 += 0.5
//                                    * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
//                                       + dd[2] * (v_bf[2] + v_af[2]));

//                             // save dust momentum back
//                             acc_rhov_d_old[id_a * ndust + (d_id - 1)] = r;
//                             acc_rho_d_old[id_a * ndust + (d_id - 1)]
//                                 = acc_rho_d_new_patch[id_a * ndust + (d_id - 1)];
//                         }

//                         // updates energy
//                         acc_rhoe_old[id_a] = acc_rhoe_new_patch[id_a]
//                                              + (1 - friction_control) * drag_work
//                                              - friction_control * dissipation;
//                     });
//             });

//             rho_new_patch.complete_event_state(e);
//             rhov_new_patch.complete_event_state(e);
//             rhoe_new_patch.complete_event_state(e);
//             rho_d_new_patch.complete_event_state(e);
//             rhov_d_new_patch.complete_event_state(e);

//             rho_old.complete_event_state(e);
//             rhov_old.complete_event_state(e);
//             rhoe_old.complete_event_state(e);
//             rho_d_old.complete_event_state(e);
//             rhov_d_old.complete_event_state(e);

//             alphas_buf.complete_event_state(e);
//         }
//     });
// }

template class shammodels::basegodunov::modules::TimeIntegratorSelfGravity<f64_3, i64_3>;
