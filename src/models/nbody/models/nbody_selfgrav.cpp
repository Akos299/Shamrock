
#include "nbody_selfgrav.hpp"
#include "core/patch/utility/serialpatchtree.hpp"
#include "core/tree/radix_tree.hpp"
#include "runscript/shamrockapi.hpp"

#include "models/generic/algs/integrators_utils.hpp"

#include "core/patch/comm/patch_object_mover.hpp"


const std::string console_tag = "[NBodySelfGrav] ";


template<class flt> 
void models::nbody::Nbody_SelfGrav<flt>::check_valid(){


    if (cfl_force < 0) {
        throw ShamAPIException(console_tag + "cfl force not set");
    }

    if (gpart_mass < 0) {
        throw ShamAPIException(console_tag + "particle mass not set");
    }
}

template<class flt> 
void models::nbody::Nbody_SelfGrav<flt>::init(){

}









template<class flt,class vec3>
void sycl_move_parts(sycl::queue &queue, u32 npart, flt dt, std::unique_ptr<sycl::buffer<vec3>> &buf_xyz,
                                        std::unique_ptr<sycl::buffer<vec3>> &buf_vxyz) {

    sycl::range<1> range_npart{npart};

    auto ker_predict_step = [&](sycl::handler &cgh) {
        auto acc_xyz  = buf_xyz->template get_access<sycl::access::mode::read_write>(cgh);
        auto acc_vxyz = buf_vxyz->template get_access<sycl::access::mode::read_write>(cgh);

        // Executing kernel
        cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            vec3 &vxyz = acc_vxyz[item];

            acc_xyz[item] = acc_xyz[item] + dt * vxyz;

        });
    };

    queue.submit(ker_predict_step);
}


template<class vec3>
void sycl_position_modulo(sycl::queue &queue, u32 npart, std::unique_ptr<sycl::buffer<vec3>> &buf_xyz,
                                std::tuple<vec3, vec3> box) {

    sycl::range<1> range_npart{npart};

    auto ker_predict_step = [&](sycl::handler &cgh) {
        auto xyz = buf_xyz->template get_access<sycl::access::mode::read_write>(cgh);

        vec3 box_min = std::get<0>(box);
        vec3 box_max = std::get<1>(box);
        vec3 delt    = box_max - box_min;

        // Executing kernel
        cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            vec3 r = xyz[gid] - box_min;

            r = sycl::fmod(r, delt);
            r += delt;
            r = sycl::fmod(r, delt);
            r += box_min;

            xyz[gid] = r;
        });
    };

    queue.submit(ker_predict_step);
}



template<class flt> 
f64 models::nbody::Nbody_SelfGrav<flt>::evolve(PatchScheduler &sched, f64 old_time, f64 target_time){

    check_valid();

    logger::info_ln("NBodySelfGrav", "evolve t=",old_time);


    //Stepper stepper(sched,periodic_bc,htol_up_tol,htol_up_iter,gpart_mass);

    const u32 ixyz      = sched.pdl.get_field_idx<vec3>("xyz");
    const u32 ivxyz     = sched.pdl.get_field_idx<vec3>("vxyz");
    const u32 iaxyz     = sched.pdl.get_field_idx<vec3>("axyz");
    const u32 iaxyz_old = sched.pdl.get_field_idx<vec3>("axyz_old");

    //const u32 ihpart    = sched.pdl.get_field_idx<flt>("hpart");

    //PatchComputeField<f32> pressure_field;


    auto lambda_update_time = [&](sycl::queue&  queue, PatchData& pdat, sycl::range<1> range_npart ,flt hdt){
            
        sycl::buffer<vec3> & vxyz =  * pdat.get_field<vec3>(ivxyz).get_buf();
        sycl::buffer<vec3> & axyz =  * pdat.get_field<vec3>(iaxyz).get_buf();

        field_advance_time(queue, vxyz, axyz, range_npart, hdt);

    };

    auto lambda_swap_der = [&](sycl::queue&  queue, PatchData& pdat, sycl::range<1> range_npart ){
        auto ker_predict_step = [&](sycl::handler &cgh) {
            auto acc_axyz = pdat.get_field<vec3>(iaxyz).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);
            auto acc_axyz_old = pdat.get_field<vec3>(iaxyz_old).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);

            // Executing kernel
            cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                vec3 axyz     = acc_axyz[item];
                vec3 axyz_old = acc_axyz_old[item];

                acc_axyz[item]     = axyz_old;
                acc_axyz_old[item] = axyz;

            });
        };

        queue.submit(ker_predict_step);
    };

    auto lambda_correct = [&](sycl::queue&  queue, PatchData& buf, sycl::range<1> range_npart ,flt hdt){
            
        auto ker_corect_step = [&](sycl::handler &cgh) {
            auto acc_vxyz     = buf.get_field<vec3>(ivxyz).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);
            auto acc_axyz     = buf.get_field<vec3>(iaxyz).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);
            auto acc_axyz_old = buf.get_field<vec3>(iaxyz_old).get_buf()->template get_access<sycl::access::mode::read_write>(cgh);

            // Executing kernel
            cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                //u32 gid = (u32)item.get_id();
                //
                //vec3 &vxyz     = acc_vxyz[item];
                //vec3 &axyz     = acc_axyz[item];
                //vec3 &axyz_old = acc_axyz_old[item];

                // v^* = v^{n + 1/2} + dt/2 a^n
                acc_vxyz[item] = acc_vxyz[item] + (hdt) * (acc_axyz[item] - acc_axyz_old[item]);
            });
        };

        queue.submit(ker_corect_step);
    };



    auto leapfrog_lambda = [&](flt old_time, bool do_force, bool do_corrector) -> flt{

        const u32 ixyz      = sched.pdl.get_field_idx<vec3>("xyz");
        const u32 ivxyz     = sched.pdl.get_field_idx<vec3>("vxyz");
        const u32 iaxyz     = sched.pdl.get_field_idx<vec3>("axyz");
        const u32 iaxyz_old = sched.pdl.get_field_idx<vec3>("axyz_old");



        logger::info_ln("NBodyleapfrog", "step t=",old_time, "do_force =",do_force, "do_corrector =",do_corrector);





        //Init serial patch tree
        SerialPatchTree<vec3> sptree(sched.patch_tree, sched.get_box_tranform<vec3>());
        sptree.attach_buf();

        //compute cfl
        flt cfl_val = 1e-3;




        //compute dt step

        flt dt_cur = cfl_val;

        logger::info_ln("SPHLeapfrog", "current dt  :",dt_cur);

        //advance time
        flt step_time = old_time;
        step_time += dt_cur;

        //leapfrog predictor
        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {

            logger::debug_ln("SPHLeapfrog", "patch : n°",id_patch,"->","predictor");

            lambda_update_time(sycl_handler::get_compute_queue(),pdat,sycl::range<1> {pdat.get_obj_cnt()},dt_cur/2);

            sycl_move_parts(sycl_handler::get_compute_queue(), pdat.get_obj_cnt(), dt_cur,
                                              pdat.get_field<vec3>(ixyz).get_buf(), pdat.get_field<vec3>(ivxyz).get_buf());

            lambda_update_time(sycl_handler::get_compute_queue(),pdat,sycl::range<1> {pdat.get_obj_cnt()},dt_cur/2);


            logger::debug_ln("SPHLeapfrog", "patch : n°",id_patch,"->","dt fields swap");

            lambda_swap_der(sycl_handler::get_compute_queue(),pdat,sycl::range<1> {pdat.get_obj_cnt()});

            if (periodic_bc) {//TODO generalise position modulo in the scheduler
                sycl_position_modulo(sycl_handler::get_compute_queue(), pdat.get_obj_cnt(),
                                               pdat.get_field<vec3>(ixyz).get_buf(), sched.get_box_volume<vec3>());
            }
        });




        //move particles between patches
        logger::debug_ln("SPHLeapfrog", "particle reatribution");
        reatribute_particles(sched, sptree, periodic_bc);





        constexpr u32 reduc_level = 5;

        //make trees
        auto tgen_trees = timings::start_timer("radix tree gen", timings::sycl);
        std::unordered_map<u64, std::unique_ptr<Radix_Tree<u_morton, vec3>>> radix_trees;

        sched.for_each_patch_data([&](u64 id_patch, Patch & cur_p, PatchData & pdat) {
            logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","making Radix Tree");

            if (pdat.is_empty()){
                logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","is empty skipping tree build");
            }else{

                auto & buf_xyz = pdat.get_field<vec3>(ixyz).get_buf();

                std::tuple<vec3, vec3> box = sched.patch_data.sim_box.get_box<flt>(cur_p);

                // radix tree computation
                radix_trees[id_patch] = std::make_unique<Radix_Tree<u_morton, vec3>>(sycl_handler::get_compute_queue(), box,
                                                                                    buf_xyz,pdat.get_obj_cnt(),reduc_level);
            }
                
        });



        /*
        sched.for_each_patch([&](u64 id_patch, Patch  cur_p) {
            logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","compute radix tree cell volumes");
            if (merge_pdat.at(id_patch).or_element_cnt == 0)
                logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","is empty skipping tree volumes step");

            radix_trees[id_patch]->compute_cellvolume(sycl_handler::get_compute_queue());
        });

        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","compute Radix Tree interaction boxes");
            if (merge_pdat.at(id_patch).or_element_cnt == 0)
                logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","is empty skipping interaction box compute");

            PatchData & mpdat = merge_pdat.at(id_patch).data;

            auto & buf_h = mpdat.get_field<flt>(ihpart).get_buf();

            radix_trees[id_patch]->compute_int_boxes(sycl_handler::get_compute_queue(), buf_h, htol_up_tol);
        });

        */

        sycl_handler::get_compute_queue().wait();
        tgen_trees.stop();




        //make interfaces



        //force



        //leapfrog predictor
        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {

            logger::debug_ln("SPHLeapfrog", "patch : n°",id_patch,"->","corrector");

            lambda_correct(sycl_handler::get_compute_queue(),pdat,sycl::range<1> {pdat.get_obj_cnt()},dt_cur/2);

        });


        return step_time;

    };
    







    f64 step_time = leapfrog_lambda(old_time,true,true);













    return step_time;
}


template<class flt> 
void models::nbody::Nbody_SelfGrav<flt>::dump(std::string prefix){
    std::cout << "dump : "<< prefix << std::endl;
}

template<class flt> 
void models::nbody::Nbody_SelfGrav<flt>::restart_dump(std::string prefix){
    std::cout << "restart dump : "<< prefix << std::endl;
}

template<class flt> 
void models::nbody::Nbody_SelfGrav<flt>::close(){
    
}



template class models::nbody::Nbody_SelfGrav<f32>;

