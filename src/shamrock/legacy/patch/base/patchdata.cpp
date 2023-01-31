// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file patchdata.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief implementation of PatchData related functions
 * @version 0.1
 * @date 2022-02-28
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "patchdata.hpp"
#include "aliases.hpp"
#include "patchdata_field.hpp"
#include "shamsys/legacy/mpi_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamrock/legacy/utils/geometry_utils.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>



void PatchData::init_fields(){

    pdl.for_each_field_any([&](auto & field){
        using f_t = typename std::remove_reference<decltype(field)>::type;
        using base_t = typename f_t::field_T;

        fields.push_back(PatchDataField<base_t>(field.name,field.nvar));

    });

}



u64 patchdata_isend(PatchData &p, std::vector<PatchDataMpiRequest> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm) {

    rq_lst.resize(rq_lst.size()+1);
    PatchDataMpiRequest & ref = rq_lst[rq_lst.size()-1];

    u64 total_data_transf = 0;


    p.for_each_field_any([&](auto & field){
        using base_t = typename std::remove_reference<decltype(field)>::type::Field_type;
        total_data_transf += patchdata_field::isend(field,ref.get_field_list<base_t>(), rank_dest, tag, comm);
    });


    return total_data_transf;
}




u64 patchdata_irecv_probe(PatchData & pdat, std::vector<PatchDataMpiRequest> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){

    rq_lst.resize(rq_lst.size()+1);
    auto & ref = rq_lst[rq_lst.size()-1];

    u64 total_data_transf = 0;

    pdat.for_each_field_any([&](auto & field){
        using base_t = typename std::remove_reference<decltype(field)>::type::Field_type;
        total_data_transf += patchdata_field::irecv_probe(field, ref.get_field_list<base_t>(), rank_source, tag, comm);
    });

    return total_data_transf;

}


PatchData patchdata_gen_dummy_data(PatchDataLayout & pdl, std::mt19937& eng){

    std::uniform_int_distribution<u64> distu64(1,6000);

    u32 num_part = distu64(eng);

    PatchData pdat(pdl);

    pdat.for_each_field_any([&](auto & field){
        field.gen_mock_data(num_part,eng);
    });

    return pdat;
}


bool patch_data_check_match(PatchData& p1, PatchData& p2){

    return p1 == p2;

}



void PatchData::extract_element(u32 pidx, PatchData & out_pdat){

    for(u32 idx = 0; idx < fields.size(); idx++){

        std::visit([&](auto & field, auto & out_field) {

            using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
            using t2 = typename std::remove_reference<decltype(out_field)>::type::Field_type;

            if constexpr (std::is_same<t1, t2>::value){
                field.extract_element(pidx,out_field);
            }else{  
                throw std::invalid_argument("missmatch");
            }

        }, fields[idx], out_pdat.fields[idx]);

    }

}

void PatchData::insert_elements(PatchData & pdat){


    for(u32 idx = 0; idx < fields.size(); idx++){

        std::visit([&](auto & field, auto & out_field) {

            using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
            using t2 = typename std::remove_reference<decltype(out_field)>::type::Field_type;

            if constexpr (std::is_same<t1, t2>::value){
                field.insert(out_field);
            }else{  
                throw std::invalid_argument("missmatch");
            }

        }, fields[idx], pdat.fields[idx]);

    }

}

void PatchData::overwrite(PatchData &pdat, u32 obj_cnt){
    
    for(u32 idx = 0; idx < fields.size(); idx++){

        std::visit([&](auto & field, auto & out_field) {

            using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
            using t2 = typename std::remove_reference<decltype(out_field)>::type::Field_type;

            if constexpr (std::is_same<t1, t2>::value){
                field.overwrite(out_field,obj_cnt);
            }else{  
                throw std::invalid_argument("missmatch");
            }

        }, fields[idx], pdat.fields[idx]);

    }
}



void PatchData::resize(u32 new_obj_cnt){

    for(auto & field_var : fields){
        std::visit([&](auto & field){
            field.resize(new_obj_cnt);
        },field_var);
    }

}



void PatchData::append_subset_to(sycl::buffer<u32> & idxs, u32 sz, PatchData & pdat) const {


    for(u32 idx = 0; idx < fields.size(); idx++){

        std::visit([&](auto & field, auto & out_field) {

            using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
            using t2 = typename std::remove_reference<decltype(out_field)>::type::Field_type;

            if constexpr (std::is_same<t1, t2>::value){
                field.append_subset_to(idxs, sz, out_field);
            }else{  
                throw std::invalid_argument("missmatch");
            }

        }, fields[idx], pdat.fields[idx]);

    }
}

void PatchData::append_subset_to(std::vector<u32> & idxs, PatchData &pdat) const {

    for(u32 idx = 0; idx < fields.size(); idx++){

        std::visit([&](auto & field, auto & out_field) {

            using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
            using t2 = typename std::remove_reference<decltype(out_field)>::type::Field_type;

            if constexpr (std::is_same<t1, t2>::value){
                field.append_subset_to(idxs, out_field);
            }else{  
                throw std::invalid_argument("missmatch");
            }

        }, fields[idx], pdat.fields[idx]);

    }

}

template<>
void PatchData::split_patchdata<f32_3>(
    PatchData &pd0, PatchData &pd1, PatchData &pd2, PatchData &pd3, PatchData &pd4, PatchData &pd5, PatchData &pd6, PatchData &pd7, 
    f32_3 bmin_p0, f32_3 bmin_p1, f32_3 bmin_p2, f32_3 bmin_p3, f32_3 bmin_p4, f32_3 bmin_p5, f32_3 bmin_p6, f32_3 bmin_p7, 
    f32_3 bmax_p0, f32_3 bmax_p1, f32_3 bmax_p2, f32_3 bmax_p3, f32_3 bmax_p4, f32_3 bmax_p5, f32_3 bmax_p6, f32_3 bmax_p7){

    
    PatchDataField<f32_3 >* pval = std::get_if<PatchDataField<f32_3 >>(&fields[0]);

    if(!pval){
        throw std::invalid_argument("the main field should be at id 0");
    }

    PatchDataField<f32_3> & xyz = * pval;

    auto get_vec_idx = [&](f32_3 vmin, f32_3 vmax) -> std::vector<u32> {
        return xyz.get_elements_with_range(
            [&](f32_3 val,f32_3 vmin, f32_3 vmax){
                return BBAA::is_particle_in_patch<f32_3>(val, vmin,vmax);
            },
            vmin,vmax
        );
    };

    std::vector<u32> idx_p0 = get_vec_idx(bmin_p0,bmax_p0);
    std::vector<u32> idx_p1 = get_vec_idx(bmin_p1,bmax_p1);
    std::vector<u32> idx_p2 = get_vec_idx(bmin_p2,bmax_p2);
    std::vector<u32> idx_p3 = get_vec_idx(bmin_p3,bmax_p3);
    std::vector<u32> idx_p4 = get_vec_idx(bmin_p4,bmax_p4);
    std::vector<u32> idx_p5 = get_vec_idx(bmin_p5,bmax_p5);
    std::vector<u32> idx_p6 = get_vec_idx(bmin_p6,bmax_p6);
    std::vector<u32> idx_p7 = get_vec_idx(bmin_p7,bmax_p7);

    u32 el_cnt_new = idx_p0.size()+
                    idx_p1.size()+
                    idx_p2.size()+
                    idx_p3.size()+
                    idx_p4.size()+
                    idx_p5.size()+
                    idx_p6.size()+
                    idx_p7.size();

    if(get_obj_cnt() != el_cnt_new){

        f32_3 vmin = sycl::fmin(bmin_p0,bmin_p1);
        vmin = sycl::fmin(vmin,bmin_p2);
        vmin = sycl::fmin(vmin,bmin_p3);
        vmin = sycl::fmin(vmin,bmin_p4);
        vmin = sycl::fmin(vmin,bmin_p5);
        vmin = sycl::fmin(vmin,bmin_p6);
        vmin = sycl::fmin(vmin,bmin_p7);

        f32_3 vmax = sycl::fmax(bmax_p0,bmax_p1);
        vmax = sycl::fmax(vmax,bmax_p2);
        vmax = sycl::fmax(vmax,bmax_p3);
        vmax = sycl::fmax(vmax,bmax_p4);
        vmax = sycl::fmax(vmax,bmax_p5);
        vmax = sycl::fmax(vmax,bmax_p6);
        vmax = sycl::fmax(vmax,bmax_p7);

        xyz.check_err_range(
            [&](f32_3 val,f32_3 vmin, f32_3 vmax){
                return BBAA::is_particle_in_patch<f32_3>(val, vmin,vmax);
            },
            vmin,vmax);

        throw ShamrockSyclException("issue in the patch split : new element count doesn't match the old one");
    }

    //TODO create a extract subpatch function

    append_subset_to(idx_p0, pd0);
    append_subset_to(idx_p1, pd1);
    append_subset_to(idx_p2, pd2);
    append_subset_to(idx_p3, pd3);
    append_subset_to(idx_p4, pd4);
    append_subset_to(idx_p5, pd5);
    append_subset_to(idx_p6, pd6);
    append_subset_to(idx_p7, pd7);

}


template<>
void PatchData::split_patchdata<f64_3>(
    PatchData &pd0, PatchData &pd1, PatchData &pd2, PatchData &pd3, PatchData &pd4, PatchData &pd5, PatchData &pd6, PatchData &pd7, 
    f64_3 bmin_p0, f64_3 bmin_p1, f64_3 bmin_p2, f64_3 bmin_p3, f64_3 bmin_p4, f64_3 bmin_p5, f64_3 bmin_p6, f64_3 bmin_p7, 
    f64_3 bmax_p0, f64_3 bmax_p1, f64_3 bmax_p2, f64_3 bmax_p3, f64_3 bmax_p4, f64_3 bmax_p5, f64_3 bmax_p6, f64_3 bmax_p7){

    PatchDataField<f64_3 >* pval = std::get_if<PatchDataField<f64_3 >>(&fields[0]);

    if(!pval){
        throw std::invalid_argument("the main field should be at id 0");
    }

    PatchDataField<f64_3> & xyz = * pval;

    auto get_vec_idx = [&](f64_3 vmin, f64_3 vmax) -> std::vector<u32> {
        return xyz.get_elements_with_range(
            [&](f64_3 val,f64_3 vmin, f64_3 vmax){
                return BBAA::is_particle_in_patch<f64_3>(val, vmin,vmax);
            },
            vmin,vmax
        );
    };

    std::vector<u32> idx_p0 = get_vec_idx(bmin_p0,bmax_p0);
    std::vector<u32> idx_p1 = get_vec_idx(bmin_p1,bmax_p1);
    std::vector<u32> idx_p2 = get_vec_idx(bmin_p2,bmax_p2);
    std::vector<u32> idx_p3 = get_vec_idx(bmin_p3,bmax_p3);
    std::vector<u32> idx_p4 = get_vec_idx(bmin_p4,bmax_p4);
    std::vector<u32> idx_p5 = get_vec_idx(bmin_p5,bmax_p5);
    std::vector<u32> idx_p6 = get_vec_idx(bmin_p6,bmax_p6);
    std::vector<u32> idx_p7 = get_vec_idx(bmin_p7,bmax_p7);

    //TODO create a extract subpatch function

    append_subset_to(idx_p0, pd0);
    append_subset_to(idx_p1, pd1);
    append_subset_to(idx_p2, pd2);
    append_subset_to(idx_p3, pd3);
    append_subset_to(idx_p4, pd4);
    append_subset_to(idx_p5, pd5);
    append_subset_to(idx_p6, pd6);
    append_subset_to(idx_p7, pd7);

}