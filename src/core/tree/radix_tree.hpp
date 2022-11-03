// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once



#include "aliases.hpp"
#include <array>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>


#include "core/patch/base/patchdata.hpp"
#include "core/sys/log.hpp"
#include "core/utils/string_utils.hpp"
#include "kernels/morton_kernels.hpp"
#include "core/sfc/morton.hpp"
#include "kernels/compute_ranges.hpp"
#include "kernels/convert_ranges.hpp"
#include "kernels/karras_alg.hpp"
#include "kernels/key_morton_sort.hpp"
#include "kernels/reduction_alg.hpp"
#include "core/utils/geometry_utils.hpp"




inline u32 get_next_pow2_val(u32 val){
    u32 val_rounded_pow = pow(2,32-__builtin_clz(val));
    if(val == pow(2,32-__builtin_clz(val)-1)){
        val_rounded_pow = val;
    }
    return val_rounded_pow;
}

template<class u_morton>
class Radix_tree_depth{

    static constexpr auto get_tree_depth = []() -> u32{
        if constexpr (std::is_same<u_morton,u32>::value){return 32;}
        if constexpr (std::is_same<u_morton,u64>::value){return 64;}
        return 0;
    };

    public : 

    static constexpr u32 tree_depth = get_tree_depth();
};






template<class u_morton,class vec3>
class Radix_Tree{
    
    Radix_Tree(){}
    
    public:

    using vec3i = typename morton_3d::morton_types<u_morton>::int_vec_repr;
    using flt = typename vec3::element_type;

    static constexpr u32 tree_depth = Radix_tree_depth<u_morton>::tree_depth;



    std::tuple<vec3,vec3> box_coord;

    u32 tree_leaf_count;
    u32 tree_internal_count;

    bool one_cell_mode = false;



    std::unique_ptr<sycl::buffer<u_morton>> buf_morton;
    std::unique_ptr<sycl::buffer<u32>> buf_particle_index_map;

    //std::vector<u32> reduc_index_map;
    std::unique_ptr<sycl::buffer<u32>> buf_reduc_index_map;

    std::unique_ptr<sycl::buffer<u_morton>> buf_tree_morton; // size = leaf cnt
    std::unique_ptr<sycl::buffer<u32>>      buf_lchild_id;   // size = internal
    std::unique_ptr<sycl::buffer<u32>>      buf_rchild_id;   // size = internal
    std::unique_ptr<sycl::buffer<u8>>       buf_lchild_flag; // size = internal
    std::unique_ptr<sycl::buffer<u8>>       buf_rchild_flag; // size = internal
    std::unique_ptr<sycl::buffer<u32>>      buf_endrange;    // size = internal

    std::unique_ptr<sycl::buffer<vec3i>>    buf_pos_min_cell;     // size = total count
    std::unique_ptr<sycl::buffer<vec3i>>    buf_pos_max_cell;     // size = total count
    std::unique_ptr<sycl::buffer<vec3>>     buf_pos_min_cell_flt; // size = total count
    std::unique_ptr<sycl::buffer<vec3>>     buf_pos_max_cell_flt; // size = total count

    inline bool is_tree_built(){
        return bool(buf_lchild_id) && bool(buf_rchild_id) && bool(buf_lchild_flag) && bool(buf_rchild_flag) && bool(buf_endrange);
    }

    inline bool are_range_int_built(){
        return bool(buf_pos_min_cell) && bool(buf_pos_max_cell);
    }

    inline bool are_range_float_built(){
        return bool(buf_pos_min_cell_flt) && bool(buf_pos_max_cell_flt);
    }


    Radix_Tree(sycl::queue & queue,std::tuple<vec3,vec3> treebox,std::unique_ptr<sycl::buffer<vec3>> & pos_buf, u32 cnt_obj, u32 reduc_level);

    

    
    std::unique_ptr<sycl::buffer<flt>> buf_cell_interact_rad; //TODO pull this one in a tree field
    
    
    
    void compute_cellvolume(sycl::queue & queue);


    

    void compute_int_boxes(sycl::queue & queue,std::unique_ptr<sycl::buffer<flt>> & int_rad_buf, flt tolerance);









    template<class T>
    class RadixTreeField{public:
        u32 nvar;
        std::unique_ptr<sycl::buffer<T>> radix_tree_field_buf;
    };

    template<class T, class LambdaComputeLeaf, class LambdaCombinator>
    inline RadixTreeField<T> compute_field(sycl::queue & queue,u32 nvar,
    
        LambdaComputeLeaf && compute_leaf, LambdaCombinator && combine) const{

        RadixTreeField<T> ret;
        ret.nvar = nvar;

        logger::debug_sycl_ln("RadixTree", "compute_field");

        ret.radix_tree_field_buf = std::make_unique<sycl::buffer<T>>(tree_internal_count + tree_leaf_count);
        sycl::range<1> range_leaf_cell{tree_leaf_count};

        queue.submit([&](sycl::handler &cgh) {
            u32 offset_leaf = tree_internal_count;

            auto tree_field = sycl::accessor{* ret.radix_tree_field_buf, cgh ,sycl::write_only, sycl::no_init};

            auto cell_particle_ids  = buf_reduc_index_map->template get_access<sycl::access::mode::read>(cgh);
            auto particle_index_map = buf_particle_index_map->template get_access<sycl::access::mode::read>(cgh);

            compute_leaf(cgh,[&](auto && lambda_loop){
                cgh.parallel_for(range_leaf_cell, [=](sycl::item<1> item) {
                    u32 gid = (u32)item.get_id(0);

                    u32 min_ids = cell_particle_ids[gid];
                    u32 max_ids = cell_particle_ids[gid + 1];


                    lambda_loop([&](auto && particle_it){
                            for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {
                                particle_it(particle_index_map[id_s]);
                            }
                        },
                        tree_field,
                        [&](){
                            return nvar*(offset_leaf + gid);
                        }
                    );

                });
            });

            
        });

        sycl::range<1> range_tree{tree_internal_count};
        auto ker_reduc_hmax = [&](sycl::handler &cgh) {
            u32 offset_leaf = tree_internal_count;

            auto tree_field = ret.radix_tree_field_buf->template get_access<sycl::access::mode::read_write>(cgh);

            auto rchild_id   = buf_rchild_id->get_access<sycl::access::mode::read>(cgh);
            auto lchild_id   = buf_lchild_id->get_access<sycl::access::mode::read>(cgh);
            auto rchild_flag = buf_rchild_flag->get_access<sycl::access::mode::read>(cgh);
            auto lchild_flag = buf_lchild_flag->get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
                u32 gid = (u32)item.get_id(0);

                u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
                u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

                combine(
                    [&](u32 nvar_id) -> T{
                        return tree_field[nvar*lid + nvar_id];
                    },
                    [&](u32 nvar_id) -> T{
                        return tree_field[nvar*rid + nvar_id];
                    },
                    tree_field,
                    [&](){
                        return nvar*(gid);
                    }
                );
            });
        };

        for (u32 i = 0; i < tree_depth; i++) {
            queue.submit(ker_reduc_hmax);
        }

        return std::move(ret);
    }


    struct CuttedTree{
        Radix_Tree<u_morton, vec3> rtree;
        std::unique_ptr<sycl::buffer<u32>> new_node_id_to_old;

        std::unique_ptr<sycl::buffer<u32>> pdat_extract_id;
    };

    CuttedTree cut_tree(sycl::queue & queue,const std::tuple<vec3,vec3> & cut_range);

    template<class T> void print_tree_field(sycl::buffer<T> & buf_field);

};









//TODO move h iter thing + multipoles to a tree field class










namespace walker {

    namespace interaction_crit {
        template<class vec3,class flt>
        inline bool sph_radix_cell_crit(vec3 xyz_a,vec3 part_a_box_min,vec3 part_a_box_max,vec3 cur_cell_box_min,vec3 cur_cell_box_max,flt box_int_sz){
            
            vec3 inter_box_b_min = cur_cell_box_min - box_int_sz;
            vec3 inter_box_b_max = cur_cell_box_max + box_int_sz;

            return 
                BBAA::cella_neigh_b(
                    part_a_box_min, part_a_box_max, 
                    cur_cell_box_min, cur_cell_box_max) ||
                BBAA::cella_neigh_b(
                    xyz_a, xyz_a,                   
                    inter_box_b_min, inter_box_b_max);
        }


        template<class vec3,class flt>
        inline bool sph_cell_cell_crit(vec3 cella_min,vec3 cella_max,vec3 cellb_min, vec3 cellb_max, flt rint_a, flt rint_b){

            vec3 inter_box_a_min = cella_min - rint_a;
            vec3 inter_box_a_max = cella_max + rint_a;

            vec3 inter_box_b_min = cellb_min - rint_b;
            vec3 inter_box_b_max = cellb_max + rint_b;

            return BBAA::cella_neigh_b(inter_box_a_min, inter_box_a_max, cellb_min,cellb_max) ||
                BBAA::cella_neigh_b(inter_box_b_min, inter_box_b_max, cella_min,cella_max) ;

        }
    }


    

    template<class u_morton,class vec3>
    class Radix_tree_accessor{public:
        sycl::accessor<u32 ,1,sycl::access::mode::read,sycl::target::device>  particle_index_map;
        sycl::accessor<u32 ,1,sycl::access::mode::read,sycl::target::device> cell_index_map;
        sycl::accessor<u32 ,1,sycl::access::mode::read,sycl::target::device>  rchild_id     ;
        sycl::accessor<u32 ,1,sycl::access::mode::read,sycl::target::device>  lchild_id     ;
        sycl::accessor<u8  ,1,sycl::access::mode::read,sycl::target::device>  rchild_flag   ;
        sycl::accessor<u8  ,1,sycl::access::mode::read,sycl::target::device>  lchild_flag   ;
        sycl::accessor<vec3,1,sycl::access::mode::read,sycl::target::device>  pos_min_cell  ;
        sycl::accessor<vec3,1,sycl::access::mode::read,sycl::target::device>  pos_max_cell  ;

        static constexpr u32 tree_depth = Radix_tree_depth<u_morton>::tree_depth;
        static constexpr u32 _nindex = 4294967295;

        u32 leaf_offset;

        
        Radix_tree_accessor(Radix_Tree< u_morton,  vec3> & rtree,sycl::handler & cgh):
            particle_index_map(rtree.buf_particle_index_map-> template get_access<sycl::access::mode::read>(cgh)),
            cell_index_map(rtree.buf_reduc_index_map-> template get_access<sycl::access::mode::read>(cgh)),
            rchild_id     (rtree.buf_rchild_id  -> template get_access<sycl::access::mode::read>(cgh)),
            lchild_id     (rtree.buf_lchild_id  -> template get_access<sycl::access::mode::read>(cgh)),
            rchild_flag   (rtree.buf_rchild_flag-> template get_access<sycl::access::mode::read>(cgh)),
            lchild_flag   (rtree.buf_lchild_flag-> template get_access<sycl::access::mode::read>(cgh)),
            pos_min_cell  (rtree.buf_pos_min_cell_flt-> template get_access<sycl::access::mode::read>(cgh)),
            pos_max_cell  (rtree.buf_pos_max_cell_flt-> template get_access<sycl::access::mode::read>(cgh)),
            leaf_offset   (rtree.tree_internal_count)
        {}
    };


    template<class Rta,class Functor_iter>
    inline void iter_object_in_cell(const Rta &acc,const u32 & cell_id, Functor_iter &&func_it){
        // loop on particle indexes
        uint min_ids = acc.cell_index_map[cell_id     -acc.leaf_offset];
        uint max_ids = acc.cell_index_map[cell_id + 1 -acc.leaf_offset];

        for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

            //recover old index before morton sort
            uint id_b = acc.particle_index_map[id_s];

            //iteration function
            func_it(id_b);

        }
        

        /*
        std::array<u32, 16> stack_run;

        u32 run_cursor = 16;

        auto is_stack_full = [&]() -> bool{
            return run_cursor == 0;
        };

        auto is_stack_not_empty = [&]() -> bool{
            return run_cursor < 16;
        };

        auto push_stack = [&](u32 val){
            run_cursor --;
            stack_run[run_cursor] = val;
        };

        auto pop_stack = [&]() -> u32 {
            u32 v = stack_run[run_cursor];
            run_cursor ++;
            return v;
        };

        auto empty_stack = [&](){
            while (is_stack_not_empty()) {
                func_it(pop_stack());
            }
        };

        for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {
            uint id_b = acc.particle_index_map[id_s];

            if(is_stack_full()){
                empty_stack();
            }
            
            push_stack(id_b);

        }

        empty_stack();
        */
        
    }


    template <class Rta, class Functor_int_cd, class Functor_iter, class Functor_iter_excl>
    inline void rtree_for_cell(const Rta &acc, Functor_int_cd &&func_int_cd, Functor_iter &&func_it, Functor_iter_excl &&func_excl) {
        u32 stack_cursor = Rta::tree_depth - 1;
        std::array<u32, Rta::tree_depth> id_stack;
        id_stack[stack_cursor] = 0;

        while (stack_cursor < Rta::tree_depth) {

            u32 current_node_id    = id_stack[stack_cursor];
            id_stack[stack_cursor] = Rta::_nindex;
            stack_cursor++;

            bool cur_id_valid = func_int_cd(current_node_id);

            if (cur_id_valid) {

                // leaf and cell can interact
                if (current_node_id >= acc.leaf_offset) {

                    func_it(current_node_id);

                    // can interact not leaf => stack
                } else {

                    u32 lid = acc.lchild_id[current_node_id] + acc.leaf_offset * acc.lchild_flag[current_node_id];
                    u32 rid = acc.rchild_id[current_node_id] + acc.leaf_offset * acc.rchild_flag[current_node_id];

                    id_stack[stack_cursor - 1] = rid;
                    stack_cursor--;

                    id_stack[stack_cursor - 1] = lid;
                    stack_cursor--;
                }
            } else {
                // grav
                func_excl(current_node_id);
            }
        }
    }

    template <class Rta, class Functor_int_cd, class Functor_iter, class Functor_iter_excl>
    inline void rtree_for(const Rta &acc, Functor_int_cd &&func_int_cd, Functor_iter &&func_it, Functor_iter_excl &&func_excl) {
        u32 stack_cursor = Rta::tree_depth - 1;
        std::array<u32, Rta::tree_depth> id_stack;
        id_stack[stack_cursor] = 0;

        while (stack_cursor < Rta::tree_depth) {

            u32 current_node_id    = id_stack[stack_cursor];
            id_stack[stack_cursor] = Rta::_nindex;
            stack_cursor++;

            bool cur_id_valid = func_int_cd(current_node_id);

            if (cur_id_valid) {

                // leaf and can interact => force
                if (current_node_id >= acc.leaf_offset) {

                    // loop on particle indexes
                    //uint min_ids = acc.cell_index_map[current_node_id     -acc.leaf_offset];
                    //uint max_ids = acc.cell_index_map[current_node_id + 1 -acc.leaf_offset];
                    //
                    //for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {
                    //
                    //    //recover old index before morton sort
                    //    uint id_b = acc.particle_index_map[id_s];
                    //
                    //    //iteration function
                    //    func_it(id_b);
                    //}

                    iter_object_in_cell(acc, current_node_id, func_it);

                    // can interact not leaf => stack
                } else {

                    u32 lid = acc.lchild_id[current_node_id] + acc.leaf_offset * acc.lchild_flag[current_node_id];
                    u32 rid = acc.rchild_id[current_node_id] + acc.leaf_offset * acc.rchild_flag[current_node_id];

                    id_stack[stack_cursor - 1] = rid;
                    stack_cursor--;

                    id_stack[stack_cursor - 1] = lid;
                    stack_cursor--;
                }
            } else {
                // grav
                func_excl(current_node_id);
            }
        }
    }

    template <class Rta, class Functor_int_cd, class Functor_iter, class Functor_iter_excl, class arr_type>
    inline void rtree_for_fill_cache(Rta &acc,arr_type & cell_cache, Functor_int_cd &&func_int_cd) {

        constexpr u32 cache_sz = cell_cache.size();
        u32 cache_pos = 0;

        auto push_in_cache = [&cell_cache,&cache_pos](u32 id){
            cell_cache[cache_pos] = id;
            cache_pos ++;
        };

        u32 stack_cursor = Rta::tree_depth - 1;
        std::array<u32, Rta::tree_depth> id_stack;
        id_stack[stack_cursor] = 0;

        auto get_el_cnt_in_stack = [&]() -> u32{
            return Rta::tree_depth - stack_cursor;
        };

        while ((stack_cursor < Rta::tree_depth) && (cache_pos + get_el_cnt_in_stack < cache_sz)) {

            u32 current_node_id    = id_stack[stack_cursor];
            id_stack[stack_cursor] = Rta::_nindex;
            stack_cursor++;

            bool cur_id_valid = func_int_cd(current_node_id);

            if (cur_id_valid) {

                // leaf and can interact => force
                if (current_node_id >= acc.leaf_offset) {

                    //can interact => add to cache
                    push_in_cache(current_node_id);

                    // can interact not leaf => stack
                } else {

                    u32 lid = acc.lchild_id[current_node_id] + acc.leaf_offset * acc.lchild_flag[current_node_id];
                    u32 rid = acc.rchild_id[current_node_id] + acc.leaf_offset * acc.rchild_flag[current_node_id];

                    id_stack[stack_cursor - 1] = rid;
                    stack_cursor--;

                    id_stack[stack_cursor - 1] = lid;
                    stack_cursor--;
                }
            } else {
                // grav
                //.....
            }
        }

        while (stack_cursor < Rta::tree_depth) {
            u32 current_node_id    = id_stack[stack_cursor];
            id_stack[stack_cursor] = Rta::_nindex;
            stack_cursor++;
            push_in_cache(current_node_id);
        }

        if(cache_pos < cache_sz){
            push_in_cache(u32_max);
        }
    }

    template <class Rta, class Functor_int_cd, class Functor_iter, class Functor_iter_excl, class arr_type>
    inline void rtree_for(Rta &acc,arr_type & cell_cache, Functor_int_cd &&func_int_cd, Functor_iter &&func_it) {

        constexpr u32 cache_sz = cell_cache.size();

        std::array<u32, Rta::tree_depth> id_stack;

        auto walk_step = [&](u32 start_id){
            u32 stack_cursor = Rta::tree_depth - 1;
            id_stack[stack_cursor] = start_id;

            while (stack_cursor < Rta::tree_depth) {

                u32 current_node_id    = id_stack[stack_cursor];
                id_stack[stack_cursor] = Rta::_nindex;
                stack_cursor++;

                bool cur_id_valid = func_int_cd(current_node_id);

                if (cur_id_valid) {

                    // leaf and can interact => force
                    if (current_node_id >= acc.leaf_offset) {

                        // loop on particle indexes
                        uint min_ids = acc.cell_index_map[current_node_id     -acc.leaf_offset];
                        uint max_ids = acc.cell_index_map[current_node_id + 1 -acc.leaf_offset];

                        for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                            //recover old index before morton sort
                            uint id_b = acc.particle_index_map[id_s];

                            //iteration function
                            func_it(id_b);
                        }

                        // can interact not leaf => stack
                    } else {

                        u32 lid = acc.lchild_id[current_node_id] + acc.leaf_offset * acc.lchild_flag[current_node_id];
                        u32 rid = acc.rchild_id[current_node_id] + acc.leaf_offset * acc.rchild_flag[current_node_id];

                        id_stack[stack_cursor - 1] = rid;
                        stack_cursor--;

                        id_stack[stack_cursor - 1] = lid;
                        stack_cursor--;
                    }
                } else {
                    // grav
                    //...
                }
            }
        };

        for (u32 cache_pos = 0; cache_pos < cache_sz && cell_cache[cache_pos] != u32_max; cache_pos ++) {
            walk_step(cache_pos);
        }

        
    }

    



}