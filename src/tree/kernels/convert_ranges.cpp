// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "convert_ranges.hpp"


template<>
void sycl_convert_cell_range<u32,f32_3>(sycl::queue & queue,

    u32 leaf_cnt , 
    u32 internal_cnt ,
    f32_3 bounding_box_min,
    f32_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u16_3>> & buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<u16_3>> & buf_pos_max_cell,
    std::unique_ptr<sycl::buffer<f32_3>> & buf_pos_min_cell_flt,
    std::unique_ptr<sycl::buffer<f32_3>> & buf_pos_max_cell_flt){

    using f3_xyzh = f32_3;

    sycl::range<1> range_cell{leaf_cnt + internal_cnt};


    auto ker_convert_cell_ranges = [&](sycl::handler &cgh) {
        f3_xyzh b_box_min = bounding_box_min;
        f3_xyzh b_box_max = bounding_box_max;

        auto pos_min_cell = buf_pos_min_cell->get_access<sycl::access::mode::read>(cgh);
        auto pos_max_cell = buf_pos_max_cell->get_access<sycl::access::mode::read>(cgh);

        // auto pos_min_cell_flt = buf_pos_min_cell_flt->get_access<sycl::access::mode::discard_write>(cgh);
        // auto pos_max_cell_flt = buf_pos_max_cell_flt->get_access<sycl::access::mode::discard_write>(cgh);

        auto pos_min_cell_flt = sycl::accessor { *buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};
        auto pos_max_cell_flt = sycl::accessor { *buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};

        cgh.parallel_for<class Convert_cell_range_u32_f32>(
            range_cell, [=](sycl::item<1> item) {

                u32 gid = (u32) item.get_id(0);

                
                    pos_min_cell_flt[gid].s0() = f32(pos_min_cell[gid].s0())*(1/1024.f);
                    pos_max_cell_flt[gid].s0() = f32(pos_max_cell[gid].s0())*(1/1024.f);

                    pos_min_cell_flt[gid].s1() = f32(pos_min_cell[gid].s1())*(1/1024.f);
                    pos_max_cell_flt[gid].s1() = f32(pos_max_cell[gid].s1())*(1/1024.f);

                    pos_min_cell_flt[gid].s2() = f32(pos_min_cell[gid].s2())*(1/1024.f);
                    pos_max_cell_flt[gid].s2() = f32(pos_max_cell[gid].s2())*(1/1024.f);
                

                pos_min_cell_flt[gid] *= b_box_max - b_box_min;
                pos_min_cell_flt[gid] += b_box_min;
                
                pos_max_cell_flt[gid] *= b_box_max - b_box_min;
                pos_max_cell_flt[gid] += b_box_min;

            }
        );

    };

    queue.submit(ker_convert_cell_ranges);
}


template<>
void sycl_convert_cell_range<u64,f32_3>(sycl::queue & queue,

    u32 leaf_cnt , 
    u32 internal_cnt,
    f32_3 bounding_box_min,
    f32_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u32_3>> & buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<u32_3>> & buf_pos_max_cell,
    std::unique_ptr<sycl::buffer<f32_3>> & buf_pos_min_cell_flt,
    std::unique_ptr<sycl::buffer<f32_3>> & buf_pos_max_cell_flt ){

    using f3_xyzh = f32_3;

    sycl::range<1> range_cell{leaf_cnt + internal_cnt};

    auto ker_convert_cell_ranges = [&](sycl::handler &cgh) {
        f3_xyzh b_box_min = bounding_box_min;
        f3_xyzh b_box_max = bounding_box_max;

        auto pos_min_cell = buf_pos_min_cell->get_access<sycl::access::mode::read>(cgh);
        auto pos_max_cell = buf_pos_max_cell->get_access<sycl::access::mode::read>(cgh);

        // auto pos_min_cell_flt = buf_pos_min_cell_flt->get_access<sycl::access::mode::discard_write>(cgh);
        // auto pos_max_cell_flt = buf_pos_max_cell_flt->get_access<sycl::access::mode::discard_write>(cgh);

        auto pos_min_cell_flt = sycl::accessor { *buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};
        auto pos_max_cell_flt = sycl::accessor { *buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};

        cgh.parallel_for<class Convert_cell_range_u64_f32>(
            range_cell, [=](sycl::item<1> item) {

                u32 gid = (u32) item.get_id(0);

                
                pos_min_cell_flt[gid].s0() = f32(pos_min_cell[gid].s0())*(1/2097152.f);
                pos_max_cell_flt[gid].s0() = f32(pos_max_cell[gid].s0())*(1/2097152.f);

                pos_min_cell_flt[gid].s1() = f32(pos_min_cell[gid].s1())*(1/2097152.f);
                pos_max_cell_flt[gid].s1() = f32(pos_max_cell[gid].s1())*(1/2097152.f);

                pos_min_cell_flt[gid].s2() = f32(pos_min_cell[gid].s2())*(1/2097152.f);
                pos_max_cell_flt[gid].s2() = f32(pos_max_cell[gid].s2())*(1/2097152.f);
                

                pos_min_cell_flt[gid] *= b_box_max - b_box_min;
                pos_min_cell_flt[gid] += b_box_min;
                
                pos_max_cell_flt[gid] *= b_box_max - b_box_min;
                pos_max_cell_flt[gid] += b_box_min;

            }
        );

    };

    queue.submit(ker_convert_cell_ranges);
}


template<>
void sycl_convert_cell_range<u32,f64_3>(sycl::queue & queue,

    u32 leaf_cnt , 
    u32 internal_cnt,
    f64_3 bounding_box_min,
    f64_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u16_3>> & buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<u16_3>> & buf_pos_max_cell,
    std::unique_ptr<sycl::buffer<f64_3>> & buf_pos_min_cell_flt,
    std::unique_ptr<sycl::buffer<f64_3>> & buf_pos_max_cell_flt ){

    using f3_xyzh = f64_3;

    sycl::range<1> range_cell{leaf_cnt + internal_cnt};

    auto ker_convert_cell_ranges = [&](sycl::handler &cgh) {
        f3_xyzh b_box_min = bounding_box_min;
        f3_xyzh b_box_max = bounding_box_max;

        auto pos_min_cell = buf_pos_min_cell->get_access<sycl::access::mode::read>(cgh);
        auto pos_max_cell = buf_pos_max_cell->get_access<sycl::access::mode::read>(cgh);

        // auto pos_min_cell_flt = buf_pos_min_cell_flt->get_access<sycl::access::mode::discard_write>(cgh);
        // auto pos_max_cell_flt = buf_pos_max_cell_flt->get_access<sycl::access::mode::discard_write>(cgh);

        auto pos_min_cell_flt = sycl::accessor { *buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};
        auto pos_max_cell_flt = sycl::accessor { *buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};

        cgh.parallel_for<class Convert_cell_range_u32_f64>(
            range_cell, [=](sycl::item<1> item) {

                u32 gid = (u32) item.get_id(0);

                
                pos_min_cell_flt[gid].s0() = f64(pos_min_cell[gid].s0())*(1/1024.);
                pos_max_cell_flt[gid].s0() = f64(pos_max_cell[gid].s0())*(1/1024.);

                pos_min_cell_flt[gid].s1() = f64(pos_min_cell[gid].s1())*(1/1024.);
                pos_max_cell_flt[gid].s1() = f64(pos_max_cell[gid].s1())*(1/1024.);

                pos_min_cell_flt[gid].s2() = f64(pos_min_cell[gid].s2())*(1/1024.);
                pos_max_cell_flt[gid].s2() = f64(pos_max_cell[gid].s2())*(1/1024.);
            

                pos_min_cell_flt[gid] *= b_box_max - b_box_min;
                pos_min_cell_flt[gid] += b_box_min;
                
                pos_max_cell_flt[gid] *= b_box_max - b_box_min;
                pos_max_cell_flt[gid] += b_box_min;

            }
        );

    };

    queue.submit(ker_convert_cell_ranges);
}


template<>
void sycl_convert_cell_range<u64,f64_3>(sycl::queue & queue,

    u32 leaf_cnt , 
    u32 internal_cnt,
    f64_3 bounding_box_min,
    f64_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u32_3>> & buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<u32_3>> & buf_pos_max_cell,
    std::unique_ptr<sycl::buffer<f64_3>> & buf_pos_min_cell_flt,
    std::unique_ptr<sycl::buffer<f64_3>> & buf_pos_max_cell_flt ){

    using f3_xyzh = f64_3;

    sycl::range<1> range_cell{leaf_cnt + internal_cnt};

    auto ker_convert_cell_ranges = [&](sycl::handler &cgh) {
        f3_xyzh b_box_min = bounding_box_min;
        f3_xyzh b_box_max = bounding_box_max;

        auto pos_min_cell = buf_pos_min_cell->get_access<sycl::access::mode::read>(cgh);
        auto pos_max_cell = buf_pos_max_cell->get_access<sycl::access::mode::read>(cgh);

        // auto pos_min_cell_flt = buf_pos_min_cell_flt->get_access<sycl::access::mode::discard_write>(cgh);
        // auto pos_max_cell_flt = buf_pos_max_cell_flt->get_access<sycl::access::mode::discard_write>(cgh);

        auto pos_min_cell_flt = sycl::accessor { *buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};
        auto pos_max_cell_flt = sycl::accessor { *buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::property::no_init{}};

        cgh.parallel_for<class Convert_cell_range_u64_f64>(
            range_cell, [=](sycl::item<1> item) {

                u32 gid = (u32) item.get_id(0);

                
                pos_min_cell_flt[gid].s0() = f64(pos_min_cell[gid].s0())*(1/2097152.);
                pos_max_cell_flt[gid].s0() = f64(pos_max_cell[gid].s0())*(1/2097152.);

                pos_min_cell_flt[gid].s1() = f64(pos_min_cell[gid].s1())*(1/2097152.);
                pos_max_cell_flt[gid].s1() = f64(pos_max_cell[gid].s1())*(1/2097152.);

                pos_min_cell_flt[gid].s2() = f64(pos_min_cell[gid].s2())*(1/2097152.);
                pos_max_cell_flt[gid].s2() = f64(pos_max_cell[gid].s2())*(1/2097152.);
                

                pos_min_cell_flt[gid] *= b_box_max - b_box_min;
                pos_min_cell_flt[gid] += b_box_min;
                
                pos_max_cell_flt[gid] *= b_box_max - b_box_min;
                pos_max_cell_flt[gid] += b_box_min;

            }
        );

    };

    queue.submit(ker_convert_cell_ranges);
}
