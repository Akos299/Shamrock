#pragma once

#include "aliases.hpp"
#include <memory>
#include <stdexcept>

namespace syclalg {

    //TODO to optimize
    template<class T>
    inline T get_max(sycl::queue & queue, std::unique_ptr<sycl::buffer<T>> & buf){

        T accum;

        if(buf){
            accum = buf->get_host_access()[0];

            {
                auto acc = buf->template get_access<sycl::access::mode::read>();

                // queue.submit([&](sycl::handler &cgh) {
                //     auto acc = buf->get_access<sycl::access::mode::read>(cgh);

                //     cgh.parallel_for(sycl::range(buf->size()), [=](sycl::item<1> item) {
                //         u32 i = (u32)item.get_id(0);

                //     });
                // });

                
                for (u32 i = 0; i < buf->size(); i++) {
                    accum = sycl::max(accum,acc[i]);
                }
            }
        }else{
            throw shamrock_exc("syclalg::get_max : input buffer not allocated");
        } 

        return accum;

    }


    //TODO to optimize
    template<class T>
    inline T get_min(sycl::queue & queue, std::unique_ptr<sycl::buffer<T>> & buf){

        T accum;

        if(buf){

            accum = buf->get_host_access()[0];

            {
                auto acc = buf->template get_access<sycl::access::mode::read>();

                // queue.submit([&](sycl::handler &cgh) {
                //     auto acc = buf->get_access<sycl::access::mode::read>(cgh);

                //     cgh.parallel_for(sycl::range(buf->size()), [=](sycl::item<1> item) {
                //         u32 i = (u32)item.get_id(0);

                //     });
                // });

                
                for (u32 i = 0; i < buf->size(); i++) {
                    accum = sycl::min(accum,acc[i]);
                }
            }

        }else {
            throw shamrock_exc("syclalg::get_min : input buffer not allocated");
        } 

        

        return accum;

    }


    template<class T> 
    inline std::tuple<T,T> get_min_max(sycl::queue & queue, std::unique_ptr<sycl::buffer<T>> & buf){
        
        T accum_min, accum_max;

        if(buf){

            accum_min = buf->get_host_access()[0];
            accum_max = buf->get_host_access()[0];

            {
                auto acc = buf->template get_access<sycl::access::mode::read>();

                // queue.submit([&](sycl::handler &cgh) {
                //     auto acc = buf->get_access<sycl::access::mode::read>(cgh);

                //     cgh.parallel_for(sycl::range(buf->size()), [=](sycl::item<1> item) {
                //         u32 i = (u32)item.get_id(0);

                //     });
                // });

                for (u32 i = 0; i < buf->size(); i++) {
                    accum_min = sycl::min(accum_min,acc[i]);
                    accum_max = sycl::max(accum_max,acc[i]);
                }
            }


        } else {
            throw shamrock_exc("syclalg::get_max : input buffer not allocated");
        } 

        return {accum_min,accum_max};

    }

}

