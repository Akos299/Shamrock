// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//
#pragma once
#include "aliases.hpp"
#include "shamsys/legacy/sycl_handler.hpp"


//%Impl status : Clean

namespace syclalgs {

    namespace basic {

    } // namespace basic

    namespace reduction {

        template <class T> 
        bool equals(sycl::buffer<T> & buf1, sycl::buffer<T> & buf2, u32 cnt);
        
    } // namespace reduction

    namespace convert {
        template<class T> sycl::buffer<T> vector_to_buf(std::vector<T> && vec);

        template<class T> sycl::buffer<T> vector_to_buf(std::vector<T> & vec);

        //template<class T> std::unique_ptr<sycl::buffer<u32>> duplicate(std::unique_ptr<sycl::buffer<u32>> & vec);
    } // namespace convert

} // namespace syclalgs


