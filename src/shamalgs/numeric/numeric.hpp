// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

namespace shamalgs::numeric {

    template<class T>
    sycl::buffer<T> exclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    template<class T>
    sycl::buffer<T> inclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    /**
     * @brief Stream compaction algorithm
     * 
     * @param q the queue to run on
     * @param buf_flags buffer of only 0 and ones
     * @param len the lenght of the buffer considered
     * @return std::tuple<sycl::buffer<u32>, u32> table of the index to extract and its size
     */
    std::tuple<sycl::buffer<u32>, u32> stream_compact(sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len);

} // namespace shamalgs::numeric
