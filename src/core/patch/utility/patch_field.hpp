// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file patch_field.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once

#include <vector>

#include "core/sys/mpi_handler.hpp"

template<class type>
class BufferedPField{public:
    sycl::buffer<type> buf_local;
    sycl::buffer<type> buf_global;
};

/**
 * @brief Define a field attached to a patch (exemple: FMM multipoles, hmax in SPH)
 * 
 * @tparam type type of object to store
 */
template<class type>
class PatchField{public:

    using T = type;

    std::vector<type> local_nodes_value;

    std::vector<type> global_values;

    inline void build_global(MPI_Datatype & dtype){
        mpi_handler::vector_allgatherv(local_nodes_value, dtype, global_values, dtype, MPI_COMM_WORLD);
    }


    

    inline BufferedPField<type> get_buffers(){
        return BufferedPField<type>{
            sycl::buffer<type>(local_nodes_value.data(),local_nodes_value.size()),
            sycl::buffer<type>(global_values.data(),global_values.size()),
        };
    }

};