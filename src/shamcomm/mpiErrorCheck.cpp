// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file worldInfo.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Use this header to include MPI properly
 *
 */

#include "mpiErrorCheck.hpp"
#include "mpi.h"
#include <cstdio>

void shamcomm::check_mpi_return(int ret, const char *log) {
    if (ret != MPI_SUCCESS) {
        fprintf(stderr,"error in MPI call : %s\n", log);
        MPI_Abort(MPI_COMM_WORLD, 10);
    }
}