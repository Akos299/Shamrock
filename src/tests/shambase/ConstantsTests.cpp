// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/constants.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambase/constants", checkconstantmatchsycl, 1) {

    using namespace shambase::constants;

    _AssertFloatEqual(pi<f32>, 4 * sycl::atan(unity<f32>), 1e-25);
    _AssertFloatEqual(pi<f64>, 4 * sycl::atan(unity<f64>), 1e-25);
}
