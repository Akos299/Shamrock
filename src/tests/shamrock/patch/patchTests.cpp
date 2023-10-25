// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/patch/Patch.hpp"

#include "shamrock/patch/Patch.hpp"

#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamrock/patch/Patch.cpp:MpiType", patch_mpi_type, 2){

    using namespace shamrock::patch;

    Patch check_patch{};
    check_patch.id_patch = 156;
    check_patch.pack_node_index = 48414;
    check_patch.load_value = 4951956;
    check_patch.coord_min[0] = 0;
    check_patch.coord_min[1] = 1;
    check_patch.coord_min[2] = 2;
    check_patch.coord_max[0] = 3;
    check_patch.coord_max[1] = 8;
    check_patch.coord_max[2] = 6;
    check_patch.data_count = 7444444;
    check_patch.node_owner_id = 44444;



    if(shammpi::world_rank() == 0){
        mpi::send(&check_patch, 1, get_patch_mpi_type<3>(), 1, 0, MPI_COMM_WORLD);
    }

    if(shammpi::world_rank() == 1){
        Patch rpatch{};

        MPI_Status st;
        mpi::recv(&rpatch, 1, get_patch_mpi_type<3>(), 0, 0, MPI_COMM_WORLD, &st);

        shamtest::asserts().assert_bool("patch are equal", rpatch == check_patch);
    }
}

TestStart(Unittest, "shamrock/patch/Patch.cpp:SplitMerge", splitmergepatch, 1){

    using namespace shamrock::patch;

    Patch check_patch{};
    check_patch.id_patch = 0;
    check_patch.pack_node_index = u64_max;
    check_patch.load_value = 8;
    check_patch.coord_min[0] = 0;
    check_patch.coord_min[1] = 0;
    check_patch.coord_min[2] = 0;
    check_patch.coord_max[0] = 256;
    check_patch.coord_max[1] = 128;
    check_patch.coord_max[2] = 1024;
    check_patch.data_count = 8;
    check_patch.node_owner_id = 0;


    std::array<Patch, 8> splts = check_patch.get_split();


    shamtest::asserts().assert_bool("", splts[0].load_value == 1);
    shamtest::asserts().assert_bool("", splts[1].load_value == 1);
    shamtest::asserts().assert_bool("", splts[2].load_value == 1);
    shamtest::asserts().assert_bool("", splts[3].load_value == 1);
    shamtest::asserts().assert_bool("", splts[4].load_value == 1);
    shamtest::asserts().assert_bool("", splts[5].load_value == 1);
    shamtest::asserts().assert_bool("", splts[6].load_value == 1);
    shamtest::asserts().assert_bool("", splts[7].load_value == 1);

    shamtest::asserts().assert_bool("", splts[0].data_count == 1);
    shamtest::asserts().assert_bool("", splts[1].data_count == 1);
    shamtest::asserts().assert_bool("", splts[2].data_count == 1);
    shamtest::asserts().assert_bool("", splts[3].data_count == 1);
    shamtest::asserts().assert_bool("", splts[4].data_count == 1);
    shamtest::asserts().assert_bool("", splts[5].data_count == 1);
    shamtest::asserts().assert_bool("", splts[6].data_count == 1);
    shamtest::asserts().assert_bool("", splts[7].data_count == 1);

    Patch p = Patch::merge_patch(splts);

    shamtest::asserts().assert_bool("patch are equal", p == check_patch);
}


TestStart(Unittest, "shamrock/patch/Patch.cpp:SplitCoord", splitcoord, 1){

    using namespace shamrock::patch;

    Patch p0{};
    p0.id_patch = 0;
    p0.pack_node_index = u64_max;
    p0.load_value = 8;
    p0.coord_min[0] = 0;
    p0.coord_min[1] = 0;
    p0.coord_min[2] = 0;
    p0.coord_max[0] = 256;
    p0.coord_max[1] = 128;
    p0.coord_max[2] = 1024;
    p0.data_count = 8;
    p0.node_owner_id = 0;

    u64 min_x = p0.coord_min[0];
    u64 min_y = p0.coord_min[1];
    u64 min_z = p0.coord_min[2];

    u64 split_x = (((p0.coord_max[0] - p0.coord_min[0]) + 1)/2) - 1 + min_x;
    u64 split_y = (((p0.coord_max[1] - p0.coord_min[1]) + 1)/2) - 1 + min_y;
    u64 split_z = (((p0.coord_max[2] - p0.coord_min[2]) + 1)/2) - 1 + min_z;

    u64 max_x = p0.coord_max[0];
    u64 max_y = p0.coord_max[1];
    u64 max_z = p0.coord_max[2];

    std::array<u64, 3> split_out = p0.get_split_coord();

    shamtest::asserts().assert_bool("split 0", split_x == split_out[0]);
    shamtest::asserts().assert_bool("split 1", split_y == split_out[1]);
    shamtest::asserts().assert_bool("split 2", split_z == split_out[2]);

}