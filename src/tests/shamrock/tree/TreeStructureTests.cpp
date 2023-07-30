// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/random.hpp"
#include "shammath/CoordRange.hpp"
#include "shamrock/tree/TreeMortonCodes.hpp"
#include "shamrock/tree/TreeReducedMortonCodes.hpp"
#include "shamrock/tree/TreeStructure.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamrock/tree/TreeStructure::serialize", teststructureserialize, 1) {

    u32 cnt = 1000;
    shammath::CoordRange<f32_3> range_coord{f32_3{0, 0, 0}, f32_3{1, 1, 1}};

    sycl::queue &q = shamsys::instance::get_compute_queue();

    auto buf = shamalgs::random::mock_buffer(0x1111, cnt, range_coord.lower, range_coord.upper);

    using u_morton = u32;

    using TreeMorton    = shamrock::tree::TreeMortonCodes<u_morton>;
    using TreeRedMorton = shamrock::tree::TreeReducedMortonCodes<u_morton>;
    using TreeStruct    = shamrock::tree::TreeStructure<u_morton>;

    TreeMorton codes;
    codes.build(q, range_coord, cnt, buf);

    bool one_cell_mode;
    TreeRedMorton redcodes;
    redcodes.build(q, codes.obj_cnt, 2, codes, one_cell_mode);

    TreeStruct strc;
    if (!one_cell_mode) {
        strc.build(q, redcodes.tree_leaf_count - 1, *redcodes.buf_tree_morton);
    } else {
        strc.build_one_cell_mode();
    }

    shamalgs::SerializeHelper ser;
    ser.allocate(strc.serialize_byte_size());
    strc.serialize(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(std::move(recov));

        TreeStruct outser = TreeStruct::deserialize(ser2);

        shamtest::asserts().assert_bool("input match out", outser == strc);
    }
}