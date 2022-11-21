#include "unittests/shamrocktest.hpp"

#include <random>
#include "core/tree/kernels/key_morton_sort.hpp"

template<class u_morton, SortImplType impl> void unit_test_key_pair(){
    std::vector<u_morton> morton_list;

    constexpr u32 size_test = 16;

    for(u32 i = 0; i < size_test; i++){
        morton_list.push_back(i);
    }

    std::mt19937 eng(0x1111);

    shuffle (morton_list.begin(), morton_list.end(), eng);

    std::vector<u_morton> unsorted(morton_list.size());

    std::copy(morton_list.begin(), morton_list.end(),unsorted.begin());

    {
        std::unique_ptr<sycl::buffer<u_morton>> buf_morton = std::make_unique<sycl::buffer<u_morton>>(morton_list.data(),morton_list.size());
        std::unique_ptr<sycl::buffer<u32>> buf_index = std::make_unique<sycl::buffer<u32>>(morton_list.size());

        sycl_sort_morton_key_pair<u_morton,impl>(
            sycl_handler::get_compute_queue(),
            size_test,
            buf_index,
            buf_morton
            );

    }

    std::sort(unsorted.begin(), unsorted.end());


    for(u32 i = 0; i < size_test; i++){
        shamrock::test::asserts().assert_add("index [" +format("%d",i)+ "]",  unsorted[i]  == morton_list[i]);
    }
}

template<class u_morton, SortImplType impl> f64 benchmark_key_pair_sort(const u32 & nobj){
    std::vector<u_morton> morton_list;

    for(u32 i = 0; i < nobj; i++){
        morton_list.push_back(i);
    }

    std::mt19937 eng(0x1111);

    shuffle (morton_list.begin(), morton_list.end(), eng);

    Timer t;

    {
        std::unique_ptr<sycl::buffer<u_morton>> buf_morton = std::make_unique<sycl::buffer<u_morton>>(morton_list.data(),morton_list.size());
        std::unique_ptr<sycl::buffer<u32>> buf_index = std::make_unique<sycl::buffer<u32>>(morton_list.size());

        sycl_handler::get_compute_queue().wait();

        t.start();

        sycl_sort_morton_key_pair<u_morton,impl>(
            sycl_handler::get_compute_queue(),
            nobj,
            buf_index,
            buf_morton
            );

        sycl_handler::get_compute_queue().wait();

        t.end();

    }

    return t.nanosec;

}







TestStart(Unittest, "core/tree/kernels/key_pair_sort", key_pair_sort_test, 1){
    unit_test_key_pair<u32,MultiKernel>();
    unit_test_key_pair<u64,MultiKernel>();
}





constexpr u32 lim_bench = 1e9;

template<class u_morton, SortImplType impl> void wrapper_bench_key_sort(std::string name){
    std::vector<f64> test_sz;
    for(f64 i = 16; i < lim_bench; i*=2){
        test_sz.push_back(i);
    }

    auto & res = shamrock::test::test_data().new_dataset(name);

    std::vector<f64> results;

    for(const f64 & sz : test_sz){
        results.push_back(benchmark_key_pair_sort<u_morton,impl>(sz));
    }

    res.add_data("Nobj", test_sz);
    res.add_data("t_sort", results);


}

TestStart(Benchmark, "core/tree/kernels/key_pair_sort (benchmark)", key_pair_sort_bench, 1){
    
    wrapper_bench_key_sort<u32,MultiKernel>("bitonic u32 multi kernel");
    wrapper_bench_key_sort<u32,MultiKernel>("bitonic u64 multi kernel");
    
}