enable_testing()

find_package(GTest REQUIRED)
include(GoogleTest)

# Create a test
function(create_test filename)
  get_filename_component(name ${filename} NAME_WE)
  add_executable(${name} ${filename})
  target_link_libraries(
    ${name}
    PRIVATE aligator_bench_robots GTest::gtest_main test_problems
  )
  gtest_discover_tests(${name})
endfunction()

create_test(test_loader.cpp)
create_test(test_aligator_to_altro.cpp)
create_test(test_ipopt_solver.cpp)
create_test(test_triang.cpp)
