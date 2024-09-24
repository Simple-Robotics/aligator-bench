enable_testing()

find_package(GTest REQUIRED)
include(GoogleTest)

# Create a test
function(create_test filename)
  get_filename_component(name ${filename} NAME_WE)
  add_executable(${name} ${filename})
  target_link_libraries(${name} PRIVATE ${PROJECT_NAME})
  target_link_libraries(${name} PRIVATE aligator_bench_robots)
  target_link_libraries(${name} PRIVATE GTest::gtest_main)
  gtest_discover_tests(${name})
endfunction()

create_test(test_loader.cpp)
