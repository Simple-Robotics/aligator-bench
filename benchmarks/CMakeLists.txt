find_package(benchmark REQUIRED)

# Add a benchmark.
function(create_benchmark filename)
  get_filename_component(name ${filename} NAME_WE)
  add_executable(${name} ${filename})
  target_link_libraries(${name} PRIVATE ${PROJECT_NAME})
  target_link_libraries(${name} PRIVATE benchmark::benchmark)
  target_link_libraries(${name} PRIVATE aligator_bench_robots)
  target_link_libraries(${name} PRIVATE Matplot++::matplot)
endfunction()

create_benchmark(ur5.cpp)
create_benchmark(pendulum.cpp)
