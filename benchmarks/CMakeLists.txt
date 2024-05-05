find_package(benchmark REQUIRED)
find_package(example-robot-data REQUIRED)

function(create_benchmark filename)
  get_filename_component(name ${filename} NAME_WE)
  add_executable(${name} ${filename})
  target_link_libraries(${name} PRIVATE ${PROJECT_NAME})
  target_link_libraries(${name} PRIVATE benchmark::benchmark)
endfunction()

create_benchmark(ur5.cpp)
