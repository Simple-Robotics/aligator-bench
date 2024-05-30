find_package(example-robot-data REQUIRED)

add_library(aligator_bench_robots SHARED robot_spec.hpp robot_spec.cpp)
target_link_libraries(aligator_bench_robots PUBLIC pinocchio::pinocchio)
target_link_libraries(aligator_bench_robots PRIVATE toml11)
target_link_libraries(aligator_bench_robots
                      PUBLIC example-robot-data::example-robot-data)
