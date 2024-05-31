find_package(example-robot-data REQUIRED)

file(GLOB robot_SOURCES CONFIGURE_DEPENDS *.cpp)

add_library(aligator_bench_robots SHARED ${robot_SOURCES})
target_link_libraries(aligator_bench_robots PUBLIC pinocchio::pinocchio)
target_link_libraries(aligator_bench_robots PRIVATE toml11)
target_link_libraries(aligator_bench_robots
                      PUBLIC example-robot-data::example-robot-data)
