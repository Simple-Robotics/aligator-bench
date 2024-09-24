find_package(example-robot-data REQUIRED)

set(ROBOT_TOML_DIR ${PROJECT_SOURCE_DIR}/assets)

file(GLOB robot_SOURCES CONFIGURE_DEPENDS *.cpp)

add_library(aligator_bench_robots SHARED ${robot_SOURCES})
target_link_libraries(aligator_bench_robots PUBLIC pinocchio::pinocchio_parsers)
target_link_libraries(aligator_bench_robots PRIVATE toml11)
target_link_libraries(aligator_bench_robots
                      PUBLIC example-robot-data::example-robot-data)
target_compile_definitions(aligator_bench_robots
                           PUBLIC ROBOT_TOML_DIR="${ROBOT_TOML_DIR}")
