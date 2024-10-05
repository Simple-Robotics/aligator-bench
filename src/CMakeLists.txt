find_package(aligator REQUIRED)
find_package(Ipopt REQUIRED)

set(
  albench_SOURCES
  aligator-to-altro-types.cpp
  aligator-problem-to-altro.cpp
  ipopt-interface.cpp
)
add_library(aligator_bench SHARED ${albench_SOURCES})
target_include_directories(
  aligator_bench
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(
  aligator_bench
  PUBLIC aligator::aligator altro::altro Ipopt::Ipopt
)

add_subdirectory(robots)
add_subdirectory(python)
