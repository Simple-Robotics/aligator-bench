find_package(pinocchio REQUIRED)
find_package(aligator REQUIRED)

add_library(aligator_bench SHARED)
file(GLOB albench_SOURCES CONFIGURE_DEPENDS *.cpp)
target_sources(aligator_bench PRIVATE ${albench_SOURCES})
target_include_directories(
  aligator_bench INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(aligator_bench PUBLIC aligator::aligator altro::altro)

add_subdirectory(robots)
add_subdirectory(python)
