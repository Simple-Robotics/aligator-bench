find_package(pinocchio REQUIRED)
find_package(aligator REQUIRED)

add_library(aligator-bench STATIC)
file(GLOB_RECURSE albench_SOURCES CONFIGURE_DEPENDS *.cpp)
target_sources(aligator-bench PUBLIC ${albench_SOURCES})
target_include_directories(aligator-bench INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(aligator-bench PUBLIC aligator::aligator altro::altro)