cmake_minimum_required(VERSION 3.20)
set(PROJECT_URL "https://github.com/Simple-Robotics/aligator-bench.git")
set(PROJECT_NAME aligator_bench)
set(PROJECT_DESCRIPTION "Very simple bench for aligator.")

set(CMAKE_DISABLE_WERROR True)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(BUILD_SHARED_LIBS ON)

include(cmake/base.cmake)
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})
set(CMAKE_VERBOSE_MAKEFILE False)

add_subdirectory(third-party/altro EXCLUDE_FROM_ALL)
add_subdirectory(third-party/toml11)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(src)
add_subdirectory(benchmarks)
add_subdirectory(tests)
