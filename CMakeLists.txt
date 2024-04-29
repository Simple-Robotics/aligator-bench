cmake_minimum_required(VERSION 3.20)
set(PROJECT_URL "https://github.com/Simple-Robotics/aligator-bench.git")
set(PROJECT_NAME aligator-bench)
set(PROJECT_DESCRIPTION "")

set(CMAKE_DISABLE_WERROR True)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin/)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(${PROJECT_NAME})

include(cmake/base.cmake)
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})
set(CMAKE_VERBOSE_MAKEFILE False)

find_package(pinocchio REQUIRED)
find_package(aligator REQUIRED)
find_package(example-robot-data REQUIRED)
find_package(benchmark REQUIRED)

add_subdirectory(third-party/altro-cpp)

add_library(aligator-bench STATIC)
target_include_directories(aligator-bench PUBLIC src)
target_sources(aligator-bench PUBLIC src/util.cpp)

target_link_libraries(aligator-bench PUBLIC example-robot-data::example-robot-data)

