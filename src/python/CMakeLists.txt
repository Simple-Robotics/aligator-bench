include(${PROJECT_SOURCE_DIR}/cmake/python.cmake)
find_package(eigenpy REQUIRED)

set(PYWRAP_SOURCES module.cpp expose-altro.cpp)

python3_add_library(aligator_bench_pywrap MODULE ${PYWRAP_SOURCES})
target_link_libraries(aligator_bench_pywrap PUBLIC aligator_bench)
target_link_libraries(aligator_bench_pywrap PRIVATE eigenpy::eigenpy)
