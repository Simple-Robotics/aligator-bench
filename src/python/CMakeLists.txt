include(${PROJECT_SOURCE_DIR}/cmake/python.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/stubs.cmake)
find_package(eigenpy REQUIRED)

set(
  PYWRAP_SOURCES
  module.cpp
  expose-altro.cpp
  expose-aligator-to-altro.cpp
  expose-ipopt-solver.cpp
  expose-collision-distance.cpp
)

set(PYLIB_NAME aligator_bench_pywrap)
python3_add_library(${PYLIB_NAME} MODULE WITH_SOABI ${PYWRAP_SOURCES})
target_link_libraries(
  ${PYLIB_NAME}
  PUBLIC aligator_bench
  PRIVATE eigenpy::eigenpy
)

if(IS_ABSOLUTE ${PYTHON_SITELIB})
  set(ABSOLUTE_PYTHON_SITELIB ${PYTHON_SITELIB})
else()
  set(ABSOLUTE_PYTHON_SITELIB ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITELIB})
endif()
load_stubgen()
generate_stubs(${CMAKE_LIBRARY_OUTPUT_DIRECTORY} ${PYLIB_NAME}
               ${ABSOLUTE_PYTHON_SITELIB} ${PYLIB_NAME}
)
