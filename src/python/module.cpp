#include <eigenpy/eigenpy.hpp>

#include "common.h"

void exposeAltro();
void exposeAligatorToAltro();
void exposeIpoptSolver();

BOOST_PYTHON_MODULE(aligator_bench_pywrap) {
  exposeAltro();
  exposeAligatorToAltro();
  exposeIpoptSolver();
}
