#include <eigenpy/eigenpy.hpp>

#include "common.h"

void exposeAltro();
void exposeAligatorToAltro();

BOOST_PYTHON_MODULE(aligator_bench_pywrap) {
  exposeAltro();
  exposeAligatorToAltro();
}
