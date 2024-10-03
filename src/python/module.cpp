#include <eigenpy/eigenpy.hpp>

#include "common.h"

void exposeAltro();

BOOST_PYTHON_MODULE(aligator_bench_pywrap) { exposeAltro(); }
