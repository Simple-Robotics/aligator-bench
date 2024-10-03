#include "common.h"
#include <aligator/core/traj-opt-problem.hpp>
#include "aligator-problem-to-altro.hpp"

using namespace aligator_bench;

void exposeAligatorToAltro() {
  bp::import("aligator");
  bp::def("init_altro_from_aligator_problem", init_altro_from_aligator_problem,
          ("aligator_problem"_a));
}
