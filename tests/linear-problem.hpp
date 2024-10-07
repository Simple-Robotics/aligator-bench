#include <aligator/context.hpp>

using namespace aligator::context;

auto createLinearProblem(const size_t horizon, const int nx, const int nu)
    -> TrajOptProblem;
