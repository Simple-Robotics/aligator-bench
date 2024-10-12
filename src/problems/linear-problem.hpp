#include <aligator/context.hpp>

namespace alcontext = aligator::context;

alcontext::TrajOptProblem createLinearProblem(const size_t horizon,
                                              const int nx, const int nu,
                                              bool terminal = true);
