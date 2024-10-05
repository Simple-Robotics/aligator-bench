#pragma once

#include <memory>
#include <aligator/context.hpp>

namespace alcontext = aligator::context;

namespace Ipopt {
class IpoptApplication;
}

namespace aligator_bench {
using alcontext::TrajOptProblem;

struct SolverIpopt {
  SolverIpopt();
  void setup(const TrajOptProblem &problem);
  std::unique_ptr<Ipopt::IpoptApplication> ipopt_app_;
};

} // namespace aligator_bench
