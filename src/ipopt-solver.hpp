#pragma once

#include <memory>
#include <IpReturnCodes.hpp>
#include <IpSmartPtr.hpp>
#include <IpTNLP.hpp>
#include <aligator/context.hpp>

namespace alcontext = aligator::context;

namespace Ipopt {
class IpoptApplication;
}

namespace aligator_bench {
using alcontext::TrajOptProblem;
class TrajOptIpoptNLP;

struct SolverIpopt {
  SolverIpopt(bool rethrow_nonipopt_exception = false);
  Ipopt::ApplicationReturnStatus setup(const TrajOptProblem &problem);

  void setOption(const std::string &name, std::string_view value);
  void setOption(const std::string &name, const std::string &value);
  void setOption(const std::string &name, int value);
  void setOption(const std::string &name, double value);

  Ipopt::ApplicationReturnStatus solve();

  std::unique_ptr<Ipopt::IpoptApplication> ipopt_app_;
  Ipopt::SmartPtr<Ipopt::TNLP> adapter_;
};

} // namespace aligator_bench
