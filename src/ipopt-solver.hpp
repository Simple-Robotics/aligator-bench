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
using alcontext::VectorOfVectors;
class TrajOptIpoptNLP;

struct SolverIpopt {
  SolverIpopt(bool rethrow_nonipopt_exception = false);
  Ipopt::ApplicationReturnStatus setup(const TrajOptProblem &problem,
                                       bool verbose = false);

  void setOption(const std::string &name, const std::string &value);
  void setOption(const std::string &name, int value);
  void setOption(const std::string &name, double value);
  void setMaxiters(int value) { setOption("max_iter", value); }

  const VectorOfVectors &xs() const;
  const VectorOfVectors &us() const;
  const VectorOfVectors &lams() const;
  const VectorOfVectors &vs() const;

  Ipopt::ApplicationReturnStatus solve();

  std::unique_ptr<Ipopt::IpoptApplication> ipopt_app_;
  Ipopt::SmartPtr<Ipopt::TNLP> adapter_;

  int num_iter_;
  double traj_cost_;
  double dual_infeas_;
  double cstr_violation_;
  double complementarity_;
};

} // namespace aligator_bench
