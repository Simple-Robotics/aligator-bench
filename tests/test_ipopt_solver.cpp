#include <gtest/gtest.h>
#include "linear-problem.hpp"
#include "ipopt-interface.hpp"
#include "ipopt-solver.hpp"
#include <IpIpoptApplication.hpp>
#include <aligator/core/traj-opt-problem.hpp>

GTEST_TEST(ipopt_interface, Adapter) {
  size_t horizon = 100;
  int nx = 50;
  int nu = 40;
  TrajOptProblem problem = createLinearProblem(horizon, nx, nu);
  aligator_bench::TrajOptIpoptNLP adapter{problem};

  int n, m, nnz_jac_g, nnz_h_lag;
  Ipopt::TNLP::IndexStyleEnum index_style;
  adapter.get_nlp_info(n, m, nnz_jac_g, nnz_h_lag, index_style);

  EXPECT_EQ(n, ((int(horizon) + 1) * nx + int(horizon) * nu));
  fmt::println("nnz_jac_g: {:d}", nnz_jac_g);
  fmt::println("nnz_h_lag: {:d}", nnz_h_lag);
}

struct SolverTest : public testing::Test {
  size_t horizon = 100;
  int nx = 50;
  int nu = 40;
  TrajOptProblem problem;

  SolverTest() : problem{createLinearProblem(horizon, nx, nu)} {}
};

GTEST_TEST_F(SolverTest, Initialize) {
  aligator_bench::SolverIpopt solver;
  auto status = solver.setup(problem);

  EXPECT_EQ(status, Ipopt::Solve_Succeeded);
}
