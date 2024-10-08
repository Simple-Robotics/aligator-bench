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
  aligator_bench::SolverIpopt solver;

  SolverTest() : problem{createLinearProblem(horizon, nx, nu)}, solver{true} {}

  void SetUp() override { solver.setup(problem); }
};

GTEST_TEST_F(SolverTest, Initialize) {
  auto status = solver.setup(problem);
  EXPECT_EQ(status, Ipopt::Solve_Succeeded);
}

GTEST_TEST_F(SolverTest, setTol) {
  solver.setOption("tol", 3.14e-6);
  std::string_view outfile = "ipopt.out";
  solver.setOption("output_file", outfile);
  auto opts = solver.ipopt_app_->Options();
  std::string optlist;
  opts->PrintUserOptions(optlist);
  fmt::println("{}", optlist);
}

GTEST_TEST_F(SolverTest, solve) {
  auto status = solver.solve();
  EXPECT_EQ(status, Ipopt::Solve_Succeeded);
}
