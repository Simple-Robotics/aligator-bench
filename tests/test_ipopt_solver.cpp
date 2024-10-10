#include <gtest/gtest.h>
#include "types.hpp"
#include "linear-problem.hpp"
#include "ipopt-interface.hpp"
#include "ipopt-solver.hpp"
#include <IpIpoptApplication.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <proxsuite-nlp/fmt-eigen.hpp>

class AdapterTest : public testing::Test {
public:
  size_t horizon = 10;
  int nx = 4;
  int nu = 2;
  TrajOptProblem problem;
  aligator_bench::TrajOptIpoptNLP adapter;

  AdapterTest()
      : problem{createLinearProblem(horizon, nx, nu)}, adapter{problem} {}

  void SetUp() override {
    get_nlp_info_flag = adapter.get_nlp_info(nvars, nconstraints, nnz_jac_g,
                                             nnz_h_lag, index_style);
  }
  bool get_nlp_info_flag;
  int nvars, nconstraints, nnz_jac_g, nnz_h_lag;
  Ipopt::TNLP::IndexStyleEnum index_style;
};

GTEST_TEST_F(AdapterTest, get_nlp_info) {
  EXPECT_TRUE(get_nlp_info_flag);
  EXPECT_EQ(nvars, ((int(horizon) + 1) * nx + int(horizon) * nu));
  fmt::println("nvars:\t\t{:d}\n"
               "nconstraints:\t{:d}\n"
               "nnz_jac_g:\t{:d}\n"
               "nnz_h_lag:\t{:d}",
               nvars, nconstraints, nnz_jac_g, nnz_h_lag);
}

GTEST_TEST_F(AdapterTest, get_bounds_info) {
  std::vector<double> traj_l, traj_u;
  traj_l.resize(size_t(nvars));
  traj_u = traj_l;
  std::vector<double> cstr_l, cstr_u;
  cstr_l.resize(size_t(nvars));
  cstr_u = cstr_l;
  bool bounds_info =
      adapter.get_bounds_info(nvars, traj_l.data(), traj_u.data(), nconstraints,
                              cstr_l.data(), cstr_u.data());
  EXPECT_TRUE(bounds_info);
}

GTEST_TEST_F(AdapterTest, get_starting_point) {
  using aligator_bench::ConstVecMap;
  double *traj_data = new double[size_t(nvars)];
  double *lambda_data = new double[size_t(nconstraints)];
  bool ret = adapter.get_starting_point(nvars, true, traj_data, false, 0, 0,
                                        nconstraints, true, lambda_data);
  delete[] traj_data;
  delete[] lambda_data;
  EXPECT_TRUE(ret);
}

struct SolverTest : public testing::Test {
  size_t horizon = 100;
  int nx = 50;
  int nu = 40;
  TrajOptProblem problem;
  aligator_bench::SolverIpopt solver;
  Ipopt::ApplicationReturnStatus init_status;

  SolverTest()
      : problem{createLinearProblem(horizon, nx, nu, false)}, solver{true} {}

  void SetUp() override { init_status = solver.setup(problem); }
};

GTEST_TEST_F(SolverTest, Initialize) {
  EXPECT_EQ(init_status, Ipopt::Solve_Succeeded);
}

GTEST_TEST_F(SolverTest, nlp_info) {
  int n, m, nnz_jac_g, nnz_h_lag;
  Ipopt::TNLP::IndexStyleEnum index_style;
  solver.adapter_->get_nlp_info(n, m, nnz_jac_g, nnz_h_lag, index_style);
  fmt::println("nvars:\t\t{:d}\n"
               "nconstraints:\t{:d}\n"
               "nnz_jac_g:\t{:d}\n"
               "nnz_h_lag:\t{:d}",
               n, m, nnz_jac_g, nnz_h_lag);
}

GTEST_TEST_F(SolverTest, setTol) {
  solver.setOption("tol", 3.14e-6);
  std::string outfile = "ipopt.out";
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
