#include <gtest/gtest.h>
#include "linear-problem.hpp"
#include "ipopt-interface.hpp"
#include "ipopt-solver.hpp"
#include <IpIpoptApplication.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <proxsuite-nlp/fmt-eigen.hpp>

using namespace aligator_bench;

class AdapterTest : public testing::Test {
public:
  size_t horizon = 10;
  int nx = 4;
  int nu = 2;
  TrajOptProblem problem;
  TrajOptIpoptNLP adapter;

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
  double *traj_data = new double[size_t(nvars)];
  double *lambda_data = new double[size_t(nconstraints)];
  bool ret = adapter.get_starting_point(nvars, true, traj_data, false, 0, 0,
                                        nconstraints, true, lambda_data);
  delete[] traj_data;
  delete[] lambda_data;
  EXPECT_TRUE(ret);
}

GTEST_TEST_F(AdapterTest, eval_f_and_grad) {
  double *traj_data = new double[size_t(nvars)];
  double *lambda_data = new double[size_t(nconstraints)];
  adapter.get_starting_point(nvars, true, traj_data, false, 0, 0, nconstraints,
                             true, lambda_data);

  double obj_value;
  EXPECT_TRUE(adapter.eval_f(nvars, traj_data, true, obj_value));

  double *grad_data = new double[size_t(nvars)];
  EXPECT_TRUE(adapter.eval_g(nvars, traj_data, true, nconstraints, grad_data));
  delete[] traj_data;
  delete[] lambda_data;
  delete[] grad_data;
}

GTEST_TEST_F(AdapterTest, eval_jac_g_sparsity) {
  double *traj_data = new double[size_t(nvars)];
  int *iRow = new int[size_t(nnz_jac_g)];
  int *jCol = new int[size_t(nnz_jac_g)];

  bool ret = adapter.eval_jac_g(nvars, traj_data, true, nconstraints, nnz_jac_g,
                                iRow, jCol, NULL);
  EXPECT_TRUE(ret);

  delete[] traj_data;
  delete[] iRow;
  delete[] jCol;
}

GTEST_TEST_F(AdapterTest, eval_jac_g) {
  double *traj_data = new double[size_t(nvars)];
  double *jac_data = new double[size_t(nnz_jac_g)];

  bool ret = adapter.eval_jac_g(nvars, traj_data, true, nconstraints, nnz_jac_g,
                                NULL, NULL, jac_data);
  EXPECT_TRUE(ret);

  delete[] traj_data;
  delete[] jac_data;
}

GTEST_TEST_F(AdapterTest, eval_h_sparsity) {
  double *traj_data = new double[size_t(nvars)];
  double *lambda_data = new double[size_t(nconstraints)];
  int *iRow = new int[size_t(nnz_h_lag)];
  int *jCol = new int[size_t(nnz_h_lag)];
  // query Hessian sparsity
  adapter.eval_h(nvars, traj_data, true, 1.0, nconstraints, lambda_data, true,
                 nnz_h_lag, iRow, jCol, NULL);
  delete[] traj_data;
  delete[] lambda_data;
  delete[] iRow;
  delete[] jCol;
}

GTEST_TEST_F(AdapterTest, eval_h) {
  double *traj_data = new double[size_t(nvars)];
  double *lambda_data = new double[size_t(nconstraints)];
  // query Hessian values
  double *hess_data = new double[size_t(nnz_h_lag)];
  adapter.eval_h(nvars, traj_data, true, 0.1, nconstraints, lambda_data, true,
                 nnz_h_lag, NULL, NULL, hess_data);
  delete[] traj_data;
  delete[] lambda_data;
  delete[] hess_data;
}

struct SolverTestLinearProb : public testing::TestWithParam<size_t> {
  size_t horizon;
  int nx = 20;
  int nu = 12;
  TrajOptProblem problem;
  SolverIpopt solver;
  Ipopt::ApplicationReturnStatus init_status;

  SolverTestLinearProb()
      : horizon(GetParam()),
        problem{createLinearProblem(horizon, nx, nu, true)}, solver{true} {}

  void SetUp() override { init_status = solver.setup(problem); }
};

TEST_P(SolverTestLinearProb, Initialize) {
  EXPECT_EQ(init_status, Ipopt::Solve_Succeeded);
}

TEST_P(SolverTestLinearProb, nlp_info) {
  int n, m, nnz_jac_g, nnz_h_lag;
  Ipopt::TNLP::IndexStyleEnum index_style;
  bool ret =
      solver.adapter_->get_nlp_info(n, m, nnz_jac_g, nnz_h_lag, index_style);
  EXPECT_TRUE(ret);
  fmt::println("nvars:\t\t{:d}\n"
               "nconstraints:\t{:d}\n"
               "nnz_jac_g:\t{:d}\n"
               "nnz_h_lag:\t{:d}",
               n, m, nnz_jac_g, nnz_h_lag);
}

TEST_P(SolverTestLinearProb, setTol) {
  solver.setOption("tol", 3.14e-6);
  std::string outfile = "ipopt.out";
  solver.setOption("output_file", outfile);
  auto opts = solver.ipopt_app_->Options();
  std::string optlist;
  opts->PrintUserOptions(optlist);
  fmt::println("{}", optlist);
}

TEST_P(SolverTestLinearProb, solve) {
  auto status = solver.solve();
  EXPECT_EQ(status, Ipopt::Solve_Succeeded);
}

INSTANTIATE_TEST_SUITE_P(Horionz, SolverTestLinearProb,
                         testing::Values(1, 10, 50, 100));
