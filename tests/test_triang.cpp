#include <gtest/gtest.h>
#include "types.hpp"
#include "triang_util.hpp"
#include <proxsuite-nlp/fmt-eigen.hpp>

using namespace aligator_bench;

class Triang : public testing::TestWithParam<int> {};

GTEST_TEST(Triang, Size) {
  EXPECT_EQ(lowTriangSize(4), 10);
  EXPECT_EQ(lowTriangSize(5), 15);
  EXPECT_EQ(lowTriangSize(10), 55);
}

TEST_P(Triang, setZero) {
  const int n = GetParam();
  VectorXs x;
  x.setOnes(lowTriangSize(n));
  lowTriangSetZero(n, x.data());
  // fmt::println("set to zero: {}", x.transpose());
  EXPECT_TRUE(x.isZero());
}

TEST_P(Triang, addEigen) {
  const int n = GetParam();
  VectorXs x;
  x.resize(lowTriangSize(n));
  lowTriangSetZero(n, x.data());
  MatrixXs a = MatrixXs::Ones(n, n);
  lowTriangAddFromEigen(x.data(), a, 2.0);
  lowTriangCoeff(n, x.data(), 2, 0) = -33;
  // fmt::println("x:  {}", x.transpose());

  MatrixXs b(n, n);
  lowTriangToEigen(n, x.data(), b);
  // fmt::println("b:\n{}", b);
}

TEST_P(Triang, setCoeff) {
  const int n = GetParam();
  VectorXs x;
  x.resize(lowTriangSize(n));
  lowTriangSetZero(n, x.data());
  lowTriangCoeff(n, x.data(), 0, 0) = 1;
  lowTriangCoeff(n, x.data(), 1, 0) = 2.2;
  lowTriangCoeff(n, x.data(), 2, 0) = -2.2;
  lowTriangCoeff(n, x.data(), 1, 1) = 42.;
  if (n > 3) {
    const int mid = n / 2;
    lowTriangCoeff(n, x.data(), 2, 2) = 13.;
    lowTriangCoeff(n, x.data(), mid, mid) = n;
    lowTriangCoeff(n, x.data(), n - 1, 0) = n;
    lowTriangCoeff(n, x.data(), n - 1, n - 1) = n;
  }
  MatrixXs b(n, n);
  lowTriangToEigen(n, x.data(), b);
  fmt::println("b:\n{}\n", b);
}

INSTANTIATE_TEST_SUITE_P(setZero, Triang, testing::Values(3, 5, 10, 20));
