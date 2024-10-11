#pragma once

#include "types.hpp"

namespace aligator_bench {

inline long lowTriangSize(long n) { return n * (n + 1) / 2; }

inline double &lowTriangCoeff(long n, double *a, long i, long j) {
  assert(j >= 0);
  assert(i >= j);
  long ix = i + j * n - j * (j + 1) / 2;
  assert(ix >= 0);
  assert(ix < lowTriangSize(n));
  return a[ix];
}

inline void lowTriangSetZero(long n, double *dst) {
  std::fill_n(dst, lowTriangSize(n), 0.);
}

inline void lowTriangAddFromEigen(double *dst, Eigen::Ref<const MatrixXs> src,
                                  double c) {
  Eigen::Index n = src.rows();
  assert(n == src.cols());
  for (long i = 0; i < n; i++)
    for (long j = 0; j <= i; j++)
      lowTriangCoeff(n, dst, i, j) += c * src(i, j);
}

inline void lowTriangToEigen(long n, double *src, Eigen::Ref<MatrixXs> dst) {
  for (long i = 0; i < n; i++)
    for (long j = 0; j <= i; j++)
      dst(i, j) = lowTriangCoeff(n, src, i, j);
}

} // namespace aligator_bench
