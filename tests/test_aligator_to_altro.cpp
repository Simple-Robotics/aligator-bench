#include "aligator-to-altro-types.hpp"
#include <gtest/gtest.h>

#include <proxsuite-nlp/modelling/constraints.hpp>
#include <altro/altro.hpp>

GTEST_TEST(aligatorToAltro, EqualityConstraint) {
  using ZeroSet = proxsuite::nlp::EqualityConstraintTpl<double>;
  auto ct = altro::ConstraintType::EQUALITY;
  auto ct2 = aligatorConstraintAltroType(ZeroSet{});
  EXPECT_EQ(ct, ct2);
}

GTEST_TEST(aligatorToAltro, NegativeOrthant) {
  using NO = proxsuite::nlp::NegativeOrthantTpl<double>;
  auto ct = altro::ConstraintType::INEQUALITY;
  auto ct2 = aligatorConstraintAltroType(NO{});
  EXPECT_EQ(ct, ct2);
}
