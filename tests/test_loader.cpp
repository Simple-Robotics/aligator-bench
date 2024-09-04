#include "robots/robot_load.hpp"
#include <gtest/gtest.h>

GTEST_TEST(LoadTest, UR5) {
  pinocchio::Model model;
  aligator_bench::loadModelFromToml("ur.toml", "ur5", model);
  EXPECT_EQ(model.njoints, 7);
  EXPECT_EQ(model.nq, 6);
  EXPECT_EQ(model.name, "ur5");
}

GTEST_TEST(LoadTest, DoublePendulum) {
  pinocchio::Model model;
  aligator_bench::loadModelFromToml("double_pendulum.toml", "double_pendulum", model);
  EXPECT_EQ(model.njoints, 3);
  EXPECT_EQ(model.nq, 2);
  EXPECT_EQ(model.nv, 2);
  EXPECT_EQ(model.name, "2dof_planar");
}


GTEST_TEST(LoadTest, DoublePendulumContinuous) {
  pinocchio::Model model;
  aligator_bench::loadModelFromToml("double_pendulum.toml", "double_pendulum_continuous", model);
  EXPECT_EQ(model.njoints, 3);
  EXPECT_EQ(model.nq, 4);
  EXPECT_EQ(model.nv, 2);
  EXPECT_EQ(model.name, "2dof_planar");
}
