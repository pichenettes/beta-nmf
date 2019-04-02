// Author: Emilie Gillet (emilie.o.gillet@gmail.com)
//
// Unit-tests for beta-divergence computation.

#include <gtest/gtest.h>

#include "beta_nmf/divergence.h"

namespace {
  
using beta_nmf::BetaDivergence;
  
class DivergenceTest : public ::testing::Test {
 public:
  void SetUp() {
    x_ << 0.5, 1, 2, 1.5, 0.5, 1;
    y_ << 1, 2, 3, 4, 5, 6;
  }

 protected:
  Eigen::Array<double, 2, 3> x_;
  Eigen::Array<double, 2, 3> y_;
};

TEST_F(DivergenceTest, TestItakuraSaito) {
  EXPECT_NEAR(3.1753, BetaDivergence(x_, y_, 0.0), 0.01);
}

TEST_F(DivergenceTest, TestKullbackLeibler) {
  EXPECT_NEAR(8.2351, BetaDivergence(x_, y_, 1.0), 0.01);
}

TEST_F(DivergenceTest, TestEuclidian) {
  EXPECT_NEAR(26.875, BetaDivergence(x_, y_, 2.0), 0.01);
}

TEST_F(DivergenceTest, TestBetaDivergence) {
  EXPECT_NEAR(4.9383, BetaDivergence(x_, y_, 0.5), 0.01);
  EXPECT_NEAR(14.540, BetaDivergence(x_, y_, 1.5), 0.01);
  EXPECT_NEAR(51.531, BetaDivergence(x_, y_, 2.5), 0.01);
}


}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
