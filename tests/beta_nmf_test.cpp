// Author: Emilie Gillet (emilie.o.gillet@gmail.com)
//
// Unit-tests for beta-divergence non-negative matrix factorization.

#include <gtest/gtest.h>

#include "beta_nmf/beta_nmf.h"

namespace {
  
using namespace Eigen;

using beta_nmf::BetaNmf;
using beta_nmf::UPDATE_BOTH;
  
class NmfTest : public ::testing::Test {
 public:
  void SetUp() {
    w_.resize(6, 3);
    w_ <<
        0, 1, 0,
        0, 1, 0,
        1, 0, 1,
        0, 0, 0,
        1, 0, 0,
        0, 0, 1;
    h_.resize(3, 11);
    h_ <<
        1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1;
    noise_.resize(6, 11);
    for (int i = 0; i < noise_.rows(); ++i) {
      for (int j = 0; j < noise_.cols(); ++j) {
        noise_(i, j) = 0.05 * (1 + (i + j) % 2);
      }
    }
    w_est_.resize(6, 3);
    w_est_ <<
        2, 1, 0.1,
        0.1, 1, 2,
        1, 2, 1,
        0.1, 2, 0.1,
        1, 0.1, 2,
        2, 0.1, 1;
    h_est_.resize(3, 11);
    h_est_ <<
        1, 2, 1, 2, 1, 2, 0, 2, 0, 1, 0,
        2, 0, 1, 1, 1, 2, 2, 2, 1, 2, 0,
        0, 0, 2, 0, 0, 0, 1, 1, 2, 0, 1;
    v_.resize(6, 11);
    v_ = (w_.matrix() * h_.matrix()).array() + noise_;
  }

 protected:
  Array<double, Dynamic, Dynamic> v_;
  Array<double, Dynamic, Dynamic> w_;
  Array<double, Dynamic, Dynamic> h_;
  Array<double, Dynamic, Dynamic> noise_;
  Array<double, Dynamic, Dynamic> w_est_;
  Array<double, Dynamic, Dynamic> h_est_;
};

TEST_F(NmfTest, TestItakuraSaito) {
  Array<double, Dynamic, Dynamic> cost;
  BetaNmf(v_, 0.0, 50, UPDATE_BOTH, &w_est_, &h_est_, &cost);

  EXPECT_NEAR(113.57, cost(0), 0.01);
  EXPECT_NEAR( 46.50, cost(1), 0.01);
  EXPECT_NEAR( 30.50, cost(2), 0.01);
  EXPECT_NEAR(  1.46, cost(50), 0.01);

  Array<double, Dynamic, Dynamic> d = w_est_.colwise().sum();
  EXPECT_NEAR(1.0, d(0), 0.01);
  EXPECT_NEAR(1.0, d(1), 0.01);
  EXPECT_NEAR(1.0, d(2), 0.01);

  EXPECT_NEAR(0.46, w_est_(2, 0), 0.01);
  EXPECT_NEAR(0.44, w_est_(4, 0), 0.01);
  EXPECT_NEAR(0.47, w_est_(0, 1), 0.01);
  EXPECT_NEAR(0.46, w_est_(1, 1), 0.01);
  EXPECT_NEAR(0.43, w_est_(2, 2), 0.01);
  EXPECT_NEAR(0.47, w_est_(5, 2), 0.01);
}

TEST_F(NmfTest, TestKullbackLeibler) {
  Array<double, Dynamic, Dynamic> cost;
  BetaNmf(v_, 1.0, 50, UPDATE_BOTH, &w_est_, &h_est_, &cost);

  EXPECT_NEAR(139.41, cost(0), 0.01);
  EXPECT_NEAR( 10.03, cost(1), 0.01);
  EXPECT_NEAR(  7.39, cost(2), 0.01);
  EXPECT_NEAR(  0.14, cost(50), 0.01);

  Array<double, Dynamic, Dynamic> d = w_est_.colwise().sum();
  EXPECT_NEAR(1.0, d(0), 0.01);
  EXPECT_NEAR(1.0, d(1), 0.01);
  EXPECT_NEAR(1.0, d(2), 0.01);

  EXPECT_NEAR(0.45, w_est_(2, 0), 0.01);
  EXPECT_NEAR(0.44, w_est_(4, 0), 0.01);
  EXPECT_NEAR(0.46, w_est_(0, 1), 0.01);
  EXPECT_NEAR(0.47, w_est_(1, 1), 0.01);
  EXPECT_NEAR(0.43, w_est_(2, 2), 0.01);
  EXPECT_NEAR(0.45, w_est_(5, 2), 0.01);
}

TEST_F(NmfTest, TestEuclidian) {
  Array<double, Dynamic, Dynamic> cost;
  BetaNmf(v_, 2.0, 50, UPDATE_BOTH, &w_est_, &h_est_, &cost);

  EXPECT_NEAR(325.08, cost(0), 0.01);
  EXPECT_NEAR(  5.64, cost(1), 0.01);
  EXPECT_NEAR(  4.66, cost(2), 0.01);
  EXPECT_NEAR(  0.01, cost(50), 0.01);

  Array<double, Dynamic, Dynamic> d = w_est_.colwise().sum();
  EXPECT_NEAR(1.0, d(0), 0.01);
  EXPECT_NEAR(1.0, d(1), 0.01);
  EXPECT_NEAR(1.0, d(2), 0.01);

  EXPECT_NEAR(0.45, w_est_(2, 0), 0.01);
  EXPECT_NEAR(0.45, w_est_(4, 0), 0.01);
  EXPECT_NEAR(0.45, w_est_(0, 1), 0.01);
  EXPECT_NEAR(0.46, w_est_(1, 1), 0.01);
  EXPECT_NEAR(0.44, w_est_(2, 2), 0.01);
  EXPECT_NEAR(0.45, w_est_(5, 2), 0.01);
}

TEST_F(NmfTest, TestBetaDivergence05) {
  Array<double, Dynamic, Dynamic> cost;
  BetaNmf(v_, 0.5, 50, UPDATE_BOTH, &w_est_, &h_est_, &cost);

  EXPECT_NEAR(112.30, cost(0), 0.01);
  EXPECT_NEAR( 20.18, cost(1), 0.01);
  EXPECT_NEAR( 14.32, cost(2), 0.01);
  EXPECT_NEAR(  0.48, cost(50), 0.01);

  Array<double, Dynamic, Dynamic> d = w_est_.colwise().sum();
  EXPECT_NEAR(1.0, d(0), 0.01);
  EXPECT_NEAR(1.0, d(1), 0.01);
  EXPECT_NEAR(1.0, d(2), 0.01);

  EXPECT_NEAR(0.45, w_est_(2, 0), 0.01);
  EXPECT_NEAR(0.44, w_est_(4, 0), 0.01);
  EXPECT_NEAR(0.46, w_est_(0, 1), 0.01);
  EXPECT_NEAR(0.47, w_est_(1, 1), 0.01);
  EXPECT_NEAR(0.43, w_est_(2, 2), 0.01);
  EXPECT_NEAR(0.46, w_est_(5, 2), 0.01);
}

TEST_F(NmfTest, TestBetaDivergence15) {
  Array<double, Dynamic, Dynamic> cost;
  BetaNmf(v_, 1.5, 50, UPDATE_BOTH, &w_est_, &h_est_, &cost);

  EXPECT_NEAR(202.17, cost(0), 0.01);
  EXPECT_NEAR(  7.35, cost(1), 0.01);
  EXPECT_NEAR(  5.77, cost(2), 0.01);
  EXPECT_NEAR(  0.04, cost(50), 0.01);

  Array<double, Dynamic, Dynamic> d = w_est_.colwise().sum();
  EXPECT_NEAR(1.0, d(0), 0.01);
  EXPECT_NEAR(1.0, d(1), 0.01);
  EXPECT_NEAR(1.0, d(2), 0.01);

  EXPECT_NEAR(0.45, w_est_(2, 0), 0.01);
  EXPECT_NEAR(0.44, w_est_(4, 0), 0.01);
  EXPECT_NEAR(0.46, w_est_(0, 1), 0.01);
  EXPECT_NEAR(0.47, w_est_(1, 1), 0.01);
  EXPECT_NEAR(0.43, w_est_(2, 2), 0.01);
  EXPECT_NEAR(0.46, w_est_(5, 2), 0.01);
}

TEST_F(NmfTest, TestBetaDivergence22) {
  Array<double, Dynamic, Dynamic> cost;
  BetaNmf(v_, 2.2, 50, UPDATE_BOTH, &w_est_, &h_est_, &cost);

  EXPECT_NEAR(401.55, cost(0), 0.01);
  EXPECT_NEAR(  5.28, cost(1), 0.01);
  EXPECT_NEAR(  4.47, cost(2), 0.01);
  EXPECT_NEAR(  0.01, cost(50), 0.01);

  Array<double, Dynamic, Dynamic> d = w_est_.colwise().sum();
  EXPECT_NEAR(1.0, d(0), 0.01);
  EXPECT_NEAR(1.0, d(1), 0.01);
  EXPECT_NEAR(1.0, d(2), 0.01);

  EXPECT_NEAR(0.45, w_est_(2, 0), 0.01);
  EXPECT_NEAR(0.45, w_est_(4, 0), 0.01);
  EXPECT_NEAR(0.44, w_est_(0, 1), 0.01);
  EXPECT_NEAR(0.45, w_est_(1, 1), 0.01);
  EXPECT_NEAR(0.44, w_est_(2, 2), 0.01);
  EXPECT_NEAR(0.46, w_est_(5, 2), 0.01);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
