// Author: Emilie Gillet (emilie.o.gillet@gmail.com)
//
// NMF with beta-divergence.

#ifndef BETA_NMF_BETA_NMF_H_
#define BETA_NMF_BETA_NMF_H_

#include <third_party/eigen3/Eigen/Eigen>

#include "beta_nmf/divergence.h"

namespace beta_nmf {

enum UpdateMode {
  UPDATE_NONE = 0,
  UPDATE_W = 1,
  UPDATE_H = 2,
  UPDATE_BOTH = 3
};

template<typename T, typename U, typename V, typename W, typename X>
void ComputeUpdate(
    const T& v,
    const U& v_est,
    const V& w,
    const W& h_T,
    typename T::Scalar beta,
    X* update) {
  typedef typename T::Scalar Scalar;
  if (beta == Scalar(1)) {
    *update = 
        ((v / v_est).matrix() * h_T.matrix()).array() /
        (T::Ones(v_est.rows(), v_est.cols()).matrix() * h_T.matrix()).array();
  } else if (beta == Scalar(2)) {
    *update = 
      (v.matrix() * h_T.matrix()).array() /
      (w.matrix() * (h_T.transpose().matrix() * h_T.matrix())).array();
  } else {
    Scalar gamma(1);
    if (beta < 1) {
      gamma = 1 / (2 - beta);
    } else if (beta > 2) {
      gamma = 1 / (beta - 1);
    }
    *update = 
      (((v * v_est.pow(beta - 2)).matrix() * h_T.matrix()).array() / 
      (v_est.pow(beta - 1).matrix() * h_T.matrix()).array()).pow(gamma);
  }
}

template<typename T, typename U>
bool BetaNmf(
    const T& v,
    const U& beta,
    UpdateMode update_mode,
    T* w,
    T* h,
    T* cost) {
  typedef typename T::Scalar Scalar;
  int f = v.rows();
  int n = v.cols();
  int r = w->cols();
  if (!update_mode || beta.rows() < 1) {
    return false;
  }
  if (w->rows() != f || w->cols() != r) {
    return false;
  }
  if (h->rows() != r || h->cols() != n) {
    return false;
  }
  
  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> update(
      w->rows(),
      v.cols());
  Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> v_est(
      w->rows(),
      v.cols());
  Eigen::Array<Scalar, Eigen::Dynamic, 1> w_scale;
  
  v_est = w->matrix() * h->matrix();
  cost->resize(beta.rows() + 1, 1);
  (*cost)(0) = BetaDivergence(v, v_est, beta(0));
  
  for (int i = 0; i < beta.rows(); ++i) {
    if (update_mode & UPDATE_W) {
      ComputeUpdate(v, v_est, *w, h->transpose(), beta(i), &update);
      *w *= update;
      v_est = w->matrix() * h->matrix();
    }
    if (update_mode & UPDATE_H) {
      ComputeUpdate(
          v.transpose(), v_est.transpose(), h->transpose(), *w, beta(i),
          &update);
      *h *= update.transpose();
      v_est = w->matrix() * h->matrix();
    }
    if (update_mode == UPDATE_BOTH) {
      w_scale = w->colwise().sum();
      *w = (w->matrix() * w_scale.inverse().matrix().asDiagonal()).array();
      *h = (w_scale.matrix().asDiagonal() * h->matrix()).array();
    }
    (*cost)(i + 1) = BetaDivergence(v, v_est, beta(i));
  }
  return true;
}

template<typename T>
bool BetaNmf(
    const T& v,
    typename T::Scalar beta,
    int num_iterations,
    UpdateMode update_mode,
    T* w,
    T* h,
    T* cost) {
  typedef typename T::Scalar Scalar;
  
  Eigen::Array<Scalar, Eigen::Dynamic, 1> beta_values(num_iterations);
  for (int i = 0; i < num_iterations; ++i) {
    beta_values[i] = beta;
  }
  return BetaNmf(v, beta_values, update_mode, w, h, cost);
}

}  // namespace beta_nmf

#endif  // BETA_NMF_BETA_NMF_H_
