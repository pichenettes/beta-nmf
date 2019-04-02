// Author: Emilie Gillet (emilie.o.gillet@gmail.com)
//
// Computation of beta-divergence.

#ifndef BETA_NMF_DIVERGENCE_H_
#define BETA_NMF_DIVERGENCE_H_

#include <third_party/eigen3/Eigen/Eigen>

namespace beta_nmf {

template<typename T>
typename T::Scalar BetaDivergenceEuclidian(const T& x, const T& y) {
  return 0.5 * ((x - y).pow(2.0)).sum();
}

template<typename T> 
typename T::Scalar BetaDivergenceItakuraSaito(const T& x, const T& y) {
  return (x / y - (x / y).log()).sum() - (x.rows() * x.cols());
}

template<typename T> 
typename T::Scalar BetaDivergenceKullbackLeibler(const T& x, const T& y) {
  return (x * (x / y).log()  + y - x).sum();
}

template<typename T>
typename T::Scalar BetaDivergence(
    const T& x,
    const T& y,
    typename T::Scalar beta) {
  typedef typename T::Scalar Scalar;
  Scalar divergence(0);
  if (beta == Scalar(0.0)) {
    divergence = BetaDivergenceItakuraSaito(x, y);
  } else if (beta == Scalar(1.0)) {
    divergence = BetaDivergenceKullbackLeibler(x, y);
  } else if (beta == Scalar(2.0)) {
    divergence = BetaDivergenceEuclidian(x, y);
  } else {
    divergence = (
        x.pow(beta) + (beta - 1) * y.pow(beta) - beta * x * y.pow(beta - 1)
    ).sum();
    divergence /= (beta * (beta - 1));
  }
  return divergence;
}

}  // namespace beta_nmf

#endif  // BETA_NMF_DIVERGENCE_H_
