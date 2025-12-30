#include "include/utils.h"

#include <cfenv>
#include <cmath>

namespace CV::Utils {

// Confirm valid value for K
// Mimics boot::cv.glm logic to find a K that evenly (or nearly evenly) divides
// n (uses FE_TONEAREST to match R's banker rounding)
int kCheck(const int n, const int k0) {
  // LOOCV
  if (n == k0) {
    return k0;
  }

  const int oldMode{std::fegetround()};
  std::fesetround(FE_TONEAREST);  // round to nearest, ties to even
  const double nDbl{static_cast<double>(n)};
  const int nHalf{n / 2};
  int closestK{n};
  int minDiff{closestK - k0};

  // Consider k values between n and 2 (iterates through possible denominators
  // to find a K value that fits n, (like the kvals <-
  // unique(round(n/(1L:floor(n/2)))) line in cv.glm)
  for (int den{1}; den < nHalf; ++den) {
    const int kVal{static_cast<int>(
        std::nearbyint(nDbl / (den + 1)))};  // use banker's rounding
    const int absDiff{std::abs(k0 - kVal)};

    if (absDiff == 0) {
      std::fesetround(oldMode);
      return k0;
    }

    if (absDiff < minDiff) {
      minDiff = absDiff;
      closestK = kVal;
    }
  }

  Rcpp::warning("K has been changed from %d to %d.", k0, closestK);
  std::fesetround(oldMode);
  return closestK;
}

// Calculate MSE: standard cost function = mean((y - yhat)^2)
double cost(const Eigen::VectorXd& y, const Eigen::VectorXd& yHat) {
  return (y - yHat).array().square().mean();
}

// Generates fold assignments
std::pair<Eigen::VectorXi, Eigen::VectorXd> cvSetup(const int seed, const int n,
                                                    const int k) {
  // Call back into R for sample and set.seed to guarantee the exact same
  // random partitions as boot::cv.glm (using C++ RNG would break
  // reproducibility against the R version)
  Rcpp::Function setSeed{"set.seed"};
  Rcpp::Function sampleR{"sample"};
  setSeed(seed);
  const int repeats{static_cast<int>(std::ceil(static_cast<double>(n) / k))};
  Rcpp::IntegerVector seqVec{Rcpp::rep(Rcpp::seq(1, k), repeats)};
  Rcpp::IntegerVector sampled{sampleR(seqVec, n)};
  Eigen::VectorXi s{Rcpp::as<Eigen::VectorXi>(sampled)};
  Eigen::VectorXd ns{Eigen::VectorXd::Zero(k)};

  // Store fold sizes to calculate the weighted CV estimate: sum((n_i / n) *
  // cost_i)
  for (int i{0}; i < k; ++i) {
    ns(i) = (s.array() == (i + 1)).count();
  }

  return {s, ns};
}

}  // namespace CV::Utils
