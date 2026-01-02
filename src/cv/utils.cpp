#include "include/cv/utils.h"

#include <cfenv>
#include <cmath>

namespace CV::Utils {

// RAII guard for the rounding mode
ScopedRoundingMode::ScopedRoundingMode(int mode) : oldMode_{std::fegetround()} {
  std::fesetround(mode);
}

ScopedRoundingMode::~ScopedRoundingMode() { std::fesetround(oldMode_); }

// Confirm valid value for K
// Mimics boot::cv.glm logic to find a K that evenly (or nearly evenly) divides
// n (uses FE_TONEAREST to match R's banker rounding)
int kCheck(const int nrow, const int k0) {
  // LOOCV
  if (nrow == k0) {
    return k0;
  }

  const ScopedRoundingMode roundGuard{
      FE_TONEAREST};  // round to nearest, ties to even
  const double nDbl{static_cast<double>(nrow)};
  const int nHalf{nrow / 2};
  int closestK{nrow};
  int minDiff{closestK - k0};

  // Consider k values between n and 2 (iterates through possible denominators
  // to find a K value that fits n, (like the kvals <-
  // unique(round(n/(1L:floor(n/2)))) line in cv.glm)
  for (int den{1}; den < nHalf; ++den) {
    const int kVal{static_cast<int>(
        std::nearbyint(nDbl / (den + 1)))};  // use banker's rounding
    const int absDiff{std::abs(k0 - kVal)};

    if (absDiff == 0) {
      return k0;
    }

    if (absDiff < minDiff) {
      minDiff = absDiff;
      closestK = kVal;
    }
  }

  Rcpp::warning("K has been changed from %d to %d.", k0, closestK);
  return closestK;
}

// Generates fold assignments
std::pair<Eigen::VectorXi, Eigen::VectorXi> cvSetup(const int seed,
                                                    const int nrow,
                                                    const int k) {
  // Call back into R for sample and set.seed to guarantee the exact same
  // random partitions as boot::cv.glm (using C++ RNG would break
  // reproducibility against the R version)
  Rcpp::Function setSeed{"set.seed"};
  Rcpp::Function sampleR{"sample"};
  setSeed(seed);
  const int repeats{static_cast<int>(std::ceil(static_cast<double>(nrow) / k))};
  Rcpp::IntegerVector seqVec{Rcpp::rep(Rcpp::seq(1, k), repeats)};
  Rcpp::IntegerVector sampled{sampleR(seqVec, nrow)};
  Eigen::VectorXi foldIDs{Rcpp::as<Eigen::VectorXi>(sampled)};
  Eigen::VectorXi foldSizes{Eigen::VectorXi::Zero(k)};

  // Store fold sizes to calculate the weighted CV estimate: sum((n_i / n) *
  // cost_i)
  for (int fold{0}; fold < k; ++fold) {
    foldSizes(fold) = (foldIDs.array() == (fold + 1)).count();
  }

  return {foldIDs, foldSizes};
}

}  // namespace CV::Utils
