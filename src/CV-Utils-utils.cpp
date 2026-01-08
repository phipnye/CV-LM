#include "CV-Utils-utils.h"

#include <Rcpp.h>

#include <algorithm>
#include <cfenv>
#include <cmath>

namespace CV::Utils {

// RAII guard for the rounding mode
ScopedRoundingMode::ScopedRoundingMode(const int mode)
    : oldMode_{std::fegetround()} {
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
std::pair<Eigen::VectorXi, Eigen::VectorXi> setupFolds(const int seed,
                                                       const int nrow,
                                                       const int k) {
  // Call back into R for sample and set.seed to guarantee the exact same
  // random partitions as boot::cv.glm (using C++ RNG would break
  // reproducibility)
  const Rcpp::Function setSeed{"set.seed"};
  const Rcpp::Function sampleR{"sample"};
  setSeed(seed);
  const int repeats{static_cast<int>(std::ceil(static_cast<double>(nrow) / k))};
  const Rcpp::IntegerVector seqVec{Rcpp::rep(Rcpp::seq(1, k), repeats)};
  const Rcpp::IntegerVector sampled{sampleR(seqVec, nrow)};
  Eigen::VectorXi foldIDs{Rcpp::as<Eigen::VectorXi>(sampled)};

  // R's internal documentation states the number of rows for a matrix are
  // limited to 32-bit values so VectorXi is safe here
  Eigen::VectorXi foldSizes{Eigen::VectorXi::Zero(k)};

  // Convert foldIDs to be zero-indexed
  foldIDs.array() -= 1;

  // Store fold sizes to calculate the weighted CV estimate: sum((n_i / n) *
  // cost_i)
  for (Eigen::Index idx{0}, size{foldIDs.size()}; idx < size; ++idx) {
    ++foldSizes[foldIDs[idx]];
  }

  return {foldIDs, foldSizes};
}

// Determine the min and max test fold sizes
std::pair<Eigen::Index, Eigen::Index> testSizeExtrema(
    const Eigen::VectorXi& foldSizes) {
  const auto [minIt, maxIt]{std::minmax_element(
      foldSizes.data(), foldSizes.data() + foldSizes.size())};
  return {static_cast<Eigen::Index>(*minIt), static_cast<Eigen::Index>(*maxIt)};
}

}  // namespace CV::Utils
