#include "Grid-Generator.h"

#include <Rcpp.h>
#include <RcppEigen.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace Grid {

Generator::Generator(const double maxLambda, const double precision)
    : maxLambda_{maxLambda},
      precision_{precision},
      size_{[&]() -> Eigen::Index {
        // These checks are already implemented in R, but we add them here to
        // double check because the overhead is small and the consequences can
        // be large
        if (precision <= 0.0) {
          Rcpp::stop("precision must be > 0");
        }

        if (maxLambda <= 0.0) {
          Rcpp::stop("max.lambda must be > 0");
        }

        // --- Make sure we can fit the grid size in an Eigen::Index and a
        // std::size_t

        const double floorDiv{std::floor(maxLambda / precision)};
        const double rawN{
            // Add 1 for zero and likely another one to force inclusion of
            // maxLambda in the grid (e.g., [0, 0.5, 1.0, 1.2] if maxLambda
            // == 1.2 and precision = 0.5)
            floorDiv + 1.0 +
            ((maxLambda > (static_cast<double>(floorDiv) * precision)) ? 1.0
                                                                       : 0.0)};

        // Determine the absolute ceiling for the grid size (througout our code,
        // we need to be able to convert the size to a std::size_t and an
        // Eigen::Index)
        constexpr double limitEigen{
            static_cast<double>(std::numeric_limits<Eigen::Index>::max())};
        constexpr double limitSizeT{
            static_cast<double>(std::numeric_limits<std::size_t>::max())};

        // Limit the grid size (this is important because we parallelize over
        // values of lambda for deterministic cv methods, requiring the ability
        // to cast the size to a std::size_t without overflow
        if (constexpr double safetyLimit{std::min(limitEigen, limitSizeT)};
            rawN >= safetyLimit) {
          Rcpp::stop(
              "Grid size is too large. Try increasing precision or reducing "
              "max lambda.");
        }

        return static_cast<Eigen::Index>(rawN);
      }()} {}

Eigen::Index Generator::size() const noexcept { return size_; }

double Generator::operator[](const Eigen::Index idx) const noexcept {
  return std::min(static_cast<double>(idx) * precision_, maxLambda_);
}

}  // namespace Grid
