#include "Grid-Generator.h"

#include <Rcpp.h>
#include <RcppEigen.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Grid {

Generator::Generator(const double maxLambda, const double precision)
    : maxLambda_{maxLambda},
      precision_{precision},

      // The number of points between 0 and max lambda
      internalN{[&]() -> Eigen::Index {
        // These checks are already implemented in R, but we add them
        // here to double check because the overhead is small and the
        // consequences can be large
        if (precision <= 0.0) {
          Rcpp::stop("Precision must be > 0");
        }

        if (maxLambda <= 0.0) {
          Rcpp::stop("Lambda must be > 0");
        }

        // Make sure we can fit the grid size in an Eigen::Index object
        const double rawN{std::floor(maxLambda / precision)};

        // Determine the absolute ceiling for the grid size (subtract 2.0 to
        // account for internalN + 1 + hasTail_ without overflow)
        constexpr double limitEigen{
            static_cast<double>(std::numeric_limits<Eigen::Index>::max())};
        constexpr double limitSizeT{
            static_cast<double>(std::numeric_limits<std::size_t>::max())};

        // Limit the grid size (this is important because we parallelize over
        // values of lambda for deterministic cv methods, requiring the ability
        // to cast the size to a std::size_t without overflow
        if (constexpr double safetyLimit{std::min(limitEigen, limitSizeT) -
                                         2.0};
            rawN >= safetyLimit) {
          Rcpp::stop(
              "Grid size is too large. Try increasing precision or reducing "
              "max lambda.");
        }

        return static_cast<Eigen::Index>(rawN);
      }()},

      // We explicitly force the max lambda to be included in the grid search
      hasTail_{(maxLambda > (static_cast<double>(internalN) * precision))} {}

Eigen::Index Generator::size() const noexcept {
  // Plus 1 for zero
  return internalN + 1 + hasTail_;
}

double Generator::operator[](const Eigen::Index idx) const noexcept {
  return std::min(static_cast<double>(idx) * precision_, maxLambda_);
}

}  // namespace Grid
