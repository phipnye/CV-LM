#include "Grid-Generator.h"

#include <Rcpp.h>
#include <RcppEigen.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Grid {

Generator::Generator(const double maxLambda, const double precision,
                     const double threshold)
    : maxLambda_{maxLambda},
      precision_{precision},
      nFull_{[&]() {
        // These checks are already implemented in R, but we add them
        // here to double check because the overhead is small and the
        // consequences can be large
        if (precision <= 0.0) {
          Rcpp::stop("Precision must be > 0");
        }

        const double rawN{std::floor(maxLambda / precision)};

        // Make sure we still have space for size() calls
        constexpr double rawLimit{
            static_cast<double>(std::numeric_limits<Eigen::Index>::max()) -
            2.0};

        if (rawN >= rawLimit) {
          Rcpp::stop(
              "Grid size is too large. Exceeds Eigen::Index limits. Try "
              "increasing precision or reducing max lambda.");
        }

        return static_cast<Eigen::Index>(rawN);
      }()},
      hasTail_{(maxLambda - (static_cast<double>(nFull_) * precision)) >
               threshold} {}

Eigen::Index Generator::size() const noexcept { return nFull_ + 1 + hasTail_; }

double Generator::operator[](const Eigen::Index idx) const {
  return std::min(static_cast<double>(idx) * precision_, maxLambda_);
}

}  // namespace Grid
