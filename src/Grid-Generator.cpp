#include "Grid-Generator.h"

#include <RcppArmadillo.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace Grid {

Generator::Generator(const double maxLambda, const double precision)
    : maxLambda_{maxLambda}, precision_{precision}, size_{[&]() -> arma::uword {
        if (precision <= 0.0) {
          Rcpp::stop("precision must be > 0");
        }

        if (maxLambda <= 0.0) {
          Rcpp::stop("max.lambda must be > 0");
        }

        // --- Make sure we can fit the grid size in an arma::uword and a
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
        // arma::uword)
        constexpr double limitArma{
            static_cast<double>(std::numeric_limits<arma::uword>::max())};
        constexpr double limitSizeT{
            static_cast<double>(std::numeric_limits<std::size_t>::max())};

        // Limit the grid size (this is important because we parallelize over
        // values of lambda for deterministic cv methods, requiring the ability
        // to cast the size to a std::size_t without overflow
        if (constexpr double safetyLimit{std::min(limitArma, limitSizeT)};
            rawN >= safetyLimit) {
          Rcpp::stop(
              "Grid size is too large. Try increasing precision or reducing "
              "max lambda.");
        }

        return static_cast<arma::uword>(rawN);
      }()} {}

arma::uword Generator::size() const noexcept { return size_; }

double Generator::operator[](const arma::uword idx) const noexcept {
  return std::min(static_cast<double>(idx) * precision_, maxLambda_);
}

}  // namespace Grid
