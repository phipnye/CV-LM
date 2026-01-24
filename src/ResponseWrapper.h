#ifndef CV_LM_RESPONSEWRAPPER_H
#define CV_LM_RESPONSEWRAPPER_H

#include <RcppEigen.h>

#include <type_traits>

#include "Enums.h"
#include "Utils-Data.h"

// Wrapper around centered response or original response mapping from R
template <Enums::CenteringMethod Centering>
class ResponseWrapper {
  // Boolean controls whether the data is a map or physical vector
  static constexpr bool meanCenter{Centering == Enums::CenteringMethod::Mean};
  using MapType = Eigen::Map<const Eigen::VectorXd>;
  using StorageType = std::conditional_t<meanCenter, Eigen::VectorXd, MapType>;
  const StorageType storage_;

  // Ctor helper
  template <typename Derived>
  [[nodiscard]] static StorageType init(const Eigen::MatrixBase<Derived>& y) {
    Utils::Data::assertColumnVector(y);

    if constexpr (meanCenter) {
      return Utils::Data::centerResponse(y);
    } else {
      return MapType{y.derived().data(), y.size()};
    }
  }

 public:
  // Main ctor
  template <typename Derived>
  explicit ResponseWrapper(const Eigen::MatrixBase<Derived>& y)
      : storage_{init(y)} {}

  // Retrieve underlying data
  [[nodiscard]] const StorageType& value() const noexcept { return storage_; }
};

#endif  // CV_LM_RESPONSEWRAPPER_H
