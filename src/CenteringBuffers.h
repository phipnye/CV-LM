#pragma once

#include <RcppEigen.h>

#include <type_traits>
#include <utility>

#include "ConstexprOptional.h"
#include "Utils-Data.h"

// Data container for workers that need to do local mean centering of data
// if meanCenter is false, the optional buffers are 1-byte empty classes
// and the data members will cause build errors if dereferencing them is
// attempted
template <bool meanCenter>
class CenteringBuffers {
  // Data members
  using OptionalMatrix = ConstexprOptional<meanCenter, Eigen::MatrixXd>;
  using OptionalVector = ConstexprOptional<meanCenter, Eigen::VectorXd>;
  using OptionalDouble = ConstexprOptional<meanCenter, double>;
  OptionalMatrix xTrainCentered_;
  OptionalVector yTrainCentered_;
  OptionalMatrix xTestCentered_;
  OptionalVector yTestCentered_;
  OptionalVector xTrainColMeans_;
  OptionalDouble yTrainMean_;

  // Helper for main ctor
  template <typename T, typename... Args>
  static T initOptional(Args&&... args) {
    if constexpr (meanCenter) {
      return T{std::forward<Args>(args)...};
    } else {
      return T::empty();
    }
  }

  // Helper for copy ctor
  template <typename T>
  static T cloneShape(const T& other) {
    if constexpr (meanCenter) {
      if constexpr (std::is_same_v<T, OptionalMatrix>) {
        return T{(*other).rows(), (*other).cols()};
      } else if constexpr (std::is_same_v<T, OptionalVector>) {
        return T{(*other).size()};
      } else {
        static_assert(std::is_same_v<T, OptionalDouble>,
                      "Unexpected cloning type");
        return T{0.0};
      }
    } else {
      return T::empty();
    }
  }

 public:
  // Main ctor
  CenteringBuffers(const Eigen::Index ncol, const Eigen::Index maxTrainSize,
                   const Eigen::Index maxTestSize)
      : xTrainCentered_{initOptional<OptionalMatrix>(maxTrainSize, ncol)},
        yTrainCentered_{initOptional<OptionalVector>(maxTrainSize)},
        xTestCentered_{initOptional<OptionalMatrix>(maxTestSize, ncol)},
        yTestCentered_{initOptional<OptionalVector>(maxTestSize)},
        xTrainColMeans_{initOptional<OptionalVector>(ncol)},
        yTrainMean_{initOptional<OptionalDouble>(0.0)} {}

  // Copy ctor (just pre-allocates identical buffer sizes - no data copying)
  CenteringBuffers(const CenteringBuffers& other)
      : xTrainCentered_{cloneShape(other.xTrainCentered_)},
        yTrainCentered_{cloneShape(other.yTrainCentered_)},
        xTestCentered_{cloneShape(other.xTestCentered_)},
        yTestCentered_{cloneShape(other.yTestCentered_)},
        xTrainColMeans_{cloneShape(other.xTrainColMeans_)},
        yTrainMean_{cloneShape(other.yTrainMean_)} {}

  // Method to center the training data in-place
  template <bool B = meanCenter, typename = std::enable_if_t<B>,
            typename DerivedX, typename DerivedY>
  void centerData(const Eigen::MatrixBase<DerivedX>& xTrain,
                  const Eigen::MatrixBase<DerivedY>& yTrain,
                  const Eigen::MatrixBase<DerivedX>& xTest,
                  const Eigen::MatrixBase<DerivedY>& yTest) {
    // Center the training data
    Utils::Data::assertColumnVector(yTrain);
    Utils::Data::centerData(xTrain, yTrain, *xTrainCentered_, *yTrainCentered_,
                            *xTrainColMeans_, *yTrainMean_);

    // Apply the same centering procedure to the procedure testing set
    const Eigen::Index testSize{xTest.rows()};
    (*xTestCentered_).topRows(testSize) =
        xTest.rowwise() - (*xTrainColMeans_).transpose();
    (*yTestCentered_).head(testSize) = yTest.array() - *yTrainMean_;
  }

  // Getters for centered data
  template <bool B = meanCenter, typename = std::enable_if_t<B>>
  [[nodiscard]] const Eigen::MatrixXd& getXTrain() const noexcept {
    return *xTrainCentered_;
  }

  template <bool B = meanCenter, typename = std::enable_if_t<B>>
  [[nodiscard]] const Eigen::VectorXd& getYTrain() const noexcept {
    return *yTrainCentered_;
  }

  template <bool B = meanCenter, typename = std::enable_if_t<B>>
  [[nodiscard]] const Eigen::MatrixXd& getXTest() const noexcept {
    return *xTestCentered_;
  }

  template <bool B = meanCenter, typename = std::enable_if_t<B>>
  [[nodiscard]] const Eigen::VectorXd& getYTest() const noexcept {
    return *yTestCentered_;
  }
};
