#ifndef CV_LM_UTILS_FOLDS_H
#define CV_LM_UTILS_FOLDS_H

#include <RcppEigen.h>

#include <type_traits>
#include <utility>

#include "ConstexprOptional.h"
#include "Enums.h"
#include "Utils-Data.h"

namespace Utils::Folds {

// RAII guard for setting rounding mode
class ScopedRoundingMode {
  // Store the original rounding mode so we can revert back to it once our
  // object goes out-of-scope
  const int oldMode_;

 public:
  explicit ScopedRoundingMode(int mode);
  ~ScopedRoundingMode();
};

// Confirm valid value for the number of folds
[[nodiscard]] int kCheck(int nrow, int k0, bool generalized);

// Container for holding assigned fold information
class DataSplitter {
  // Eigen objects
  const Eigen::VectorXi testIDs_;
  const Eigen::VectorXi testSizes_;
  const Eigen::VectorXi testStarts_;

  // Sizes
  const Eigen::Index maxTestSize_;
  const Eigen::Index maxTrainSize_;
  const Eigen::Index nrow_;

 public:
  // Ctor
  explicit DataSplitter(int seed, Eigen::Index nrow, int k);

  // We should just be passing this object by ref to stochastic workers
  DataSplitter(const DataSplitter&) = delete;

  // Construct indices to permute the design matrix and response vector such
  // that test observations are stored contiguously
  [[nodiscard]] Eigen::VectorXi buildPermutation() const;

  // Retrieve the current test fold start and size
  [[nodiscard]] std::pair<int, int> operator[](Eigen::Index idx) const;

  // Get the max training set size
  [[nodiscard]] Eigen::Index maxTrain() const noexcept;

  // Get the max test set size
  [[nodiscard]] Eigen::Index maxTest() const noexcept;
};

// Container in charge of partitioning (and mean centering if necessary) the
// test and training data sets
template <Enums::CenteringMethod Centering>
class DataLoader {
  // Static boolean to check if we mean center the data
  static constexpr bool meanCenter{Centering == Enums::CenteringMethod::Mean};

  // --- Data members

  // References to "sorted" design matrix and response vector
  const Eigen::MatrixXd& xSorted_;
  const Eigen::VectorXd& ySorted_;

  // Buffers for storing contiguous training set data
  Eigen::MatrixXd xTrain_;
  Eigen::VectorXd yTrain_;

  // Buffers for storing data for centering the test data if necessary
  using OptionalMatrix = ConstexprOptional<meanCenter, Eigen::MatrixXd>;
  using OptionalVector = ConstexprOptional<meanCenter, Eigen::VectorXd>;
  using OptionalDouble = ConstexprOptional<meanCenter, double>;
  OptionalMatrix xTest_;
  OptionalVector yTest_;
  OptionalVector xTrainColMeans_;
  OptionalDouble yTrainMean_;

  // --- Helpers

  // POD return type for prepData
  struct Refs {
    using MatrixRef = Eigen::Ref<const Eigen::MatrixXd>;
    using VectorRef = Eigen::Ref<const Eigen::VectorXd>;
    MatrixRef xTrain_;
    VectorRef yTrain_;
    MatrixRef xTest_;
    VectorRef yTest_;
  };

 public:
  // Main ctor
  DataLoader(const Eigen::VectorXd& ySorted, const Eigen::MatrixXd& xSorted,
             const Eigen::Index maxTrainSize, const Eigen::Index maxTestSize)
      : xSorted_{xSorted},
        ySorted_{ySorted},
        xTrain_{maxTrainSize, xSorted_.cols()},
        yTrain_{maxTrainSize},
        xTest_{OptionalMatrix::make(maxTestSize, xSorted_.cols())},
        yTest_{OptionalVector::make(maxTestSize)},
        xTrainColMeans_{OptionalVector::make(xSorted_.cols())},
        yTrainMean_{OptionalDouble::make(0.0)} {}

  // Copy ctor (just pre-allocates identical buffer sizes - no data copying)
  DataLoader(const DataLoader& other)
      : xSorted_{other.xSorted_},
        ySorted_{other.ySorted_},
        xTrain_{other.xTrain_.rows(), other.xTrain_.cols()},
        yTrain_{other.yTrain_.size()},
        xTest_{other.xTest_.clone()},
        yTest_{other.yTest_.clone()},
        xTrainColMeans_{other.xTrainColMeans_.clone()},
        yTrainMean_{other.yTrainMean_.clone()} {}

  // Method to center the training data in-place
  [[nodiscard]] Refs prepData(const Eigen::Index testStart,
                              const Eigen::Index testSize) {
    // Number of rows remaining in the sorted design matrix after the test block
    const Eigen::Index trainSize{xSorted_.rows() - testSize};
    const Eigen::Index remaining{trainSize - testStart};

    // Consolidate training data into buffers (everything before and after the
    // test block)
    auto xTrain{xTrain_.topRows(trainSize)};
    auto yTrain{yTrain_.head(trainSize)};
    xTrain.topRows(testStart) = xSorted_.topRows(testStart);
    yTrain.head(testStart) = ySorted_.head(testStart);
    xTrain.bottomRows(remaining) = xSorted_.bottomRows(remaining);
    yTrain.tail(remaining) = ySorted_.tail(remaining);

    // Retrieve the test data blocks
    const auto xSortedTestBlock{
        xSorted_.block(testStart, 0, testSize, xSorted_.cols())};
    const auto ySortedTestBlock{ySorted_.segment(testStart, testSize)};

    if constexpr (meanCenter) {
      // Copy test data into buffers
      auto xTest{xTest_.value().topRows(testSize)};
      auto yTest{yTest_.value().head(testSize)};
      xTest = xSortedTestBlock;
      yTest = ySortedTestBlock;

      // Center the training data and then apply identical mean centering to the
      // test data set
      centerTrainData(xTrain, yTrain);
      centerTestData(xTest, yTest);
      return Refs{xTrain, yTrain, xTest, yTest};
    } else {
      return Refs{xTrain, yTrain, xSortedTestBlock, ySortedTestBlock};
    }
  }

 private:
  // Center the training set and store the training set column means
  template <bool B = meanCenter, typename = std::enable_if_t<B>,
            typename DerivedX, typename DerivedY>
  void centerTrainData(Eigen::MatrixBase<DerivedX>& xTrain,
                       Eigen::MatrixBase<DerivedY>& yTrain) {
    // Make sure the data is what we expect
    Data::assertDataStructure(xTrain, yTrain);

    // Center the training data and store the column means so we can apply the
    // same to the test data
    Data::centerData(xTrain, yTrain, xTrainColMeans_.value(),
                     yTrainMean_.value());
  }

  // Apply training data column means to the test sets
  template <bool B = meanCenter, typename = std::enable_if_t<B>,
            typename DerivedX, typename DerivedY>
  void centerTestData(Eigen::MatrixBase<DerivedX>& xTest,
                      Eigen::MatrixBase<DerivedY>& yTest) {
    // Make sure the data is what we expect
    Data::assertDataStructure(xTest, yTest);

    // Apply the training set means to the test data
    xTest.rowwise() -= xTrainColMeans_.value().transpose();
    yTest.array() -= yTrainMean_.value();
  }
};

}  // namespace Utils::Folds

#endif  // CV_LM_UTILS_FOLDS_H
