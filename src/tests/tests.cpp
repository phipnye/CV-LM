#define EIGEN_RUNTIME_NO_MALLOC
#include <RcppEigen.h>

#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "../ConstexprOptional.h"
#include "../Enums.h"
#include "../Utils-Folds.h"

TEST_CASE("Verify loader::prepData is strictly malloc-free") {
  using Utils::Folds::DataLoader;
  using namespace Enums;
  constexpr Eigen::Index ncol{2};
  constexpr Eigen::Index nrow{15};
  constexpr Eigen::Index testStart{5};
  constexpr Eigen::Index testSize{4};
  constexpr Eigen::Index trainSize{nrow - testSize};
  const Eigen::MatrixXd xSorted{Eigen::MatrixXd::Random(nrow, ncol)};
  const Eigen::VectorXd ySorted{Eigen::VectorXd::Random(nrow)};

  SECTION("Mean centering enabled") {
    DataLoader<CenteringMethod::Mean> fd{ySorted, xSorted, trainSize, testSize};
    Eigen::internal::set_is_malloc_allowed(false);
    const auto refs{fd.prepData(testStart, testSize)};
    Eigen::internal::set_is_malloc_allowed(true);
    std::cout << "\n================= MEAN CENTERED =================\n";
    std::cout << "Refs.xTest:\n" << refs.xTest_ << "\n\n";
    std::cout << "Refs.xTrain:\n" << refs.xTrain_ << "\n";
    REQUIRE((refs.xTrain_.rows() == trainSize));
    REQUIRE((refs.xTest_.rows() == testSize));
  }

  SECTION("Mean centering disabled") {
    DataLoader<CenteringMethod::None> fd{ySorted, xSorted, trainSize, testSize};
    Eigen::internal::set_is_malloc_allowed(false);
    const auto refs{fd.prepData(testStart, testSize)};
    Eigen::internal::set_is_malloc_allowed(true);
    std::cout << "\n=============== NO MEAN CENTERING ===============\n";
    std::cout << "Refs.xTest:\n" << refs.xTest_ << "\n\n";
    std::cout << "Refs.xTrain:\n" << refs.xTrain_ << "\n";
    REQUIRE((refs.xTrain_.rows() == trainSize));
    REQUIRE((refs.xTest_.rows() == testSize));
  }
}

TEST_CASE(
    "Ensure ConstexprOptional clones shapes of Eigen objects and copies other "
    "data") {
  SECTION("Eigen matrix clones shape but not values") {
    using OptionalMatrix = ConstexprOptional<true, Eigen::MatrixXd>;
    constexpr Eigen::Index nrow{3};
    constexpr Eigen::Index ncol{4};
    OptionalMatrix optMat{OptionalMatrix::make(nrow, ncol)};

    // Fill original with known values
    constexpr double constVal{42.0};
    optMat.value().setConstant(constVal);
    const auto cloned{optMat.clone()};
    REQUIRE(cloned.value().rows() == nrow);     // data shape should be copied
    REQUIRE(cloned.value().cols() == ncol);     // data shape should be copied
    REQUIRE(cloned.value()(0, 0) != constVal);  // values should not be copied
    std::cout << "\n===== Eigen Matrix Clone Test =====\n";
    std::cout << "Original matrix:\n" << optMat.value() << "\n\n";
    std::cout << "Cloned matrix:\n" << cloned.value() << "\n\n";
  }

  SECTION("Eigen vector clones shape but not values") {
    using OptionalVector = ConstexprOptional<true, Eigen::VectorXd>;
    constexpr Eigen::Index size{5};
    OptionalVector optVec{OptionalVector::make(size)};
    constexpr double constVal{42.0};
    optVec.value().setConstant(constVal);
    const auto cloned{optVec.clone()};
    REQUIRE(cloned.value().rows() == 5);     // data shape should be copied
    REQUIRE(cloned.value().cols() == 1);     // data shape should be copied
    REQUIRE(cloned.value()[0] != constVal);  // values should not be copied
    std::cout << "\n===== Eigen Vector Clone Test =====\n";
    std::cout << "Original vector:\n" << optVec.value().transpose() << "\n";
    std::cout << "Cloned vector:\n" << cloned.value().transpose() << "\n\n";
  }

  SECTION("Non-Eigen types are copied") {
    using OptionalInt = ConstexprOptional<true, int>;
    using OptionalDbl = ConstexprOptional<true, double>;
    constexpr int intVal{123};
    constexpr double dblVal{52.46};
    constexpr OptionalInt optInt{OptionalInt::make(intVal)};
    constexpr OptionalDbl optDbl{OptionalDbl::make(dblVal)};
    constexpr auto clonedInt{optInt.clone()};
    constexpr auto clonedDbl{optDbl.clone()};
    REQUIRE(clonedInt.value() == intVal);
    REQUIRE(clonedDbl.value() == dblVal);
    std::cout << "\n===== Non-Eigen Clone Test =====\n";
    std::cout << "Original int: " << optInt.value() << "\n";
    std::cout << "Cloned int:   " << clonedInt.value() << "\n";
    std::cout << "Original dbl: " << optDbl.value() << "\n";
    std::cout << "Cloned dbl:   " << clonedDbl.value() << "\n";
  }
}
