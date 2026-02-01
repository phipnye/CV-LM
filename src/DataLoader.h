#ifndef CV_LM_DATALOADER_H
#define CV_LM_DATALOADER_H

#include <RcppArmadillo.h>

class DataLoader {
  arma::mat Xsorted_;      // rows sorted by fold ID
  arma::vec ySorted_;      // rows sorted by fold ID
  arma::uvec testIDs_;     // fold assignment for each original row
  arma::uvec testStarts_;  // starting index of each fold in sorted data
  arma::uvec testSizes_;   // number of rows in each fold
  arma::uword nrow_;
  arma::uword ncol_;
  arma::uword maxTrainSize_{0};
  arma::uword maxTestSize_{0};

  // POD structure for returning values
  struct LoadValues {
    arma::subview<double> Xtest_;
    arma::subview_col<double> yTest_;
    arma::uword testSize_;
    arma::uword trainSize;
  };

 public:
  explicit DataLoader(const arma::mat& X, const arma::vec& y, int seed,
                      arma::uword k);

  [[nodiscard]] LoadValues load(arma::uword testID, arma::mat& XtrainBuf,
                                arma::vec& yTrainBuf) const;

  [[nodiscard]] arma::uword maxTrain() const noexcept;
  [[nodiscard]] arma::uword maxTest() const noexcept;
  [[nodiscard]] arma::uword nrow() const noexcept;
  [[nodiscard]] arma::uword ncol() const noexcept;
};

#endif  // CV_LM_DATALOADER_H
