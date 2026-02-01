# Cross-Validation for Linear & Ridge Regression Models (RcppArmadillo & RcppParallel)

This package provides efficient implementations of cross-validation techniques for linear and ridge regression models, leveraging C++17 with RcppArmadillo and RcppParallel. It supports leave-one-out, generalized, and K-fold cross-validation methods, utilizing Singular Value Decomposition (SVD) and Complete Orthogonal Decomposition (COD) for high performance and numerical stability in high-dimensional settings.

## Dependencies

- [Rcpp](https://github.com/RcppCore/Rcpp): Integration between R and C++.
- [RcppParallel](https://github.com/RcppCore/RcppParallel): Parallel computing support for Rcpp.
- [RcppArmadillo](https://github.com/RcppCore/RcppArmadillo): Integration between R and the Armadillo C++ library.
- [RhpcBLASctl](https://cran.r-project.org/package=RhpcBLASctl): Control of BLAS/LAPACK thread counts.

### Requirements

- [R](https://www.r-project.org/)
- [Rcpp](https://cran.r-project.org/package=Rcpp)
- [RcppParallel](https://cran.r-project.org/package=RcppParallel)
- [RcppArmadillo](https://cran.r-project.org/package=RcppArmadillo)
- C++17 compatible compiler

### Acknowledgments

This code is adapted and extended from various sources, leveraging the capabilities of the following:

- [Rcpp](https://github.com/RcppCore/Rcpp) by Dirk Eddelbuettel, Romain Francois, et al., for R and C++ integration.
- [RcppParallel](https://github.com/RcppCore/RcppParallel) by JJ Allaire, Romain Francois, et al., for parallel computing support.
- [RcppArmadillo](https://github.com/RcppCore/RcppArmadillo) by Dirk Eddelbuettel, Conrad Sanderson, et al., for high-performance linear algebra.

Please refer to the source files for detailed information and licenses.

## Contributors

- [Philip Nye]: [GitHub Profile](https://github.com/phipnye)

## License

This code is under [MIT License](LICENSE).

## Example Usage

```R
library(cvLM)
data(mtcars)

# 10-fold CV for a linear regression model
cvLM(mpg ~ ., data = mtcars, K.vals = 10)

# Comparing 5-fold, 10-fold, and Leave-One-Out CV configurations using 2 threads
cvLM(mpg ~ ., data = mtcars, K.vals = c(5, 10, nrow(mtcars)), n.threads = 2)

# Ridge regression with analytic GCV (using lm interface)
fitted.lm <- lm(mpg ~ ., data = mtcars)
cvLM(fitted.lm, data = mtcars, lambda = 0.5, generalized = TRUE)

grid.search(
  formula = mpg ~ ., 
  data = mtcars,
  K = 5L,           # Use 5-fold CV
  max.lambda = 100, # Search values between 0 and 100
  precision = 0.01  # Increment in steps of 0.01
)
```