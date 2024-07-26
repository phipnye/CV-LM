# Cross-Validation for Linear & Ridge Regression Models (Rcpp, RcppParallel & Eigen)

This package provides efficient implementations of cross-validation techniques for linear and ridge regression models, leveraging C++ code with Rcpp, RcppParallel, and Eigen libraries. It supports leave-one-out, generalized, and K-fold cross-validation methods, utilizing Eigen matrices for high performance.

## Dependencies

- [Rcpp](https://github.com/RcppCore/Rcpp): Integration between R and C++.
- [RcppParallel](https://github.com/RcppCore/RcppParallel): Parallel computing support for Rcpp.
- [RcppEigen](https://github.com/RcppCore/RcppEigen): Integration between R and Eigen C++ library.

### Requirements

- [R](https://www.r-project.org/)
- [Rcpp](https://cran.r-project.org/package=Rcpp)
- [RcppParallel](https://cran.r-project.org/package=RcppParallel)
- [RcppEigen](https://cran.r-project.org/package=RcppEigen)

### Acknowledgments

This code is adapted and extended from various sources, leveraging the capabilities of the following:

- [Rcpp](https://github.com/RcppCore/Rcpp) by Dirk Eddelbuettel, Romain Francois, et al., for R and C++ integration.
- [RcppParallel](https://github.com/RcppCore/RcppParallel) by Romain Francois, et al., for parallel computing support in Rcpp.
- [RcppEigen](https://github.com/RcppCore/RcppEigen) by Douglas Bates, Romain Francois, et al., for integration between R and Eigen C++ library.

Please refer to the source files for detailed information and licenses.

## Contributors

- [Philip Nye]: [GitHub Profile](https://github.com/phipnye)

## License

This code is under [MIT License](LICENSE).

## Example Usage

```R
library(cvLM)
data(mtcars)
n <- nrow(mtcars)

# Formula method
cvLM(
  mpg ~ .,
  data = mtcars,
  K.vals = n,    # Leave-one-out CV
  lambda = 10    # Shrinkage parameter of 10
)

# lm method
my.lm <- lm(mpg ~ ., data = mtcars)
cvLM(
  my.lm,
  data = mtcars,
  K.vals = c(5L, 8L), # Perform both 5- and 8-fold CV
  n.threads = 8L,     # Allow up to 8 threads for computation
  seed = 1234L
)

# glm method
my.glm <- glm(mpg ~ ., data = mtcars)
cvLM(
  my.glm,
  data = mtcars,
  K.vals = n, generalized = TRUE # Use generalized CV
)
```
