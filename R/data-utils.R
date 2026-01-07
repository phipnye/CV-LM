## data-utils.R: Data preparation for C++ tasks
##
## This file is part of the cvLM package.

.center_data <- function(y, X, mt) {
  # Identify the intercept column
  intercept.col <- which(colnames(X) == "(Intercept)")

  if (length(intercept.col) > 0L) {
    # Center y
    y <- y - mean(y)

    # Drop intercept and center X columns (when you center y and X, the regression line is forced to pass
    # through the origin of the new coordinate system, where the intercept is mathematically guaranteed to be
    # zero, hence a column of ones is not needed anymore
    X <- X[, -intercept.col, drop = FALSE]
    X <- sweep(X, 2L, colMeans(X), FUN = "-", check.margin = FALSE)
  }

  list(y = y, X = X)
}
