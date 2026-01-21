## data-utils.R: Data preparation for C++ tasks
##
## This file is part of the cvLM package.

.drop_intercept <- function(X) {
  # Identify the intercept column
  intercept.col <- which(colnames(X) == "(Intercept)")

  # Drop the intercept if it exists
  if (length(intercept.col) > 0L) {
    X <- X[, -intercept.col, drop = FALSE]
  }

  X
}
