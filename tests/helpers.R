library(testthat)
library(boot)
library(cvLM)

make.data <- function(n = 100L, p = 5L, seed = 1L, intercept = TRUE) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  beta <- runif(p)
  y <- X %*% beta + rnorm(n)
  df <- as.data.frame(X)
  names(df) <- paste0("x", seq_len(p))
  df$y <- as.numeric(y)

  if (intercept) {
    form <- y ~ .
  } else {
    form <- y ~ . - 1
  }

  list(df = df, formula = form, X = X, y = y)
}

boot.cv.ols <- function(formula, data, K, seed) {
  fit <- glm(formula, data = data)
  set.seed(seed)
  boot::cv.glm(data, fit, K = K)$delta[1L]
}
