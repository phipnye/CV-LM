source("tests/helpers.R")

test_that("cvLM ridge regression reduces to OLS as lambda approaches 0", {
  # Sample sizes and predictor counts
  sample.sizes <- c(20, 50, 100, 1000)
  predictors <- c(2, 4, 6)
  seeds <- c(1L, 42L)
  
  # Other parameters
  K.vals.list <- list(small = c(2,3), typical = c(5, 10), full = NULL) # include LOOCV
  lambdas <- c(1e-20) # near-zero value to test against the 0 baseline
  generalized.opts <- c(FALSE, TRUE)
  # penalize.intercept.opts <- c(TRUE, FALSE)
  penalize.intercept.opts <- FALSE
  na.actions <- list(na.omit)
  
  for (n in sample.sizes) {
    for (p in predictors) {
      if (p >= n) {
        next
      } # skip wide case for OLS
      
      dat <- make.data(n = n, p = p)
      df <- dat$df
      
      for (seed in seeds) {
        for (K.name in names(K.vals.list)) {
          K.vals <- K.vals.list[[K.name]]
          
          if (is.null(K.vals)) {
            K.vals <- n
          } # LOOCV
          
          for (lambda in lambdas) {
            for (generalized in generalized.opts) {
              for (penalize.intercept in penalize.intercept.opts) {
                for (na.action in na.actions) {
                  
                  # Run OLS baseline (lambda = 0) with current settings
                  res.ols <- cvLM(
                    dat$formula,
                    data = df,
                    na.action = na.action,
                    K.vals = K.vals,
                    lambda = 0,
                    generalized = generalized,
                    seed = seed,
                    n.threads = 1,
                    penalize.intercept = penalize.intercept
                  )
                  
                  # Run Ridge (lambda > 0) with identical settings
                  res.ridge <- cvLM(
                    dat$formula,
                    data = df,
                    na.action = na.action,
                    K.vals = K.vals,
                    lambda = lambda,
                    generalized = generalized,
                    seed = seed,
                    n.threads = 1,
                    penalize.intercept = penalize.intercept
                  )
                  
                  # Compare CV results
                  diff.ols <- abs(res.ols$CV - res.ridge$CV)
                  
                  msg.base <- paste(
                    "n=",
                    n,
                    ", p=",
                    p,
                    ", K=",
                    paste(K.vals, collapse = ","),
                    ", lambda=",
                    lambda,
                    ", generalized=",
                    generalized,
                    ", seed=",
                    seed,
                    ", n.threads=1",
                    ", penalize.intercept=",
                    penalize.intercept,
                    ", na.action=",
                    deparse(substitute(na.action))
                  )
                  
                  expect_true(
                    all(diff.ols < 1e-8),
                    info = paste(
                      "Ridge to OLS convergence failure:",
                      msg.base,
                      "abs(diff)=",
                      diff.ols
                    )
                  )
                }
              }
            }
          }
        }
      }
    }
  }
})
