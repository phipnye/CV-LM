source("tests/helpers.R")

test_that("cvLM produces consistent results regardless of the number of threads used (Serial vs Parallel)", {
  # Sample sizes and predictor counts
  sample.sizes <- c(20, 50, 100, 1000)
  predictors <- c(2, 4, 6)
  seeds <- c(1L, 42L)
  
  # Other parameters
  K.vals.list <- list(few = c(2, 3), typical = c(5, 10), full = NULL) # include LOOCV
  lambdas <- c(
    0,
    0.062,
    0.523,
    1.544,
    13.324,
    34.324,
    325.141,
    1204.1457,
    4936.5067
  )
  generalized.opts <- c(FALSE, TRUE)
  penalize.intercept.opts <- c(FALSE, TRUE)
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
                if (lambda == 0 && penalize.intercept) {
                  next
                } # invalid
                
                for (na.action in na.actions) {
                  
                  # Run with Single Thread (Serial)
                  res.serial <- cvLM(
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
                  
                  # Run with Multiple Threads (Parallel)
                  # We use 2 threads to verify parallel logic works
                  res.parallel <- cvLM(
                    dat$formula,
                    data = df,
                    na.action = na.action,
                    K.vals = K.vals,
                    lambda = lambda,
                    generalized = generalized,
                    seed = seed,
                    n.threads = 2,
                    penalize.intercept = penalize.intercept
                  )
                  
                  # Compare CV results
                  diff.threads <- abs(res.serial$CV - res.parallel$CV)
                  
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
                    ", penalize.intercept=",
                    penalize.intercept,
                    ", na.action=",
                    deparse(substitute(na.action))
                  )
                  
                  expect_true(
                    all(diff.threads < 1e-12),
                    info = paste(
                      "Serial vs Parallel mismatch:",
                      msg.base,
                      "abs(diff)=",
                      diff.threads
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
