source("tests/helpers.R")

test_that("cvLM.formula, cvLM.lm, and cvLM.glm produce consistent results under varied seeds, sample sizes, and predictor counts", {
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
                  # Run formula interface
                  res.formula <- cvLM(
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

                  # Run lm interface
                  fit.lm <- lm(
                    dat$formula,
                    data = df,
                    na.action = na.action
                  )
                  
                  res.lm <- cvLM(
                    fit.lm,
                    data = df,
                    K.vals = K.vals,
                    lambda = lambda,
                    generalized = generalized,
                    seed = seed,
                    n.threads = 1,
                    penalize.intercept = penalize.intercept
                  )

                  # Run glm interface
                  fit.glm <- glm(
                    dat$formula,
                    data = df,
                    na.action = na.action
                  )
                  res.glm <- cvLM(
                    fit.glm,
                    data = df,
                    K.vals = K.vals,
                    lambda = lambda,
                    generalized = generalized,
                    seed = seed,
                    n.threads = 1,
                    penalize.intercept = penalize.intercept
                  )

                  # Compare CV results
                  diff.lm <- abs(res.formula$CV - res.lm$CV)
                  diff.glm <- abs(res.formula$CV - res.glm$CV)

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
                    all(diff.lm < 1e-12),
                    info = paste(
                      "Formula vs LM mismatch:",
                      msg.base,
                      "abs(diff)=",
                      diff.lm
                    )
                  )
                  expect_true(
                    all(diff.glm < 1e-12),
                    info = paste(
                      "Formula vs GLM mismatch:",
                      msg.base,
                      "abs(diff)=",
                      diff.glm
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
