# Baseline ----------------------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(cvLM)
  library(bench)
  library(data.table)
  library(RhpcBLASctl)
  library(boot)
  library(ggplot2)
})
blas_set_num_threads(1L)

# --- Helper: generate test data
make.data <- function(n, p, seed = 1L) {
  set.seed(seed)
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta <- rnorm(p)
  y <- drop(x %*% beta + rnorm(n))
  list(x = x, y = y)
}

# --- Benchmark scenarios
scenarios <- list(
  narrow.small = list(n = 1e3, p = 20),
  narrow.large = list(n = 1e4, p = 500),
  square = list(n = 2e3, p = 2e3),
  wide.small = list(n = 20, p = 1e3),
  wide.large = list(n = 500, p = 1e4)
)

# --- CV configurations
cv.configs <- list(
  k10 = list(k = 10L, generalized = FALSE),
  loocv = list(k = NA_integer_, generalized = FALSE),
  gcv = list(k = NA_integer_, generalized = TRUE)
)

# --- Benchmark runner
run.benchmarks <- function(
  lambda = 0.0,
  center = TRUE,
  tolerance = 1e-8,
  nThreads = 1L,
  seed = 123L
) {
  results <- list()

  for (sc in names(scenarios)) {
    cat("\n=== Scenario:", sc, "===\n")
    pars <- scenarios[[sc]]
    dat <- make.data(pars$n, pars$p, seed)

    for (cv in names(cv.configs)) {
      cfg <- cv.configs[[cv]]

      cat("  -> CV:", cv, "\n")

      res <- bench::mark(
        cvLM:::cv.lm.rcpp(
          X = dat$x,
          y = dat$y,
          k0 = if (is.na(cfg$k)) pars$n else cfg$k,
          lambda = lambda,
          generalized = cfg$generalized,
          seed = seed,
          nThreads = nThreads,
          tolerance = tolerance,
          center = center
        ),
        iterations = 10,
        check = FALSE,
        time_unit = "s",
        memory = FALSE
      )

      results[[paste(sc, cv, sep = ".")]] <- res
    }
  }

  DT <- rbindlist(results)
  DT[, id := names(results)]
  DT[,
    c(
      "expression",
      "mem_alloc",
      "gc/sec",
      "n_itr",
      "n_gc",
      "result",
      "memory",
      "time",
      "gc"
    ) := NULL
  ]
  DT[,
    names(.SD) := lapply(.SD, function(x) sprintf("%.5f", x)),
    .SDcols = is.numeric
  ]
  setcolorder(DT, "id")
  DT
}

DT <- run.benchmarks(nThreads = 10L)

# Compare to boot ---------------------------------------------------------------------------------------

# Scenarios where cv.glm is valid / meaningful
glm.scenarios <- c("narrow.small", "narrow.large")

compare.with.boot <- function(
  lambda = 0.0,
  nThreads = 1L,
  seed = 123L
) {
  res.list <- lapply(glm.scenarios, function(sc) {
    cat("\n=== Benchmarking scenario:", sc, "===\n")
    pars <- scenarios[[sc]]
    dat <- make.data(pars$n, pars$p, seed)
    df <- cbind(data.frame(y = dat$y), dat$x)
    fit <- glm(y ~ ., data = df)

    bm <- bench::mark(
      "boot::cv.glm" = {
        set.seed(seed)
        boot::cv.glm(fit, data = df, K = 10L)$delta[1L]
      },
      "cvLM" = cvLM:::cv.lm.rcpp(
        X = cbind(1, dat$x),
        y = dat$y,
        k0 = 10L,
        lambda = 0,
        generalized = FALSE,
        seed = seed,
        nThreads = nThreads,
        tolerance = fit$control$epsilon,
        center = FALSE
      ),
      iterations = 20,
      time_unit = "s"
    )

    bm.dt <- as.data.table(bm[, c("expression", "time")])
    bm.dt[, expression := as.character(expression)]
    bm.dt <- bm.dt[, .(time = unlist(time)), by = .(expression)]
    bm.dt[, scenario := sprintf("%d x %d", nrow(dat$x), ncol(dat$x))]
    bm.dt
  })

  rbindlist(res.list)
}

DT.compare <- compare.with.boot(nThreads = 10L)

ggplot(DT.compare, aes(x = expression, y = time, fill = expression)) +
  geom_boxplot(width = 0.1, outlier.shape = NA, color = "grey30") +
  geom_jitter(width = 0.1, alpha = 0.5, size = 0.5) +
  facet_wrap(~scenario, scales = "free_y") +
  scale_y_log10() +
  labs(
    title = "Runtime Distribution",
    subtitle = "Comparing cvLM vs boot::cv.glm across 20 iterations per scenario",
    y = "Execution Time (seconds, log scale)",
    x = "Method"
  ) +
  theme(legend.position = "none", strip.text = element_text(face = "bold"))
ggsave(file.path(dirname(rstudioapi::getSourceEditorContext()$path), "boot_comp.png"), scale = 1.2)
