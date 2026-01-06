## model-stats.R: Helpers to retrieve model statistics
##
## This file is part of the cvLM package.

.get_fun <- function(FUN) {
  get(FUN, envir = asNamespace("cvLM"), mode = "function")
}
r.squared <- function(model) summary(model)$r.squared
adj.r.squared <- function(model) summary(model)$adj.r.squared
fstatistic <- function(model) summary(model)$fstatistic$value

.coef_df <- function(
  model.summary,
  n.digits,
  big.mark,
  type = c("latex", "html")
) {
  type <- match.arg(type)
  coefs <- coef(model.summary)
  predictors <- rownames(coefs)
  estimates <- .fmt(coefs[, "Estimate"], n.digits, big.mark)
  p.vals <- coefs[, "Pr(>|t|)"]
  p.stars <- rep_len("", length(p.vals))

  if (type == "latex") {
    p.stars[p.vals <= 0.1] <- "$^{*}$"
    p.stars[p.vals <= 0.05] <- "$^{**}$"
    p.stars[p.vals <= 0.01] <- "$^{***}$"
  } else if (type == "html") {
    p.stars[p.vals <= 0.1] <- "<sup>*</sup>"
    p.stars[p.vals <= 0.05] <- "<sup>**</sup>"
    p.stars[p.vals <= 0.01] <- "<sup>***</sup>"
  }

  estimates <- paste0(estimates, p.stars)
  std.err <- paste0("(", .fmt(coefs[, "Std. Error"], n.digits, big.mark), ")")

  data.frame(
    Predictor = c(rbind(predictors, sprintf("%s.std.err", predictors))),
    Value = c(rbind(estimates, std.err))
  )
}
