\name{reg.table}
\alias{reg.table}
\title{Create Regression Tables in LaTeX or HTML}
\description{
  \code{reg.table} generates formatted regression tables in either LaTeX or HTML format for one or more linear regression models. 
  The tables include coefficient estimates, standard errors, and various model statistics.
}
\usage{
reg.table(models, type = c("latex", "html"), split.size = 4L, n.digits = 3L,
          big.mark = "", caption = "Regression Results", spacing = 5L,
          stats = c("AIC", "BIC", "CV", "r.squared", "adj.r.squared",
                    "fstatistic", "nobs"),
          ...)
}
\arguments{
  \item{models}{A list of linear regression models (objects of class \code{lm} or \code{glm} with gaussian family and identity link function).}
  \item{type}{A string specifying the output format. Must be one of \code{"latex"} or \code{"html"}.}
  \item{split.size}{An integer specifying the number of models per table. Tables with more models than this number will be split.}
  \item{n.digits}{An integer specifying the number of decimal places for numerical values.}
  \item{big.mark}{A string to use for thousands separators in numeric formatting.}
  \item{caption}{A string for the table caption.}
  \item{spacing}{A numeric specifying the spacing between columns in the output table.}
  \item{stats}{A character vector specifying which model statistics to include in the table. Options include:
    \describe{
      \item{\code{AIC}}{Akaike Information Criterion}
      \item{\code{BIC}}{Bayesian Information Criterion}
      \item{\code{CV}}{Cross-validation score}
      \item{\code{r.squared}}{R-squared}
      \item{\code{adj.r.squared}}{Adjusted R-squared}
      \item{\code{fstatistic}}{F-statistic}
      \item{\code{nobs}}{Number of observations}
    }
  }
  \item{...}{Additional arguments passed to \code{\link{cvLM}}.}
}
\details{
  The function generates tables summarizing linear regression models, either in LaTeX or HTML format. The tables include coefficients, standard errors, and optional statistics such as AIC, BIC, cross-validation scores, and more. The output is formatted based on the specified type and options. The names of the models list will be used as column headers if provided.
}
\value{
  A character vector of length equal to the number of tables with each element containing the LaTeX or HTML code for a regression table.
}
\examples{
data(mtcars)

# Create a list of 6 different linear regression models with names
# Names get used for column names of the table
models <- list(
  `Model 1` = lm(mpg ~ wt + hp, data = mtcars),
  `Model 2` = lm(mpg ~ wt + qsec, data = mtcars),
  `Model 3` = lm(mpg ~ wt + hp + qsec, data = mtcars),
  `Model 4` = lm(mpg ~ wt + cyl, data = mtcars),
  `Model 5` = lm(mpg ~ wt + hp + cyl, data = mtcars),
  `Model 6` = lm(mpg ~ hp + qsec + drat, data = mtcars)
)

# Example usage for LaTeX
reg.table(
  models,
  "latex",
  split.size = 3L,
  n.digits = 3L,
  big.mark = ",",
  caption = "My regression results",
  spacing = 7.5,
  stats = c("AIC", "BIC", "CV"),
  data = mtcars,
  K.vals = 5L,
  seed = 123L
)

# Example usage for HTML
reg.table(
  models,
  "html",
  split.size = 3L,
  n.digits = 3L,
  big.mark = ",",
  caption = "My regression results",
  spacing = 7.5,
  stats = c("AIC", "BIC", "CV"),
  data = mtcars,
  K.vals = 5L,
  seed = 123L
)
}